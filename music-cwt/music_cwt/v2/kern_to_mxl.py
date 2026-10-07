"""
kern_to_mxl.py — Convertit un fichier .krn (notre format v2) en MusicXML, en
restaurant les éléments que music21 ne parse pas nativement depuis kern :

  * TrillExtension : on a marqué la note de fin par `Tr]` ; après le parse on
    re-crée un spanner music21 entre la note T et la note Tr].
  * Inversion staff1 / staff2 (convention Humdrum vs convention MuseScore).

Usage :
    python kern_to_mxl.py input.krn [output.mxl]

Ou en bibliothèque :
    from kern_to_mxl import kern_file_to_mxl
    kern_file_to_mxl(Path("recon.krn"), Path("recon.mxl"))
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _strip_tr_end(kern_text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Enlève les marqueurs `Tr]` du texte kern (qui perturbent music21) et
    retourne (staff_idx, measure_num) pour chaque marker rencontré.
    Le matching par numéro de mesure est plus robuste qu'un matching par
    index de note.
    """
    new_lines: list[str] = []
    tr_end_at: list[tuple[int, int]] = []
    current_measure = 0

    for raw in kern_text.splitlines():
        line = raw.rstrip("\r\n")
        # Suivre le numéro de mesure
        if line.startswith("=") and not line.startswith("=="):
            try:
                # ex. "=15\t=15\t=15"
                first_col = line.split("\t", 1)[0]
                num_str   = first_col.lstrip("=").rstrip("|")
                if num_str.isdigit():
                    current_measure = int(num_str)
            except Exception:
                pass
            new_lines.append(line)
            continue
        # Lignes de structure
        if (not line) or line.startswith("!") or line.startswith("**") \
           or line.startswith("*") or line.startswith("="):
            new_lines.append(line)
            continue

        cols = line.split("\t")
        new_cols = []
        for staff_idx, col in enumerate(cols[:2]):
            tok = col.strip()
            if not tok or tok == ".":
                new_cols.append(col)
                continue
            if "Tr]" in tok:
                tr_end_at.append((staff_idx, current_measure))
            new_cols.append(" ".join(n.replace("Tr]", "") for n in tok.split(" ")))
        if len(cols) > 2:
            new_cols.extend(cols[2:])
        new_lines.append("\t".join(new_cols))

    return "\n".join(new_lines), tr_end_at


def _collect_t_starts(score) -> list[tuple[int, int, object]]:
    """
    Pour chaque note avec une expression Trill, retourne (staff_idx, measure_num, note_obj).
    """
    import music21
    starts: list[tuple[int, int, object]] = []
    for staff_idx, part in enumerate(score.parts[:2]):
        for n in part.flatten().notesAndRests:
            for e in getattr(n, "expressions", []):
                if isinstance(e, music21.expressions.Trill):
                    m = n.getContextByClass("Measure")
                    starts.append((staff_idx, m.number if m else -1, n))
                    break
    return starts


def _last_note_in_measure(score, staff_idx: int, measure_num: int):
    """Retourne la dernière note de la mesure donnée dans le staff donné, ou None."""
    if staff_idx >= len(score.parts):
        return None
    for m in score.parts[staff_idx].getElementsByClass("Measure"):
        if m.number == measure_num:
            notes = [n for n in m.flatten().notesAndRests
                     if hasattr(n, "pitch") or hasattr(n, "pitches")]
            return notes[-1] if notes else None
    return None


def _add_trill_extensions(score, t_starts: list[tuple[int, int, object]],
                          tr_ends: list[tuple[int, int]]) -> int:
    """
    Pour chaque note avec une expression Trill, ajoute un spanner TrillExtension
    pour faire apparaître la ligne ondulée dans le MXL.

    Si un marqueur `Tr]` existe dans le même staff à une mesure ≥ celle de la
    note T, le spanner s'étend jusqu'à la dernière note de cette mesure
    (trille « long » sur plusieurs notes). Sinon, le spanner couvre uniquement
    la note T elle-même (trille « court » sur une seule note longue tenue).
    """
    import music21

    # Mapper les staff_idx kern (0=RH, 1=LH) vers les indices music21 (parts).
    # Heuristique : déterminer quelle part contient les notes les plus aiguës.
    # On suppose 2 parts ; si la part 0 a un range moyen plus haut que la part 1,
    # alors part[0] = RH et le mapping kern_staff[0] = part_idx[0].
    if len(score.parts) >= 2:
        def _avg_pitch(part):
            ps = [n.pitch.midi for n in part.flatten().notes if hasattr(n, "pitch")]
            return sum(ps) / len(ps) if ps else 0
        if _avg_pitch(score.parts[0]) >= _avg_pitch(score.parts[1]):
            staff_to_part = {0: 0, 1: 1}   # part[0] = RH = kern staff 0
        else:
            staff_to_part = {0: 1, 1: 0}   # part[0] = LH (cas Humdrum standard)
    else:
        staff_to_part = {0: 0}

    # Index des Tr] par (part_idx, measure)
    tr_by_part: dict[int, list[int]] = {}
    for s, m in tr_ends:
        p = staff_to_part.get(s)
        if p is not None:
            tr_by_part.setdefault(p, []).append(m)
    used_ends: set[tuple[int, int]] = set()

    n_added = 0
    for staff_idx, t_meas, start_note in t_starts:
        # Chercher un Tr] dans le MÊME part à une mesure ≥ t_meas (proche)
        end_note = start_note
        candidates = sorted(m for m in tr_by_part.get(staff_idx, [])
                            if m >= t_meas and (staff_idx, m) not in used_ends)
        if candidates:
            m_end = candidates[0]
            cand = _last_note_in_measure(score, staff_idx, m_end)
            if cand is not None:
                end_note = cand
                used_ends.add((staff_idx, m_end))

        try:
            tx = music21.expressions.TrillExtension([start_note, end_note]
                                                    if end_note is not start_note
                                                    else [start_note])
            score.insert(0, tx)
            n_added += 1
        except Exception:
            pass
    return n_added


def _kern_dur_to_qn(tok: str) -> float:
    """Kern note token → durée en quarter-notes (0 pour les grace notes)."""
    import re
    t = re.sub(r"^[\[(_\{&]+", "", tok)   # strip prefix markers
    m = re.match(r"(\d+)(\.{0,3})", t)
    if not m:
        return 0.0
    base = int(m.group(1))
    if base == 0:
        return 0.0
    # Grace note (q ou Q juste après les chiffres+points)
    rest_str = t[m.end():]
    if rest_str[:1] in ("q", "Q"):
        return 0.0
    dots = len(m.group(2))
    dur = 4.0 / base
    multiplier = 2.0 - (1.0 / (2 ** dots)) if dots else 1.0
    return dur * multiplier


def _rebuild_slurs(score, cleaned_text: str) -> int:
    """
    music21 ne crée aucun Slur depuis les marqueurs ( ) kern.
    Cette fonction re-parse le kern, calcule les offsets des notes par mesure,
    identifie les notes portant ( ou ), les associe aux objets music21
    par (part, mesure, offset), et crée les music21.spanner.Slur manquants.
    Retourne le nombre de slurs ajoutés.
    """
    import re
    import music21

    if len(score.parts) < 2:
        return 0

    # ── Identifier treble vs bass part ────────────────────────────────────
    def _avg_pitch(part):
        ps = [n.pitch.midi for n in part.flatten().notes if hasattr(n, "pitch")]
        return sum(ps) / len(ps) if ps else 0

    if _avg_pitch(score.parts[0]) >= _avg_pitch(score.parts[1]):
        treble_part, bass_part = score.parts[0], score.parts[1]
    else:
        treble_part, bass_part = score.parts[1], score.parts[0]

    # ── Déterminer quelle colonne kern est staff1 (treble) / staff2 (bass) ─
    kern_cols: list[int] = []
    col_staff: dict[int, int] = {}

    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("**"):
            fields = line.split("\t")
            kern_cols = [i for i, c in enumerate(fields) if c == "**kern"]
        elif line.startswith("*") and "staff" in line:
            fields = line.split("\t")
            for ci in kern_cols:
                if ci < len(fields) and re.match(r"\*staff\d", fields[ci]):
                    try:
                        col_staff[ci] = int(fields[ci][6:])
                    except Exception:
                        pass
            if col_staff:
                break

    if len(kern_cols) < 2:
        return 0

    col_to_part: dict[int, object] = {}
    for ci in kern_cols[:2]:
        sn = col_staff.get(ci)
        col_to_part[ci] = treble_part if sn == 1 else bass_part
    if not col_to_part:
        col_to_part = {kern_cols[0]: bass_part, kern_cols[1]: treble_part}

    # ── Index music21 : (id(part), measure_num, offset) → liste de notes ─
    note_index: dict[tuple, list] = {}
    for part_obj in (treble_part, bass_part):
        for m in part_obj.getElementsByClass("Measure"):
            mnum = m.number
            for el in m.flatten().notesAndRests:
                if isinstance(el, music21.note.Rest):
                    continue
                off = round(float(el.offset), 4)
                key = (id(part_obj), mnum, off)
                note_index.setdefault(key, []).append(el)

    # ── Parser le kern pour les événements slur par colonne ───────────────
    current_measure  = 0
    col_offset: dict[int, float]              = {ci: 0.0 for ci in kern_cols}
    col_meas_start:  dict[int, float]         = {ci: 0.0 for ci in kern_cols}
    slur_events:     dict[int, list]          = {ci: []  for ci in kern_cols[:2]}

    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                try:
                    num_str = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                    if num_str.isdigit():
                        current_measure = int(num_str)
                        for ci in kern_cols:
                            col_meas_start[ci] = col_offset[ci]
                except Exception:
                    pass
            continue
        if line.startswith("*"):
            continue

        fields = line.split("\t")
        for ci in kern_cols[:2]:
            if ci >= len(fields):
                continue
            col = fields[ci].strip()
            if not col or col == ".":
                continue

            max_dur = 0.0
            for tok in col.split():
                if tok == "." or not tok:
                    continue
                # Ignorer les silences
                stripped = re.sub(r"^[\[(_\{&]+", "", tok)
                if not stripped:
                    continue
                if re.match(r"[\d.qQ%]*r", stripped):
                    continue
                if not re.search(r"[a-gA-G]", stripped):
                    continue

                dur = _kern_dur_to_qn(tok)
                if dur > 0:
                    max_dur = max(max_dur, dur)

                off_in_m = round(col_offset[ci] - col_meas_start[ci], 4)
                has_start = "(" in tok
                has_end   = ")" in tok
                if has_start or has_end:
                    slur_events[ci].append(
                        (current_measure, off_in_m, has_start, has_end)
                    )

            if max_dur > 0:
                col_offset[ci] = round(col_offset[ci] + max_dur, 6)

    # ── Stack-match ( avec ) et créer les Slur spanners ───────────────────
    n_slurs = 0
    for ci in kern_cols[:2]:
        if ci not in col_to_part:
            continue
        part_obj = col_to_part[ci]
        pid      = id(part_obj)
        stack: list[object] = []

        for mnum, off, has_start, has_end in slur_events[ci]:
            key = (pid, mnum, off)
            candidates = note_index.get(key, [])
            note_obj = candidates[0] if candidates else None

            # IMPORTANT : on push/pop TOUJOURS, même si la note n'a pas été
            # retrouvée (note_obj is None). Sinon une note manquante désaligne
            # la pile et un `)` finit par se marier avec un `(` d'une mesure
            # antérieure → slur fantôme inter-mesures (cf. bug mesure 29).
            if has_start:
                stack.append(note_obj)
            if has_end and stack:
                start_note = stack.pop()
                if start_note is not None and note_obj is not None \
                        and start_note is not note_obj:
                    try:
                        slur = music21.spanner.Slur([start_note, note_obj])
                        score.insert(0, slur)
                        n_slurs += 1
                    except Exception:
                        pass

    return n_slurs


def _rebuild_dynamics(score, cleaned_text: str) -> int:
    """
    music21 ne parse pas le spine **dynam depuis kern. Cette fonction le
    re-parse, identifie les nuances (p, f, ff…) et les soufflets (< / >),
    et insère les music21.dynamics.Dynamic / Crescendo / Diminuendo manquants.

    L'offset de chaque ligne kern est reconstruit comme dans `_rebuild_slurs`
    (offset = min des offsets RH/LH avant la ligne).
    Retourne le nombre d'objets de nuance ajoutés.
    """
    import re
    import music21

    if len(score.parts) < 2:
        return 0

    _STATIC = {"pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff",
               "ffff", "fp", "fz", "sfz", "sfp"}

    # ── treble vs bass ────────────────────────────────────────────────────
    def _avg_pitch(part):
        ps = [n.pitch.midi for n in part.flatten().notes if hasattr(n, "pitch")]
        return sum(ps) / len(ps) if ps else 0

    if _avg_pitch(score.parts[0]) >= _avg_pitch(score.parts[1]):
        treble_part = score.parts[0]
    else:
        treble_part = score.parts[1]

    # ── colonnes kern (+ staff) et colonne dynam ─────────────────────────
    kern_cols: list[int] = []
    dynam_col: int | None = None
    col_staff: dict[int, int] = {}
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("**"):
            fields = line.split("\t")
            kern_cols = [i for i, c in enumerate(fields) if c == "**kern"]
            for i, c in enumerate(fields):
                if c == "**dynam":
                    dynam_col = i
        elif line.startswith("*") and "staff" in line:
            fields = line.split("\t")
            for ci in kern_cols:
                if ci < len(fields) and re.match(r"\*staff\d", fields[ci]):
                    try:
                        col_staff[ci] = int(fields[ci][6:])
                    except Exception:
                        pass
            if col_staff:
                break

    if dynam_col is None or len(kern_cols) < 2:
        return 0

    rh_col = next((ci for ci in kern_cols if col_staff.get(ci) == 1), kern_cols[0])
    lh_col = next((ci for ci in kern_cols if col_staff.get(ci) == 2), kern_cols[1])

    # ── 1er token-note réel d'une colonne → durée en quarter-notes ───────
    def _first_real_dur(col: str) -> float:
        for tok in col.split():
            if not tok or tok == ".":
                continue
            stripped = re.sub(r"^[\[(_\{&]+", "", tok)
            if not stripped or re.match(r"[\d.qQ%]*r", stripped):
                continue
            if not re.search(r"[a-gA-G]", stripped):
                continue
            d = _kern_dur_to_qn(tok)
            if d > 0:
                return d
        return 0.0

    # ── parser : collecter (measure, offset_dans_mesure, token) ──────────
    current_measure = 0
    rh_off = lh_off = 0.0
    events: list[tuple[int, float, str]] = []

    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num_str = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num_str.isdigit():
                    current_measure = int(num_str)
            rh_off = lh_off = 0.0
            continue
        if line.startswith("*"):
            continue

        fields = line.split("\t")
        rh_c = fields[rh_col].strip()    if rh_col    < len(fields) else "."
        lh_c = fields[lh_col].strip()    if lh_col    < len(fields) else "."
        dy_c = fields[dynam_col].strip() if dynam_col < len(fields) else "."

        line_off = min(rh_off, lh_off)
        if dy_c and dy_c != ".":
            events.append((current_measure, round(line_off, 4), dy_c))

        rh_d = _first_real_dur(rh_c) if (rh_c and rh_c != ".") else 0.0
        lh_d = _first_real_dur(lh_c) if (lh_c and lh_c != ".") else 0.0
        if rh_off <= line_off + 1e-6 and rh_d > 0:
            rh_off = round(rh_off + rh_d, 6)
        if lh_off <= line_off + 1e-6 and lh_d > 0:
            lh_off = round(lh_off + lh_d, 6)

    if not events:
        return 0

    # ── index note (treble) par (measure, offset) ────────────────────────
    note_index: dict[tuple, object] = {}
    meas_by_num: dict[int, object] = {}
    for m in treble_part.getElementsByClass("Measure"):
        meas_by_num[m.number] = m
        for el in m.flatten().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            off = round(float(el.offset), 4)
            note_index.setdefault((m.number, off), el)

    # ── insertion des nuances et soufflets ───────────────────────────────
    n_added = 0
    for idx, (mnum, off, tok) in enumerate(events):
        if tok in _STATIC:
            m_obj = meas_by_num.get(mnum)
            if m_obj is not None:
                try:
                    m_obj.insert(off, music21.dynamics.Dynamic(tok))
                    n_added += 1
                except Exception:
                    pass
        elif tok in ("<", ">"):
            start_note = note_index.get((mnum, off))
            if start_note is None:
                continue
            # Fin du soufflet : prochain événement dynam DANS LA MÊME MESURE ;
            # sinon le soufflet s'arrête à la dernière note de sa mesure de
            # départ. Un soufflet ne franchit jamais une barre dans ce
            # répertoire — l'étendre à un événement de la mesure suivante
            # donnait des hairpins traversant les barres (cf. m4, m12, m20…).
            end_note = None
            for mnum2, off2, _t2 in events[idx + 1:]:
                if mnum2 != mnum:
                    break
                cand = note_index.get((mnum2, off2))
                if cand is not None and cand is not start_note:
                    end_note = cand
                    break
            if end_note is None:
                same = [el for (mn, _o), el in sorted(note_index.items())
                        if mn == mnum]
                end_note = same[-1] if same else start_note
            # Soufflet sur une seule note (le `<` est sur la dernière note de
            # la mesure) : spanner mono-note, valide dans music21.
            members = ([start_note] if end_note is start_note
                       else [start_note, end_note])
            try:
                cls = (music21.dynamics.Crescendo if tok == "<"
                       else music21.dynamics.Diminuendo)
                score.insert(0, cls(members))
                n_added += 1
            except Exception:
                pass

    return n_added


def _rebuild_tempo_text(score, cleaned_text: str) -> int:
    """
    music21 ignore les interprétations tandem *rall / *accel / *Atempo / *rubato
    (directions de tempo, head H14). Cette fonction les re-parse et insère les
    music21.expressions.TextExpression correspondantes à la bonne mesure/offset.
    """
    import re
    import music21

    if len(score.parts) < 2:
        return 0

    _INTERP_TO_TEXT = {
        "*rall":        "rall.",
        "*pocorall":    "poco rall.",
        "*moltorall":   "molto rall.",
        "*accel":       "accel.",
        "*pocoaccel":   "poco accel.",
        "*moltoaccel":  "molto accel.",
        "*Atempo":      "a tempo",
        "*rubato":      "rubato",
        "*pocorubato":  "poco rubato",
        "*moltorubato": "molto rubato",
    }

    # Part aiguë (on y insère les TextExpression).
    def _avg_pitch(part):
        ps = [n.pitch.midi for n in part.flatten().notes if hasattr(n, "pitch")]
        return sum(ps) / len(ps) if ps else 0
    treble_part = (score.parts[0]
                   if _avg_pitch(score.parts[0]) >= _avg_pitch(score.parts[1])
                   else score.parts[1])

    # Colonnes kern (pour suivre l'offset des lignes).
    kern_cols: list[int] = []
    col_staff: dict[int, int] = {}
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("**"):
            kern_cols = [i for i, c in enumerate(line.split("\t")) if c == "**kern"]
        elif line.startswith("*") and "staff" in line:
            fields = line.split("\t")
            for ci in kern_cols:
                if ci < len(fields) and re.match(r"\*staff\d", fields[ci]):
                    try:
                        col_staff[ci] = int(fields[ci][6:])
                    except Exception:
                        pass
            if col_staff:
                break
    if len(kern_cols) < 2:
        return 0
    rh_col = next((ci for ci in kern_cols if col_staff.get(ci) == 1), kern_cols[0])
    lh_col = next((ci for ci in kern_cols if col_staff.get(ci) == 2), kern_cols[1])

    def _first_real_dur(col: str) -> float:
        for tok in col.split():
            if not tok or tok == ".":
                continue
            stripped = re.sub(r"^[\[(_\{&]+", "", tok)
            if not stripped or re.match(r"[\d.qQ%]*r", stripped):
                continue
            if not re.search(r"[a-gA-G]", stripped):
                continue
            d = _kern_dur_to_qn(tok)
            if d > 0:
                return d
        return 0.0

    # Parse : un tandem *rall apparaît ENTRE deux lignes de notes ; son offset
    # est la position courante du spine (min des offsets RH/LH).
    current_measure = 0
    rh_off = lh_off = 0.0
    events: list[tuple[int, float, str]] = []

    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num_str = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num_str.isdigit():
                    current_measure = int(num_str)
            rh_off = lh_off = 0.0
            continue
        if line.startswith("*"):
            for f in line.split("\t"):
                fs = f.strip()
                if fs in _INTERP_TO_TEXT:
                    events.append((current_measure,
                                   round(min(rh_off, lh_off), 4),
                                   _INTERP_TO_TEXT[fs]))
            continue

        fields = line.split("\t")
        rh_c = fields[rh_col].strip() if rh_col < len(fields) else "."
        lh_c = fields[lh_col].strip() if lh_col < len(fields) else "."
        line_off = min(rh_off, lh_off)
        rh_d = _first_real_dur(rh_c) if (rh_c and rh_c != ".") else 0.0
        lh_d = _first_real_dur(lh_c) if (lh_c and lh_c != ".") else 0.0
        if rh_off <= line_off + 1e-6 and rh_d > 0:
            rh_off = round(rh_off + rh_d, 6)
        if lh_off <= line_off + 1e-6 and lh_d > 0:
            lh_off = round(lh_off + lh_d, 6)

    if not events:
        return 0

    meas_by_num = {m.number: m for m in treble_part.getElementsByClass("Measure")}
    n_added = 0
    for mnum, off, text in events:
        m_obj = meas_by_num.get(mnum)
        if m_obj is None:
            continue
        try:
            m_obj.insert(off, music21.expressions.TextExpression(text))
            n_added += 1
        except Exception:
            pass
    return n_added


def kern_file_to_mxl(src_krn: Path, dst_mxl: Path) -> dict:
    """
    Pipeline complète : lit src_krn, restaure les TrillExtension, écrit dst_mxl.
    Retourne un dict de stats (notes, trill_extensions_added, …).
    """
    import music21

    raw_text = src_krn.read_text(encoding="utf-8")
    cleaned_text, tr_ends = _strip_tr_end(raw_text)

    # On écrit un fichier temporaire « clean » pour le donner à music21
    tmp_krn = src_krn.with_suffix(".clean.krn")
    tmp_krn.write_text(cleaned_text, encoding="utf-8")
    try:
        score = music21.converter.parse(str(tmp_krn))
    finally:
        try:
            tmp_krn.unlink()
        except Exception:
            pass

    # Restaurer TrillExtension et Slur AVANT le swap (offsets calculés sur l'ordre kern).
    t_starts = _collect_t_starts(score)
    n_tx     = _add_trill_extensions(score, t_starts, tr_ends)
    n_slurs  = _rebuild_slurs(score, cleaned_text)
    n_dyn    = _rebuild_dynamics(score, cleaned_text)
    n_tempo  = _rebuild_tempo_text(score, cleaned_text)

    # Inversion staff1 ↔ staff2 (Humdrum convention vs MuseScore)
    parts = list(score.parts)
    if len(parts) == 2:
        score.remove(parts[0])
        score.remove(parts[1])
        score.insert(0, parts[1])
        score.insert(0, parts[0])

    score.write("musicxml", str(dst_mxl))
    return {
        "notes":                    len(list(score.flatten().notes)),
        "trill_extensions_added":   n_tx,
        "slurs_added":              n_slurs,
        "dynamics_added":           n_dyn,
        "tempo_text_added":         n_tempo,
        "trill_starts_in_kern":     len(t_starts),
        "trill_ends_in_kern":       len(tr_ends),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input.krn> [output.mxl]")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".mxl")

    stats = kern_file_to_mxl(src, dst)
    print(f"OK  {src.name}  ->  {dst.name}")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
