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

import re
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


def _part_by_spine(score) -> dict:
    """Mappe l'index de spine → part music21. music21 inverse l'ordre de `.parts`
    à l'import Humdrum mais conserve l'index VRAI dans `part.id` (`spine_0` = colonne
    kern la plus à gauche = portée 1). Sert à attribuer la bonne part à chaque
    colonne **kern, pour N portées (pas seulement 2)."""
    out: dict = {}
    for p in score.parts:
        m = re.match(r"spine_(\d+)", str(getattr(p, "id", "")) or "")
        if m:
            out[int(m.group(1))] = p
    return out


def _cols_to_parts(score, kern_cols: list) -> dict:
    """Chaque colonne **kern (dans l'ordre gauche→droite) → sa part : la i-ème
    colonne **kern correspond à `spine_i`. Repli registre si pas d'id spine."""
    bs = _part_by_spine(score)
    if len(bs) >= len(kern_cols):
        return {ci: bs[i] for i, ci in enumerate(kern_cols) if i in bs}
    # Repli : ordonner les parts par registre décroissant (aigu = portée 1).
    parts = sorted(score.parts, key=_part_avg_pitch, reverse=True)
    return {ci: parts[i] for i, ci in enumerate(kern_cols) if i < len(parts)}


def _part_avg_pitch(part) -> float:
    ps = [n.pitch.midi for n in part.flatten().notes if hasattr(n, "pitch")]
    return sum(ps) / len(ps) if ps else 0.0


def _top_part(score):
    """Part de la portée du haut (staff1 = `spine_0`) pour ancrer les nuances /
    tempo globaux. Repli : registre moyen le plus aigu."""
    bs = _part_by_spine(score)
    if 0 in bs:
        return bs[0]
    parts = list(score.parts)
    return max(parts, key=_part_avg_pitch) if parts else None


def _first_real_dur(col: str) -> float:
    """Durée (quarter-notes) du 1er token-note réel d'une cellule kern (ignore
    silences/grace markers)."""
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


def _cell_dur(cell: str) -> float:
    """Durée (quarter-notes) du contenu d'une cellule kern — note OU silence (pour
    le suivi temporel). 0 pour `.`/grace (q/Q : pas d'avance de temps)."""
    for tok in cell.split():
        if not tok or tok == ".":
            continue
        if "q" in tok or "Q" in tok:   # grace note : n'avance pas le temps
            return 0.0
        d = _kern_dur_to_qn(tok)
        if d > 0:
            return d
    return 0.0


def _iter_kern_offsets(cleaned_text: str):
    """Parcourt le kern via le `_SpineTracker` (gère les splits `*^`/`*v`) et yield,
    par ligne : (kind, measure, line_offset, fields, dynam_cols, col_info).

    Modèle d'offset = reconstruction temporelle Humdrum correcte : on suit la FIN
    de la note courante de chaque colonne (`col_end`) et on avance l'offset de
    ligne au prochain onset = min des fins > offset courant. Les colonnes vides
    (`.` / voix silencieuse) n'ont PAS de fin → ne tirent pas l'offset à 0 (bug
    précédent : nuances toutes collées en début de mesure)."""
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent / "tokenizers"))  # added at import: tokenizers/ was not on sys.path
    from multihead_tokenizer import _SpineTracker
    tracker = _SpineTracker()
    measure = 0
    cur_off = 0.0
    col_end: list = []   # fin de la note courante par colonne (None = pas démarrée)

    def _reshape(fields):
        nonlocal col_end
        if "*^" in fields:
            new = []
            for i, f in enumerate(fields):
                e = col_end[i] if i < len(col_end) else None
                new.extend([e, e] if f.strip() == "*^" else [e])
            col_end = new
        elif "*v" in fields:
            new, i = [], 0
            while i < len(fields):
                e = col_end[i] if i < len(col_end) else None
                if fields[i].strip() == "*v":
                    j = i
                    while j < len(fields) and fields[j].strip() == "*v":
                        j += 1
                    new.append(e); i = j
                else:
                    new.append(e); i += 1
            col_end = new

    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("**"):
            tracker.init_from_exclusive(line.split("\t"))
            col_end = [None] * len(line.split("\t"))
            cur_off = 0.0
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num.isdigit():
                    measure = int(num)
            cur_off = 0.0
            col_end = [None] * len(col_end)
            continue
        if line.startswith("*"):
            fields = line.split("\t")
            yield ("interp", measure, round(cur_off, 4), fields, [], [], {})
            _reshape(fields)
            tracker.update(line)
            continue
        if not line or line.startswith("!"):
            continue
        fields = line.split("\t")
        loff = round(cur_off, 4)
        col_info: dict = {}
        for s in tracker.active_staves:
            for vi, ci in enumerate(tracker.staff_cols(s)):
                col_info[ci] = (s, vi)
        yield ("data", measure, loff, fields, list(tracker.dynam_cols),
               list(tracker.pedal_cols), col_info, tracker.dynam_cols_by_staff)
        # Mettre à jour la fin des colonnes qui ONSET sur cette ligne.
        for ci in tracker.kern_cols:
            if ci < len(fields) and ci < len(col_end):
                cell = fields[ci].strip()
                if cell and cell != ".":
                    d = _cell_dur(cell)
                    if d > 0:
                        col_end[ci] = round(cur_off + d, 6)
        # Avancer au prochain onset = plus petite fin strictement > offset courant.
        future = [e for e in col_end if e is not None and e > cur_off + 1e-6]
        if future:
            cur_off = min(future)


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

    if not list(score.parts):
        return 0

    # ── Colonnes **kern (toutes — slurs par portée, N portées) ─────────────
    #    NB : offset PAR COLONNE (chaque slur vit dans UNE colonne, qui avance
    #    correctement). On n'utilise pas l'offset système global ici : une voix
    #    silencieuse (`.`) tirerait le min à 0 et écraserait tous les offsets.
    kern_cols: list[int] = []
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("**"):
            kern_cols = [i for i, c in enumerate(line.split("\t")) if c == "**kern"]
            break
    if not kern_cols:
        return 0
    col_to_part = _cols_to_parts(score, kern_cols)
    if not col_to_part:
        return 0

    note_index: dict[tuple, list] = {}
    for part_obj in set(col_to_part.values()):
        for m in part_obj.getElementsByClass("Measure"):
            for el in m.flatten().notesAndRests:
                if isinstance(el, music21.note.Rest):
                    continue
                key = (id(part_obj), m.number, round(float(el.offset), 4))
                note_index.setdefault(key, []).append(el)

    current_measure = 0
    col_offset = {ci: 0.0 for ci in kern_cols}
    col_meas_start = {ci: 0.0 for ci in kern_cols}
    slur_events = {ci: [] for ci in kern_cols}
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num_str = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num_str.isdigit():
                    current_measure = int(num_str)
                    for ci in kern_cols:
                        col_meas_start[ci] = col_offset[ci]
            continue
        if line.startswith("*"):
            continue
        fields = line.split("\t")
        for ci in kern_cols:
            if ci >= len(fields):
                continue
            col = fields[ci].strip()
            if not col or col == ".":
                continue
            max_dur = 0.0
            for tok in col.split():
                stripped = re.sub(r"^[\[(_\{&]+", "", tok)
                if not stripped or re.match(r"[\d.qQ%]*r", stripped):
                    continue
                if not re.search(r"[a-gA-G]", stripped):
                    continue
                dur = _kern_dur_to_qn(tok)
                if dur > 0:
                    max_dur = max(max_dur, dur)
                off_in_m = round(col_offset[ci] - col_meas_start[ci], 4)
                if "(" in tok or ")" in tok:
                    slur_events[ci].append(
                        (current_measure, off_in_m, "(" in tok, ")" in tok))
            if max_dur > 0:
                col_offset[ci] = round(col_offset[ci] + max_dur, 6)

    n_slurs = 0
    for ci in kern_cols:
        part_obj = col_to_part.get(ci)
        if part_obj is None:
            continue
        pid = id(part_obj)
        stack: list = []
        for mnum, off, has_start, has_end in slur_events[ci]:
            candidates = note_index.get((pid, mnum, off), [])
            note_obj = candidates[0] if candidates else None
            # Slur_EndStart : fermer D'ABORD l'ancien slur, puis ouvrir le
            # nouveau. L'ordre inverse (push avant pop) crée une auto-boucle
            # (start_note is note_obj) et perd les deux slurs.
            if has_end and stack:
                start_note = stack.pop()
                if start_note is not None and note_obj is not None \
                        and start_note is not note_obj:
                    try:
                        score.insert(0, music21.spanner.Slur([start_note, note_obj]))
                        n_slurs += 1
                    except Exception:
                        pass
            if has_start:
                stack.append(note_obj)
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

    # ── Portée du haut (staff1 = spine_0) : on y ancre les nuances globales.
    #    Robuste à N portées (avant : max avg_pitch des 2 PREMIÈRES parts → faux
    #    pour 3+ portées où le haut peut être parts[2]).
    treble_part = _top_part(score)

    # ── parser via le tracker (suit les colonnes dynam par portée) ─────────
    # events = [(mnum, off, tok, part)] où part = l'objet Part music21 cible
    # (None = global = toutes les portées, pour les kern sans *staffN sur dynam).
    events: list[tuple[int, float, str, object]] = []
    parts_list = score.parts[:]
    for kind, mnum, off, fields, _dc_flat, _pc, _ci, dynam_by_staff in _iter_kern_offsets(cleaned_text):
        if kind != "data":
            continue
        if dynam_by_staff:
            # Mode per-staff : chaque colonne dynam porte un *staffN → portée ciblée.
            for sn, cols in dynam_by_staff.items():
                for dc in cols:
                    if dc < len(fields):
                        dv = fields[dc].strip()
                        if dv and dv != "." and (dv in _STATIC or dv in ("<", ">")):
                            target_part = None
                            for p in parts_list:
                                if p.id and f"spine_{sn - 1}" in p.id:
                                    target_part = p
                                    break
                            events.append((mnum, off, dv, target_part))
        else:
            # Mode global (ancien format kern sans *staffN sur les colonnes dynam).
            # Fallback : lire toutes les colonnes dynam plates, pas de staff cible
            # → insertion sur toutes les portées (comportement "add to all staves").
            for dc in _dc_flat:
                if dc < len(fields):
                    dv = fields[dc].strip()
                    if dv and dv != "." and (dv in _STATIC or dv in ("<", ">")):
                        events.append((mnum, off, dv, None))

    if not events:
        return 0

    # ── index notes par (measure, offset), UNE entrée PAR PORTÉE ────────
    # La colonne **dynam est globale (pas par portée) → on insère chaque
    # nuance/soufflet sur TOUTES les portées qui ont une note à cet offset,
    # sans heuristique de choix. Cela évite de perdre les hairpins de la LH.
    # note_by_part[(mnum, off)] = {part: first_note_at_that_position}
    note_by_part: dict[tuple, dict] = {}
    meas_by_num_per_part: dict[object, dict] = {}  # part → {mnum: measure}
    meas_by_num: dict[int, object] = {}             # pour nuances ponctuelles (top part)
    for part in score.parts:
        mbp: dict[int, object] = {}
        meas_by_num_per_part[id(part)] = mbp
        for m in part.getElementsByClass("Measure"):
            mbp[m.number] = m
            if part is treble_part:
                meas_by_num[m.number] = m
            for el in m.flatten().notesAndRests:
                if isinstance(el, music21.note.Rest):
                    continue
                off = round(float(el.offset), 4)
                key = (m.number, off)
                note_by_part.setdefault(key, {}).setdefault(id(part), el)

    # ── insertion des nuances et soufflets ───────────────────────────────
    n_added = 0
    for idx, (mnum, off, tok, target_part) in enumerate(events):
        # target_part = None → mode global, on insère sur toutes les portées.
        parts_for_event = [target_part] if target_part is not None else list(score.parts)
        if tok in _STATIC:
            for part in parts_for_event:
                m_obj = meas_by_num_per_part[id(part)].get(mnum)
                if m_obj is not None:
                    try:
                        m_obj.insert(off, music21.dynamics.Dynamic(tok))
                        n_added += 1
                    except Exception:
                        pass
        elif tok in ("<", ">"):
            cls = (music21.dynamics.Crescendo if tok == "<"
                   else music21.dynamics.Diminuendo)
            for part in parts_for_event:
                start_note = note_by_part.get((mnum, off), {}).get(id(part))
                if start_note is None:
                    continue
                part_notes = {(mn, o): el
                              for (mn, o), pd in note_by_part.items()
                              if id(part) in pd
                              for el in [pd[id(part)]]}
                end_note = None
                for _mnum2, _off2, _t2, _tp2 in events[idx + 1:]:
                    if _mnum2 != mnum:
                        break
                    cand = part_notes.get((_mnum2, _off2))
                    if cand is not None and cand is not start_note:
                        end_note = cand
                        break
                if end_note is None:
                    same = [el for (mn, _o), el in sorted(part_notes.items())
                            if mn == mnum]
                    end_note = same[-1] if same else start_note
                members = ([start_note] if end_note is start_note
                           else [start_note, end_note])
                try:
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

    # Portée du haut (staff1 = spine_0) : on y insère les TextExpression de tempo.
    treble_part = _top_part(score)

    # Parse via le tracker : un tandem *rall apparaît sur une ligne d'interp ;
    # son offset = position courante (offset système, robuste aux splits `*^`).
    events: list[tuple[int, float, str]] = []
    for kind, mnum, off, fields, _dc, _pc, _ci, _dsb in _iter_kern_offsets(cleaned_text):
        if kind != "interp":
            continue
        for f in fields:
            fs = f.strip()
            if fs in _INTERP_TO_TEXT:
                events.append((mnum, off, _INTERP_TO_TEXT[fs]))

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


def _rebuild_carac(score, cleaned_text: str) -> int:
    """Reconstruit les expressions de caractère H18 (*dolce, *espr, …) en
    TextExpression sur la portée du haut. music21 ignore ces tandems kern."""
    import music21

    _CARAC_TO_TEXT = {
        "*dolce":      "dolce",
        "*cantabile":  "cantabile",
        "*espr":       "espr.",
        "*leggiero":   "leggiero",
        "*marcato":    "marcato",
        "*sostenuto":  "sostenuto",
        "*tranquillo": "tranquillo",
        "*agitato":    "agitato",
        "*sottovoce":  "sotto voce",
        "*pesante":    "pesante",
    }

    treble_part = _top_part(score)
    events: list[tuple[int, float, str]] = []
    for kind, mnum, off, fields, _dc, _pc, _ci, _dsb in _iter_kern_offsets(cleaned_text):
        if kind != "interp":
            continue
        for f in fields:
            fs = f.strip()
            if fs in _CARAC_TO_TEXT:
                events.append((mnum, off, _CARAC_TO_TEXT[fs]))

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


def _rebuild_pedal(score, cleaned_text: str) -> int:
    """Reconstruit la pédale (kern `*ped`/`*Xped` en tandem, OU spine `**pedal`
    D/U/C) en directions textuelles `Ped.` / `*` sur la portée du haut. music21
    ne recrée pas la pédale depuis le kern → sinon elle disparaît du MXL."""
    import music21
    treble = _top_part(score)
    if treble is None:
        return 0
    events: list[tuple[int, float, str]] = []
    for kind, mnum, off, fields, _dc, pedal_cols, _ci, _dsb in _iter_kern_offsets(cleaned_text):
        if kind == "interp":
            for f in fields:
                fs = f.strip()
                if fs == "*ped":
                    events.append((mnum, off, "Ped."))
                elif fs == "*Xped":
                    events.append((mnum, off, "*"))
        else:
            for pc in pedal_cols:
                if pc < len(fields):
                    pv = fields[pc].strip()
                    if pv in ("D", "C"):
                        events.append((mnum, off, "Ped."))
                    elif pv == "U":
                        events.append((mnum, off, "*"))
    if not events:
        return 0
    meas_by_num = {m.number: m for m in treble.getElementsByClass("Measure")}
    n = 0
    for mnum, off, text in events:
        m = meas_by_num.get(mnum)
        if m is None:
            continue
        try:
            m.insert(off, music21.expressions.TextExpression(text))
            n += 1
        except Exception:
            pass
    return n


def _rebuild_ornaments(score, cleaned_text: str) -> int:
    """Reconstruit les ornements que music21 ne parse pas depuis kern.
    Actuellement : `s` (InvertedTurn) — les autres (T,t,M,m,W,w,S) sont gérés
    par le parser kern de music21. Retourne le nb d'ornements ajoutés."""
    import music21

    # kern char → music21 expression class (uniquement ceux perdus par music21)
    _MISSING = {"s": music21.expressions.InvertedTurn}

    kern_cols: list[int] = []
    for raw in cleaned_text.splitlines():
        if raw.startswith("**"):
            kern_cols = [i for i, c in enumerate(raw.split("\t")) if c == "**kern"]
            break
    if not kern_cols:
        return 0

    col_to_part = _cols_to_parts(score, kern_cols)
    note_index: dict = {}
    for part in set(col_to_part.values()):
        for m in part.getElementsByClass("Measure"):
            for el in m.flatten().notesAndRests:
                if isinstance(el, music21.note.Rest):
                    continue
                note_index.setdefault(
                    (id(part), m.number, round(float(el.offset), 4)), []).append(el)

    measure = 0
    col_off = {ci: 0.0 for ci in kern_cols}
    col_ms  = {ci: 0.0 for ci in kern_cols}
    n_added = 0
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num.isdigit():
                    measure = int(num)
                    for ci in kern_cols:
                        col_ms[ci] = col_off[ci]
            continue
        if line.startswith("*"):
            continue
        fields = line.split("\t")
        for ci in kern_cols:
            if ci >= len(fields):
                continue
            col = fields[ci].strip()
            if not col or col == ".":
                continue
            max_dur = 0.0
            for tok in col.split():
                stripped = re.sub(r"^[\[(_\{&]+", "", tok)
                if not stripped:
                    continue
                if re.match(r"[\d.qQ%]*r", stripped):
                    # Rest : occupe du temps, doit avancer col_off (sinon
                    # toutes les offsets suivantes de la mesure dérivent).
                    d = _kern_dur_to_qn(tok)
                    if d > max_dur:
                        max_dur = d
                    continue
                if not re.search(r"[a-gA-G]", stripped):
                    continue
                d = _kern_dur_to_qn(tok)
                if d > max_dur:
                    max_dur = d
                # Chercher un ornement manquant dans le token
                for orn_char, expr_cls in _MISSING.items():
                    if orn_char in stripped:
                        off_in_m = round(col_off[ci] - col_ms[ci], 4)
                        part = col_to_part.get(ci)
                        cands = note_index.get((id(part), measure, off_in_m), []) if part else []
                        target = cands[0] if cands else None
                        if target is not None and not any(
                                isinstance(e, expr_cls) for e in target.expressions):
                            try:
                                target.expressions.append(expr_cls())
                                n_added += 1
                            except Exception:
                                pass
            if max_dur > 0:
                col_off[ci] = round(col_off[ci] + max_dur, 6)
    return n_added


def _rebuild_beams(score, cleaned_text: str) -> int:
    """Reconstruit les ligatures (beams) NOUS-MÊMES depuis les marqueurs kern
    `L`/`LL`/`LLL` (début, n niveaux) et `J`/`JJ`/`JJJ` (fin), `K`/`k` (partiels),
    en posant directement les `music21.beam.Beams` — on contourne le parser kern
    de music21 qui corrompt le beaming non-standard (44 niveaux fantômes). On efface
    d'abord les beams lus, puis on reconstruit groupe par groupe et par colonne.
    Préserve le beaming EXACT de la source (ex. m32 : 8 doubles-croches sous une
    ligature). Retourne le nb de notes dont le beam a été posé."""
    import music21
    for n in score.recurse().notes:
        n.beams = music21.beam.Beams()

    kern_cols: list[int] = []
    for raw in cleaned_text.splitlines():
        if raw.startswith("**"):
            kern_cols = [i for i, c in enumerate(raw.split("\t")) if c == "**kern"]
            break
    if not kern_cols:
        return 0
    col_to_part = _cols_to_parts(score, kern_cols)
    if not col_to_part:
        return 0
    note_index: dict = {}
    for part in set(col_to_part.values()):
        for m in part.getElementsByClass("Measure"):
            for el in m.flatten().notesAndRests:
                if isinstance(el, music21.note.Rest):
                    continue
                note_index.setdefault(
                    (id(part), m.number, round(float(el.offset), 4)), []).append(el)

    measure = 0
    col_off = {ci: 0.0 for ci in kern_cols}
    col_ms  = {ci: 0.0 for ci in kern_cols}
    open_lv = {ci: 0   for ci in kern_cols}   # niveaux de beam ouverts par colonne
    n_set = 0
    for raw in cleaned_text.splitlines():
        line = raw.rstrip("\r\n")
        if not line or line.startswith("!") or line.startswith("**"):
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                num = line.split("\t", 1)[0].lstrip("=").rstrip("|")
                if num.isdigit():
                    measure = int(num)
                    for ci in kern_cols:
                        col_ms[ci] = col_off[ci]
            for ci in kern_cols:
                open_lv[ci] = 0   # une ligature ne franchit pas la barre
            continue
        if line.startswith("*"):
            continue
        fields = line.split("\t")
        for ci in kern_cols:
            if ci >= len(fields):
                continue
            col = fields[ci].strip()
            if not col or col == ".":
                continue
            off_in_m = round(col_off[ci] - col_ms[ci], 4)
            max_dur = 0.0
            beam_tok = None
            for tok in col.split():
                stripped = re.sub(r"^[\[(_\{&]+", "", tok)
                if not stripped:
                    continue
                if re.match(r"[\d.qQ%]*r", stripped):
                    d = _kern_dur_to_qn(tok)
                    if d > max_dur:
                        max_dur = d
                    continue
                if not re.search(r"[a-gA-G]", stripped):
                    continue
                d = _kern_dur_to_qn(tok)
                if d > max_dur:
                    max_dur = d
                if beam_tok is None:
                    beam_tok = tok
            if beam_tok is not None:
                n_L = beam_tok.count("L"); n_J = beam_tok.count("J")
                has_K = "K" in beam_tok; has_k = "k" in beam_tok
                part = col_to_part.get(ci)
                cands = note_index.get((id(part), measure, off_in_m), []) if part else []
                target = cands[0] if cands else None
                if target is not None:
                    beams = music21.beam.Beams()
                    if n_L:
                        for _ in range(min(n_L, 3)):
                            beams.append("start")
                        open_lv[ci] = min(n_L, 3)
                    elif n_J:
                        for _ in range(open_lv[ci]):
                            beams.append("stop")
                        open_lv[ci] = 0
                    elif has_K or has_k:
                        beams.append("partial", "right" if has_K else "left")
                    elif open_lv[ci] > 0:
                        for _ in range(open_lv[ci]):
                            beams.append("continue")
                    if beams.beamsList:
                        try:
                            target.beams = beams
                            n_set += 1
                        except Exception:
                            pass
            if max_dur > 0:
                col_off[ci] = round(col_off[ci] + max_dur, 6)
    return n_set


def _merge_voice_parts(score, cleaned_text: str) -> int:
    """Fusionne les parts music21 qui partagent le même `*staff` en UNE part à
    plusieurs voix. Le postprocess émet les voix intra-portée comme des colonnes
    `**kern` séparées (même `*staffN`) → music21 en fait des parts distinctes
    (portée fantôme). On déplace les notes (pas de copie) → les slurs/nuances déjà
    posées (qui référencent ces notes) restent valides. Retourne le nb de parts
    fusionnées."""
    import music21
    from collections import OrderedDict
    kc: list[int] = []
    cs: dict[int, int] = {}
    for raw in cleaned_text.splitlines():
        if raw.startswith("**"):
            kc = [i for i, c in enumerate(raw.split("\t")) if c == "**kern"]
        elif raw.startswith("*") and "staff" in raw:
            f = raw.split("\t")
            for ci in kc:
                if ci < len(f):
                    m = re.match(r"\*staff(\d+)", f[ci])
                    if m:
                        cs[ci] = int(m.group(1))
            if cs:
                break
    if not kc:
        return 0
    bs = _part_by_spine(score)
    groups: "OrderedDict[int, list]" = OrderedDict()
    for i, ci in enumerate(kc):
        sn = cs.get(ci, i + 1)
        part = bs.get(i)
        if part is not None:
            groups.setdefault(sn, []).append(part)

    n_merged = 0
    for sn, plist in groups.items():
        if len(plist) < 2:
            continue
        base = plist[0]
        base_meas = {m.number: m for m in base.getElementsByClass("Measure")}
        vnum = 2
        for extra in plist[1:]:
            for me in extra.getElementsByClass("Measure"):
                mb = base_meas.get(me.number)
                if mb is None:
                    continue
                # 1re fusion : envelopper les notes nues de base dans la voix 1.
                if not mb.getElementsByClass("Voice"):
                    v1 = music21.stream.Voice(id="1")
                    for el in list(mb.notesAndRests):
                        off = el.offset
                        mb.remove(el)
                        v1.insert(off, el)
                    mb.insert(0, v1)
                v = music21.stream.Voice(id=str(vnum))
                for el in list(me.notesAndRests):
                    off = el.offset
                    me.remove(el)
                    v.insert(off, el)
                # N'insérer la voix que si elle contient au moins une vraie note
                # (pas uniquement des silences de remplissage). Les silences seuls
                # dans une voix secondaire s'affichent comme des rectangles dans
                # MuseScore — on préfère les supprimer silencieusement.
                if any(isinstance(el, music21.note.Note) or
                       (isinstance(el, music21.chord.Chord) and el.notes)
                       for el in v.notesAndRests):
                    mb.insert(0, v)
            vnum += 1
            score.remove(extra)
            n_merged += 1
    return n_merged


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

    # Restaurer tous les spanners/ornements AVANT merge et swap (offsets sur l'ordre kern).
    t_starts  = _collect_t_starts(score)
    n_tx      = _add_trill_extensions(score, t_starts, tr_ends)
    n_slurs   = _rebuild_slurs(score, cleaned_text)
    n_dyn     = _rebuild_dynamics(score, cleaned_text)
    n_tempo   = _rebuild_tempo_text(score, cleaned_text)
    n_carac   = _rebuild_carac(score, cleaned_text)
    n_ped     = _rebuild_pedal(score, cleaned_text)
    n_orns    = _rebuild_ornaments(score, cleaned_text)
    # Beaming AVANT merge : col_to_part est valide (parts pas encore réorganisées).
    n_beams   = _rebuild_beams(score, cleaned_text)
    # Recoller les voix intra-portée (colonnes **kern de même *staff) en UNE part
    # à plusieurs voix — sinon music21 sort une portée fantôme par voix.
    _merge_voice_parts(score, cleaned_text)

    # Ordre des portées : music21 inverse la liste `.parts` à l'import Humdrum,
    # mais préserve l'ordre VRAI des spines dans `part.id` (`spine_0` = colonne
    # kern la plus à gauche = portée 1, telle qu'émise par mxl_to_kern). On relit
    # cet index plutôt que de DEVINER par registre — le tri par registre se
    # trompe quand une portée basse passe au-dessus d'une haute (cas Liszt :
    # staff2 a un registre moyen > staff1). Repli registre si l'id n'est pas
    # parseable (fichiers d'origine non issus de notre convertisseur).
    parts = list(score.parts)
    if len(parts) >= 2:
        idxs = [re.match(r"spine_(\d+)", str(p.id) or "") for p in parts]
        if all(m is not None for m in idxs):
            ordered = [p for _, p in sorted(
                zip((int(m.group(1)) for m in idxs), parts), key=lambda t: t[0])]
        else:
            def _avg_ps(part):
                ps = [x.ps for n in part.recurse().notes for x in n.pitches]
                return sum(ps) / len(ps) if ps else 0.0
            ordered = sorted(parts, key=_avg_ps, reverse=True)   # aigu d'abord
        if [id(p) for p in ordered] != [id(p) for p in parts]:
            for p in parts:
                score.remove(p)
            # score.insert(0, p) ajoute à la fin du conteneur (offset 0 = temps,
            # pas un index) → l'ordre de .parts suit l'ordre des appels.
            for p in ordered:
                score.insert(0, p)

    # Anacrouse : marquer la 1re mesure partielle comme `implicit="yes"` (sinon
    # music21 écrit `implicit="no"` → les éditeurs la rendent en mesure normale
    # numérotée et l'anacrouse « disparaît »). implicit ⇔ showNumber=NEVER.
    try:
        never = music21.stream.enums.ShowNumber.NEVER
        for part in score.parts:
            ms = list(part.getElementsByClass("Measure"))
            if not ms:
                continue
            m0 = ms[0]
            is_pickup = (float(getattr(m0, "paddingLeft", 0)) > 0
                         or float(m0.barDuration.quarterLength)
                            > float(m0.duration.quarterLength) + 1e-6)
            if is_pickup:
                m0.showNumber = never
    except Exception:
        pass

    score.write("musicxml", str(dst_mxl))
    return {
        "notes":                    len(list(score.flatten().notes)),
        "trill_extensions_added":   n_tx,
        "slurs_added":              n_slurs,
        "dynamics_added":           n_dyn,
        "tempo_text_added":         n_tempo,
        "carac_added":              n_carac,
        "pedal_added":              n_ped,
        "ornaments_added":          n_orns,
        "beams_set":                n_beams,
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
