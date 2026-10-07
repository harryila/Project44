"""
postprocess.py — Convertit une séquence de CompoundTokens en fichier **kern valide.

Étapes :
  1. Groupement des tokens en timesteps (RH + LH simultanés)
  2. Extraction des notes de grace (Fioritura_Q / q) avant la note principale
  3. Reconstruction des chaînes kern depuis les 12 heads
  4. Insertion des signes naturels (tracker par mesure)
  5. Fusion des accords (Chord_Start → … → Chord_End → "4c 4e 4g")
  6. Numérotation séquentielle des barlines
  7. Reconstruction de l'en-tête Humdrum

Usage :
    # Round-trip test (tokenize → reconstruct)
    python postprocess.py kern_data/Chopin_op10n01.krn [output.krn]

    # Depuis Python
    from postprocess import tokens_to_kern, kern_file
    kern_str = tokens_to_kern(score)
    kern_file(score, Path("output.krn"))
"""

import re
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import (CompoundToken, MetaPrefix, TokenizedScore,  # noqa: E402
                                 _atomic_chord_pitches, _atomic_chord_decoration)

# H14 (Tempo) → tandem interpretation kern
_H14_TO_KERN: dict[str, str] = {
    "Rall":         "*rall",
    "Poco_Rall":    "*pocorall",
    "Molto_Rall":   "*moltorall",
    "Accel":        "*accel",
    "Poco_Accel":   "*pocoaccel",
    "Molto_Accel":  "*moltoaccel",
    "A_Tempo":      "*Atempo",
    "Rubato":       "*rubato",
    "Poco_Rubato":  "*pocorubato",
    "Molto_Rubato": "*moltorubato",
    "Calando":      "*calando",
    "Poco_Calando": "*pococal",
}

# H18 (Caractère/Expression) → tandem interpretation kern
_H18_TO_KERN: dict[str, str] = {
    "Dolce":      "*dolce",
    "Cantabile":  "*cantabile",
    "Espr":       "*espr",
    "Leggiero":   "*leggiero",
    "Marcato_P":  "*marcato",
    "Sostenuto":  "*sostenuto",
    "Tranquillo": "*tranquillo",
    "Agitato":    "*agitato",
    "Sotto_Voce": "*sottovoce",
    "Pesante":    "*pesante",
}

# ── v3 — expansion des accords atomiques (epellation partagee avec le tokeniseur) ──
def _atomic_chord_to_kern(tok: "CompoundToken", spine: str,
                          tracker: "AccidentalTracker") -> str:
    """Chord_Atomic -> token kern espace ('4c 4e- 4g'). L'epellation est calculee
    par le tokeniseur (_atomic_chord_pitches), partagee pour garantir l'idempotence
    et la verification lossless a l'atomisation."""
    pitches = _atomic_chord_pitches(tok.h15, tok.h16, tok.h3)
    if not pitches:
        return "."
    dur = tok.h2 if tok.h2 not in ("NULL", "") else "4"
    members = [dur + tracker.process(ps, spine) for ps in pitches]
    if tok.h9 == "/":
        members = [m + "/" for m in members]
    elif tok.h9 == "\\":
        members = [m + "\\" for m in members]
    # H4/H5 : decoration commune re-emise sur CHAQUE membre (relaxation 2026-06).
    suffix = _atomic_chord_decoration(tok.h4, tok.h5)
    if suffix:
        members = [m + suffix for m in members]
    # H10 : beam mis sur le PREMIER membre uniquement (convention kern).
    if tok.h10 not in ("NULL", "", None):
        members[0] = members[0] + tok.h10
    return " ".join(members)

# ── Types ──────────────────────────────────────────────────────────────────────

# Un timestep est :
#   - "BARLINE"                                                 (sentinelle)
#   - ("CLEF", stave_num, kern_spec)  ex. ("CLEF", 1, "G2")     (clef mid-piece)
#   - ("EVENT", {stave_num: [CompoundToken, ...]})              (\xe9venement musical)
# stave_num est 1-based (Staff_1 = 1, Staff_2 = 2, ...).
Timestep = Union[str, tuple]


def _staff_to_num(h11: str) -> int:
    """Convertit 'Staff_1'..'Staff_4' -> 1..4. Tolere les alias legacy 'RH'/'LH'."""
    if h11 == "RH":
        return 1
    if h11 == "LH":
        return 2
    if h11.startswith("Staff_"):
        try:
            return int(h11.split("_", 1)[1])
        except (ValueError, IndexError):
            return 1
    return 1


def _voice_to_num(h17: str) -> int:
    """Convertit 'V1'..'V3' -> 1..3. NULL / inconnu -> 1."""
    if h17 and h17.startswith("V"):
        try:
            return int(h17[1:])
        except ValueError:
            return 1
    return 1


# ── Tracker d'accidentels ──────────────────────────────────────────────────────

class AccidentalTracker:
    """
    Suit l'état des accidentels actifs par classe de hauteur (a-g) et par voix.
    Réinitialise à chaque barline.
    Si le modèle prédit `f` alors que `f#` était actif dans la mesure → ajoute `n`.
    """

    def __init__(self) -> None:
        # v3 \xe2\x80\x94 un dict imbrique par stave (string), cree \xe0 la demande.
        self._state: dict[str, dict[str, str]] = {}

    def reset(self) -> None:
        self._state = {}

    def process(self, pitch: str, spine: str) -> str:
        """Retourne le pitch kern \xe0 ecrire (avec `n` si signe naturel requis)."""
        if not pitch or pitch in ("r", "rr", "NULL"):
            return pitch

        m = re.match(r"^([A-Ga-g]+)([#\-]{0,2})$", pitch)
        if not m:
            return pitch

        letter_key = m.group(1)[0].lower()
        acc        = m.group(2)
        per_spine  = self._state.setdefault(spine, {})
        active     = per_spine.get(letter_key, "")

        result = pitch
        if acc == "" and active != "":
            result = pitch + "n"

        per_spine[letter_key] = acc
        return result


# ── Token → chaîne kern ────────────────────────────────────────────────────────

def _token_to_str(tok: CompoundToken, spine: str, tracker: AccidentalTracker) -> str:
    """Reconstruit la chaîne kern d'un seul CompoundToken."""
    if tok.h1 in ("Barline", "SEP", "Sustain"):
        return ""

    # v3 — Chord_Atomic : un seul token represente un accord complet.
    # On delegue directement a l'expansion (qui produit une string '4c 4e- 4g').
    if tok.h1 == "Chord_Atomic":
        return _atomic_chord_to_kern(tok, spine, tracker)

    parts: list[str] = []
    is_rest = tok.h1 == "Rest"

    # ── Préfixes (interdits sur les rests) ────────────────────────────────
    if not is_rest:
        if tok.h8 == "&Phrase_Start":
            parts.append("&{")
        elif tok.h8 == "Phrase_Start":
            parts.append("{")

        if tok.h6 == "Tie_Start":
            parts.append("[")
        # Note : Tie_Continue (`_`) est un suffixe en kern, pas un préfixe.

        if tok.h7 in ("Slur_Start", "Slur_EndStart"):
            parts.append("(")

    # ── Durée + hauteur ────────────────────────────────────────────────────
    dur = tok.h2 if tok.h2 not in ("NULL", "") else "4"

    if is_rest:
        pitch = tok.h3 if tok.h3 in ("r", "rr") else "r"
        parts.append(dur + pitch)

    elif tok.h1 == "Fioritura_Q":
        # Appoggiature : kern `<dur>Q<pitch>`
        raw   = tok.h3 if tok.h3 not in ("NULL", "") else "c"
        pitch = tracker.process(raw, spine)
        parts.append(dur + "Q" + pitch)

    elif tok.h1 == "Fioritura_q":
        # Acciaccatura : kern `<dur>q<pitch>` (slashed grace)
        raw   = tok.h3 if tok.h3 not in ("NULL", "") else "c"
        pitch = tracker.process(raw, spine)
        parts.append(dur + "q" + pitch)

    else:
        raw   = tok.h3 if tok.h3 not in ("NULL", "") else "c"
        pitch = tracker.process(raw, spine)
        parts.append(dur + pitch)

    # ── Ornements (interdits sur les rests) ────────────────────────────────
    if not is_rest and tok.h4 not in ("NULL", ""):
        parts.append(tok.h4)

    # ── Articulations ──────────────────────────────────────────────────────
    if tok.h5 not in ("NULL", ""):
        parts.append(tok.h5)

    # ── Voicing ────────────────────────────────────────────────────────────
    if tok.h9 == "/":
        parts.append("/")
    elif tok.h9 == "\\":
        parts.append("\\")

    # ── Beam (interdit sur les rests) ─────────────────────────────────────
    if not is_rest and tok.h10 not in ("NULL", ""):
        parts.append(tok.h10)

    # ── Suffixes fermants (interdits sur les rests) ────────────────────────
    if not is_rest:
        if tok.h6 == "Tie_End":
            parts.append("]")
        elif tok.h6 == "Tie_Continue":
            parts.append("_")

        if tok.h7 in ("Slur_End", "Slur_EndStart"):
            parts.append(")")

        if tok.h8 == "&Phrase_End":
            parts.append("&}")
        elif tok.h8 == "Phrase_End":
            parts.append("}")

    return "".join(parts)


def _group_to_kern(tokens: list[CompoundToken], spine: str,
                   tracker: AccidentalTracker) -> str:
    """
    Convertit une liste de CompoundTokens (accord ou note simple) en chaîne kern.
    Les membres d'accord sont séparés par des espaces.
    """
    if not tokens:
        return "."
    parts = [_token_to_str(t, spine, tracker) for t in tokens]
    parts = [p for p in parts if p]
    return " ".join(parts) if parts else "."


# ── Groupement par timestep ────────────────────────────────────────────────────

_KERN_EVENT_END = frozenset({
    "Note_Single", "Chord_End", "Chord_Atomic",
    "Rest", "Fioritura_Q", "Fioritura_q",
    "Sustain",   # v3 Phase 2 — l'ancre Staff_1/V1 qui tient ; groupe d'un seul token
})

# Tokens that act like a Barline (segment timestep boundary, no audio event)
_NON_DATA_H1 = frozenset({"Barline", "SEP", "Clef_G2", "Clef_F4", "Clef_G1"})


def _group_timesteps(tokens: list[CompoundToken]) -> list[Timestep]:
    """
    Regroupe la s\xe9quence lineaire en timesteps multi-staff.

    Chaque timestep est :
      - "BARLINE"
      - ("CLEF", stave_num, spec)
      - ("EVENT", {stave_num: [tokens]})   un \xe9venement musical par stave

    Heuristique : on parcourt les tokens lin\xe9airement, on collecte les
    kern events par stave dans l'ordre d'apparition jusqu'\xe0 ce qu'on
    rencontre un SEP (separateur de ligne), un Barline ou une Clef change,
    ou jusqu'\xe0 ce qu'un m\xeame stave reapparaisse (== nouvelle ligne implicite).
    """
    timesteps: list[Timestep] = []
    i = 0
    n = len(tokens)

    def _collect_event(start: int) -> tuple[list[CompoundToken], int]:
        """Collecte UN kern event sur une (portee, voix) donnee, jusqu'a fin d'event."""
        event: list[CompoundToken] = []
        spine = tokens[start].h11
        voice = tokens[start].h17
        j = start
        while j < n:
            tok = tokens[j]
            if tok.h1 in _NON_DATA_H1 or tok.h11 != spine or tok.h17 != voice:
                break
            event.append(tok)
            j += 1
            if tok.h1 in _KERN_EVENT_END:
                break
        return event, j

    while i < n:
        tok = tokens[i]

        if tok.h1 == "Barline":
            timesteps.append("BARLINE")
            i += 1
            continue
        if tok.h1 == "SEP":
            i += 1
            continue
        if tok.h1 in ("Clef_G2", "Clef_F4", "Clef_G1"):
            spec = tok.h1.split("_", 1)[1]
            timesteps.append(("CLEF", _staff_to_num(tok.h11), spec))
            i += 1
            continue

        # \xc9venement musical : collecter un kern event par stave, ordre d'apparition.
        per_sv: dict[tuple[int, int], list[CompoundToken]] = {}
        while i < n:
            tok = tokens[i]
            if tok.h1 in _NON_DATA_H1:
                break
            key = (_staff_to_num(tok.h11), _voice_to_num(tok.h17))
            if key in per_sv:
                # m\xeame stave reapparait \xe2\x86\x92 nouvelle ligne
                break
            event, i = _collect_event(i)
            if event:
                per_sv[key] = event

        if per_sv:
            timesteps.append(("EVENT", per_sv))

    return timesteps


# ── Extraction des notes de grace ──────────────────────────────────────────────

def _split_grace(
    tokens: list[CompoundToken],
) -> tuple[list[CompoundToken], list[CompoundToken]]:
    """Sépare notes de grace / fioritura des notes principales."""
    grace     = [t for t in tokens if t.h1 in ("Fioritura_Q", "Fioritura_q")]
    principal = [t for t in tokens if t not in grace]
    return grace, principal


# ── En-tête Humdrum ────────────────────────────────────────────────────────────

def _fmt_key(key_sig: str) -> str:
    """'K_f#c#_maj' → '*k[f#c#]'"""
    parts = key_sig.split("_")
    notes = parts[1] if len(parts) >= 2 else "0"
    return "*k[]" if notes == "0" else f"*k[{notes}]"


def _fmt_ts(time_sig: str) -> str:
    """'M_3/4' → '*M3/4'"""
    return "*" + time_sig.replace("M_", "M", 1)


def _fmt_tempo(tempo: str) -> str:
    """'MM_72' → '*MM72'"""
    return "*" + tempo.replace("MM_", "MM", 1)


def _meta_clef_to_kern(clef: str) -> str:
    return "*clefG2" if clef == "CLEF_G" else (
           "*clefF4" if clef == "CLEF_F" else
           "*clefG2")


def _build_header(meta: MetaPrefix, layout: list,
                  with_dynam: bool = True,
                  with_pedal: bool = False) -> list:
    """v3 - entete pour un layout de colonnes [(stave, voice), ...].

    Chaque colonne = un **kern avec *staff{stave}. Les voix multiples d'une meme
    portee = plusieurs **kern partageant le meme *staff (convention Humdrum
    multi-spine, relue correctement par le tokeniseur via staff_cols)."""
    def clef_of(stave: int) -> str:
        idx = stave - 1
        return meta.clefs[idx] if 0 <= idx < len(meta.clefs) else "CLEF_G"

    key   = _fmt_key(meta.key_sig)
    ts    = _fmt_ts(meta.time_sig)
    tempo = _fmt_tempo(meta.tempo)

    cols_excl = ["**kern"] * len(layout)
    cols_st   = [f"*staff{s}" for (s, v) in layout]
    cols_clef = [_meta_clef_to_kern(clef_of(s)) for (s, v) in layout]
    cols_key  = [key]   * len(layout)
    cols_ts   = [ts]    * len(layout)
    cols_tmp  = [tempo] * len(layout)

    if with_dynam:
        staves = sorted({s for (s, _v) in layout})
        for s in staves:
            cols_excl.append("**dynam")
            cols_st.append(f"*staff{s}")
            for c in (cols_clef, cols_key, cols_ts, cols_tmp):
                c.append("*")
    if with_pedal:
        cols_excl.append("**pedal")
        for c in (cols_st, cols_clef, cols_key, cols_ts, cols_tmp):
            c.append("*")

    staves = sorted({s for (s, v) in layout})
    deco = "!!!system-decoration: " + ",".join(f"s{s}" for s in staves)

    return [
        deco,
        "\t".join(cols_excl),
        "\t".join(cols_st),
        "\t".join(cols_clef),
        "\t".join(cols_key),
        "\t".join(cols_ts),
        "\t".join(cols_tmp),
    ]


# -- Conversion principale -----------------------------------------------------

def tokens_to_kern(score: TokenizedScore) -> str:
    """Convertit un TokenizedScore en **kern valide, multi-portees / multi-voix.

    Layout = ensemble des (portee, voix) rencontrees ; chaque (portee, voix) ->
    une colonne **kern fixe (avec *staff{portee}). Une voix absente d'une ligne
    recoit '.'. Dynam/pedale/tempo proviennent de l'ancre (portee 1, voix 1)."""
    timesteps = _group_timesteps(score.tokens)

    has_dyn = any(t.h12 not in ("NULL", "") for t in score.tokens)
    has_ped = any(getattr(t, "h13", "NULL") not in ("NULL", "") for t in score.tokens)

    # Layout : toutes les (portee, voix) vues, capees a 4 portees.
    seen: set = set()
    for ts in timesteps:
        if isinstance(ts, tuple) and ts and ts[0] == "EVENT":
            seen.update(ts[1].keys())
    layout = sorted((s, v) for (s, v) in seen if 1 <= s <= 4)
    if not layout:
        layout = [(1, 1), (2, 1)]

    col_index = {sv: i for i, sv in enumerate(layout)}
    n_cols    = len(layout)
    staves    = sorted({s for (s, _v) in layout})   # portées uniques dans l'ordre
    n_staves  = len(staves)

    tracker     = AccidentalTracker()
    lines       = _build_header(score.meta, layout, with_dynam=has_dyn, with_pedal=has_ped)
    measure_num = 1

    def _extras(dyn_vals: "list[str]", ped: str = ".") -> list:
        e = []
        if has_dyn: e.extend(dyn_vals)
        if has_ped: e.append(ped)
        return e

    def _line(cells: list, dyn_vals: "list[str] | None" = None, ped: str = ".") -> str:
        if dyn_vals is None:
            dyn_vals = ["."] * n_staves
        return "\t".join(cells + _extras(dyn_vals, ped))

    def _bar_line(num: int) -> str:
        cols = [f"={num}"] * n_cols
        if has_dyn:
            for _ in range(n_staves): cols.append(f"={num}")
        if has_ped: cols.append(f"={num}")
        return "\t".join(cols)

    def _tandem_at_stave(stave: int, mark: str) -> str:
        cols = ["*"] * n_cols
        # marquer uniquement la 1ere colonne (voix la plus basse) de la portee
        scols = sorted(i for (s, v), i in col_index.items() if s == stave)
        if scols:
            cols[scols[0]] = mark
        if has_dyn: cols.append("*")
        if has_ped: cols.append("*")
        return "\t".join(cols)

    for ts in timesteps:
        if ts == "BARLINE":
            tracker.reset()
            lines.append(_bar_line(measure_num))
            measure_num += 1
            continue

        if isinstance(ts, tuple) and len(ts) == 3 and ts[0] == "CLEF":
            _, stave_num, spec = ts
            lines.append(_tandem_at_stave(stave_num, f"*clef{spec}"))
            continue

        if isinstance(ts, tuple) and len(ts) == 2 and ts[0] == "EVENT":
            per_sv: dict = ts[1]
            anchor = per_sv.get((1, 1), [])

            # H12 event-based, par portée : l'ancre de chaque portée (stave, V1)
            # porte la dynamique de cette portée. On émet N valeurs (une par portée).
            dyn_by_stave: "dict[int, str]" = {}
            for s in staves:
                s_anchor = per_sv.get((s, 1), [])
                for t in s_anchor:
                    if t.h12 not in ("NULL", ""):
                        dyn_by_stave[s] = t.h12
                        break
            dyn_vals_main = [dyn_by_stave.get(s, ".") for s in staves]

            ped_val = "."
            for t in anchor:
                h13 = getattr(t, "h13", "NULL")
                if h13 == "Ped_Down":   ped_val = "D"; break
                if h13 == "Ped_Up":     ped_val = "U"; break
                if h13 == "Ped_Change": ped_val = "C"; break
                if h13 == "Una_Corda":
                    lines.append(_tandem_at_stave(1, "*unacorda")); break
                if h13 == "Tre_Corde":
                    lines.append(_tandem_at_stave(1, "*trecorde")); break

            for t in anchor:
                h14 = getattr(t, "h14", "NULL")
                if h14 in _H14_TO_KERN:
                    lines.append(_tandem_at_stave(1, _H14_TO_KERN[h14]))
                    break

            for t in anchor:
                h18 = getattr(t, "h18", "NULL")
                if h18 in _H18_TO_KERN:
                    lines.append(_tandem_at_stave(1, _H18_TO_KERN[h18]))
                    break

            grace_cells = ["."] * n_cols
            main_cells  = ["."] * n_cols
            has_grace   = False
            for sv in layout:
                toks = per_sv.get(sv)
                if not toks:
                    continue
                if toks[0].h1 == "Sustain":
                    continue   # cette (portee, voix) tient -> '.' (cellule par defaut)
                i     = col_index[sv]
                spine = f"Staff_{sv[0]}"
                grace, main = _split_grace(toks)
                if grace:
                    has_grace = True
                    grace_cells[i] = _group_to_kern(grace, spine, tracker)
                if main:
                    main_cells[i] = _group_to_kern(main, spine, tracker)

            has_main = any(c != "." for c in main_cells)
            if has_grace:
                g_dyn = ["."] * n_staves if has_main else dyn_vals_main
                g_ped = "." if has_main else ped_val
                lines.append(_line(grace_cells, g_dyn, g_ped))
            if has_main:
                lines.append(_line(main_cells, dyn_vals_main, ped_val))
            continue

    final_cols = ["=||"] * n_cols
    term_cols  = ["*-"]  * n_cols
    if has_dyn: final_cols.append("=||"); term_cols.append("*-")
    if has_ped: final_cols.append("=||"); term_cols.append("*-")
    final_line = "\t".join(final_cols)
    # Si la derniere ligne est deja une barline, la remplacer (evite l'accumulation
    # d'une barline finale a chaque round-trip) ; sinon l'ajouter.
    if lines and lines[-1].split("\t")[0].startswith("="):
        lines[-1] = final_line
    else:
        lines.append(final_line)
    lines.append("\t".join(term_cols))

    return "\n".join(lines)


def kern_file(score: TokenizedScore, output_path: Path) -> None:
    """Écrit un TokenizedScore dans un fichier .krn."""
    content = tokens_to_kern(score)
    output_path.write_text(content, encoding="utf-8")
    print(f"Écrit : {output_path}  ({len(content.splitlines())} lignes)")


# ── CLI — test round-trip ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from multihead_tokenizer import MultiHeadTokenizer

    if len(sys.argv) < 2:
        print(f"Usage: python postprocess.py <kern_file> [output.krn]")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".reconstructed.krn")

    tok   = MultiHeadTokenizer()
    score = tok.tokenize_file(src)

    print(f"Source     : {src}")
    print(f"Tokens     : {len(score.tokens)}")
    print(f"Destination: {dst}")
    print()

    kern_file(score, dst)

    print()
    print("=== Aperçu (30 premières lignes) ===")
    for i, line in enumerate(dst.read_text(encoding="utf-8").splitlines()[:30]):
        print(f"  {i + 1:3d}  {line}")
