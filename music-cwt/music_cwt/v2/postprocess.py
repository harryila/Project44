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
from multihead_tokenizer import CompoundToken, MetaPrefix, TokenizedScore  # noqa: E402

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
}

# ── Types ──────────────────────────────────────────────────────────────────────

# Un timestep est :
#   - "BARLINE"                                                 (string sentinelle)
#   - ("CLEF", h11, kern_spec)  ex. ("CLEF", "RH", "G2")        (clef en milieu de pièce)
#   - (rh_tokens, lh_tokens)                                   (événement musical)
Timestep = Union[str, tuple]


# ── Tracker d'accidentels ──────────────────────────────────────────────────────

class AccidentalTracker:
    """
    Suit l'état des accidentels actifs par classe de hauteur (a-g) et par voix.
    Réinitialise à chaque barline.
    Si le modèle prédit `f` alors que `f#` était actif dans la mesure → ajoute `n`.
    """

    def __init__(self) -> None:
        self._state: dict[str, dict[str, str]] = {"RH": {}, "LH": {}}

    def reset(self) -> None:
        self._state = {"RH": {}, "LH": {}}

    def process(self, pitch: str, spine: str) -> str:
        """Retourne le pitch kern à écrire (avec `n` si signe naturel requis)."""
        if not pitch or pitch in ("r", "rr", "NULL"):
            return pitch

        m = re.match(r"^([A-Ga-g]+)([#\-]{0,2})$", pitch)
        if not m:
            return pitch

        letter_key = m.group(1)[0].lower()
        acc        = m.group(2)            # "", "#", "##", "-", "--"
        active     = self._state[spine].get(letter_key, "")

        result = pitch
        if acc == "" and active != "":
            result = pitch + "n"

        self._state[spine][letter_key] = acc
        return result


# ── Token → chaîne kern ────────────────────────────────────────────────────────

def _token_to_str(tok: CompoundToken, spine: str, tracker: AccidentalTracker) -> str:
    """Reconstruit la chaîne kern d'un seul CompoundToken."""
    if tok.h1 in ("Barline", "SEP"):
        return ""

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

        if tok.h7 == "Slur_Start":
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

        if tok.h7 == "Slur_End":
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

_KERN_EVENT_END = frozenset({"Note_Single", "Chord_End", "Rest", "Fioritura_Q", "Fioritura_q"})

# Tokens that act like a Barline (segment timestep boundary, no audio event)
_NON_DATA_H1 = frozenset({"Barline", "SEP", "Clef_G2", "Clef_F4", "Clef_G1"})


def _group_timesteps(tokens: list[CompoundToken]) -> list[Timestep]:
    """
    Regroupe la séquence linéaire de CompoundTokens en timesteps.
    Chaque timestep est soit "BARLINE" soit (rh_tokens, lh_tokens).

    Un "kern event" = Note_Single | Chord_Start…Chord_End | Rest | Fioritura_Q.
    Chaque paire (un kern event RH + zéro ou un kern event LH) = un timestep.
    Les RH consécutifs sans LH génèrent chacun leur propre ligne (avec "." pour LH).
    """
    timesteps: list[Timestep] = []
    i = 0
    n = len(tokens)

    def _collect_event(start: int, spine: str) -> tuple[list[CompoundToken], int]:
        """Collecte un kern event complet (jusqu'à Note_Single/Chord_End/Rest/Fioritura_*)."""
        event: list[CompoundToken] = []
        j = start
        while j < n:
            tok = tokens[j]
            if tok.h1 in _NON_DATA_H1 or tok.h11 != spine:
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
        elif tok.h1 == "SEP":
            i += 1
        elif tok.h1 in ("Clef_G2", "Clef_F4", "Clef_G1"):
            # Mid-piece clef change → emit as ("CLEF", h11, "G2"/"F4"/"G1")
            spec = tok.h1.split("_", 1)[1]   # "Clef_G2" → "G2"
            timesteps.append(("CLEF", tok.h11, spec))
            i += 1
        elif tok.h11 == "RH":
            rh_event, i = _collect_event(i, "RH")
            # Chercher un kern event LH immédiatement après
            lh_event: list[CompoundToken] = []
            if i < n and tokens[i].h11 == "LH" and tokens[i].h1 not in _NON_DATA_H1:
                lh_event, i = _collect_event(i, "LH")
            timesteps.append((rh_event, lh_event))
        elif tok.h11 == "LH":
            # LH orphelin (pas de RH précédent)
            lh_event, i = _collect_event(i, "LH")
            timesteps.append(([], lh_event))
        else:
            i += 1

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


def _build_header(meta: MetaPrefix, with_dynam: bool = True,
                  with_pedal: bool = False) -> list[str]:
    clef_rh = "*clefG2" if meta.clef_rh == "CLEF_G" else "*clefF4"
    clef_lh = "*clefF4" if meta.clef_lh == "CLEF_F" else "*clefG2"
    key     = _fmt_key(meta.key_sig)
    ts      = _fmt_ts(meta.time_sig)
    tempo   = _fmt_tempo(meta.tempo)

    # Décoration de système : accolade {(s1,s2)} = piano à 2 portées (par défaut)
    bracket = getattr(meta, "bracket", "BRACKET_PIANO")
    deco = "!!!system-decoration: {(s1,s2)}" if bracket == "BRACKET_PIANO" else "!!!system-decoration: s1,s2"

    # Construction colonne par colonne pour gérer toutes les combinaisons
    cols_excl = ["**kern", "**kern"]
    cols_st   = ["*staff1", "*staff2"]
    cols_clef = [clef_rh,  clef_lh]
    cols_key  = [key,      key]
    cols_ts   = [ts,       ts]
    cols_tmp  = [tempo,    tempo]

    if with_dynam:
        cols_excl.append("**dynam")
        for c in (cols_st, cols_clef, cols_key, cols_ts, cols_tmp):
            c.append("*")
    if with_pedal:
        cols_excl.append("**pedal")
        for c in (cols_st, cols_clef, cols_key, cols_ts, cols_tmp):
            c.append("*")

    return [
        deco,
        "\t".join(cols_excl),
        "\t".join(cols_st),
        "\t".join(cols_clef),
        "\t".join(cols_key),
        "\t".join(cols_ts),
        "\t".join(cols_tmp),
    ]


# ── Conversion principale ──────────────────────────────────────────────────────

def tokens_to_kern(score: TokenizedScore) -> str:
    """
    Convertit un TokenizedScore en chaîne **kern valide (jusqu'à 4 spines :
    `**kern`, `**kern`, optionnellement `**dynam`, optionnellement `**pedal`).

    - Header Humdrum depuis MetaPrefix (et `!!!system-decoration` selon meta.bracket)
    - Parcours des timesteps :
        * "BARLINE"               → ligne `=N\\t=N\\t...`, reset accidentels
        * ("CLEF", h11, spec)     → ligne tandem `*clefG2\\t*\\t...`
        * (rh_toks, lh_toks)      → grace notes puis ligne principale
        * H12 du premier RH       → spine `**dynam` (sticky, on ne réémet qu'en cas de changement)
        * H13 du premier RH       → spine `**pedal` (Ped_Down=`D`, Ped_Up=`U`, Ped_Change=`C`)
    - Footer `=||` puis `*-`
    """
    timesteps = _group_timesteps(score.tokens)
    tracker   = AccidentalTracker()

    has_dyn = any(t.h12 not in ("NULL", "") for t in score.tokens)
    has_ped = any(getattr(t, "h13", "NULL") not in ("NULL", "") for t in score.tokens)

    # Nombre de colonnes par ligne
    n_extra = (1 if has_dyn else 0) + (1 if has_ped else 0)
    extra_dot = "\t.".join([""] * (n_extra + 1))   # "" if n_extra=0 else "\t.\t."[:…]

    lines = _build_header(score.meta, with_dynam=has_dyn, with_pedal=has_ped)

    last_dyn     = "NULL"
    measure_num  = 1

    def _line(rh: str, lh: str, dyn: str = ".", ped: str = ".") -> str:
        cols = [rh, lh]
        if has_dyn: cols.append(dyn)
        if has_ped: cols.append(ped)
        return "\t".join(cols)

    def _bar_line(num: int) -> str:
        cols = [f"={num}", f"={num}"]
        if has_dyn: cols.append(f"={num}")
        if has_ped: cols.append(f"={num}")
        return "\t".join(cols)

    def _tandem_line(rh: str = "*", lh: str = "*", dyn: str = "*", ped: str = "*") -> str:
        cols = [rh, lh]
        if has_dyn: cols.append(dyn)
        if has_ped: cols.append(ped)
        return "\t".join(cols)

    for ts in timesteps:
        if ts == "BARLINE":
            tracker.reset()
            lines.append(_bar_line(measure_num))
            measure_num += 1
            continue

        if isinstance(ts, tuple) and len(ts) == 3 and ts[0] == "CLEF":
            _, h11, spec = ts
            clef_str = f"*clef{spec}"
            if h11 == "RH":
                lines.append(_tandem_line(rh=clef_str))
            else:
                lines.append(_tandem_line(lh=clef_str))
            continue

        rh_toks, lh_toks = ts  # type: ignore[misc]

        # Nuance « locale » à ce timestep : H12 du premier token RH (sticky → ne réémettre que si change)
        dyn_val = "."
        for t in rh_toks:
            if t.h12 not in ("NULL", "") and t.h12 != last_dyn:
                dyn_val = t.h12
                last_dyn = t.h12
                break

        # Événement pédale ponctuel : H13 du premier token RH
        ped_val = "."
        for t in rh_toks:
            h13 = getattr(t, "h13", "NULL")
            if h13 == "Ped_Down":   ped_val = "D"; break
            if h13 == "Ped_Up":     ped_val = "U"; break
            if h13 == "Ped_Change": ped_val = "C"; break

        # Direction de tempo (H14) du premier token RH : tandem AVANT l'événement.
        for t in rh_toks:
            h14 = getattr(t, "h14", "NULL")
            if h14 in _H14_TO_KERN:
                lines.append(_tandem_line(rh=_H14_TO_KERN[h14]))
                break

        rh_grace, rh_main = _split_grace(rh_toks)
        lh_grace, lh_main = _split_grace(lh_toks)

        # Ligne(s) de grace notes (avant la note principale)
        if rh_grace or lh_grace:
            rh_g = _group_to_kern(rh_grace, "RH", tracker) if rh_grace else "."
            lh_g = _group_to_kern(lh_grace, "LH", tracker) if lh_grace else "."
            lines.append(_line(rh_g, lh_g, ".", "."))

        # Ligne de la note principale
        if rh_main or lh_main:
            rh_s = _group_to_kern(rh_main, "RH", tracker) if rh_main else "."
            lh_s = _group_to_kern(lh_main, "LH", tracker) if lh_main else "."
            lines.append(_line(rh_s, lh_s, dyn_val, ped_val))

    # Barline finale + terminateur de spines
    lines.append(_bar_line_str := "=||")
    cols_final  = ["=||", "=||"]
    cols_term   = ["*-",  "*-"]
    if has_dyn:
        cols_final.append("=||")
        cols_term.append("*-")
    if has_ped:
        cols_final.append("=||")
        cols_term.append("*-")
    lines[-1] = "\t".join(cols_final)
    lines.append("\t".join(cols_term))

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
