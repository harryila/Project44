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

# ── Types ──────────────────────────────────────────────────────────────────────

# Un timestep est soit "BARLINE" soit un couple (rh_tokens, lh_tokens)
Timestep = Union[str, tuple[list[CompoundToken], list[CompoundToken]]]


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
        elif tok.h6 == "Tie_Continue":
            parts.append("_")

        if tok.h7 == "Slur_Start":
            parts.append("(")

    # ── Durée + hauteur ────────────────────────────────────────────────────
    dur = tok.h2 if tok.h2 not in ("NULL", "") else "4"

    if is_rest:
        pitch = tok.h3 if tok.h3 in ("r", "rr") else "r"
        parts.append(dur + pitch)

    elif tok.h1 == "Fioritura_Q":
        raw   = tok.h3 if tok.h3 not in ("NULL", "") else "c"
        pitch = tracker.process(raw, spine)
        parts.append(dur + pitch + "Q")

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

_KERN_EVENT_END = frozenset({"Note_Single", "Chord_End", "Rest", "Fioritura_Q"})


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
        """Collecte un kern event complet (jusqu'à Note_Single/Chord_End/Rest/Fioritura_Q)."""
        event: list[CompoundToken] = []
        j = start
        while j < n:
            tok = tokens[j]
            if tok.h1 in ("Barline", "SEP") or tok.h11 != spine:
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
        elif tok.h11 == "RH":
            rh_event, i = _collect_event(i, "RH")
            # Chercher un kern event LH immédiatement après
            lh_event: list[CompoundToken] = []
            if i < n and tokens[i].h11 == "LH" and tokens[i].h1 not in ("Barline", "SEP"):
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
    grace     = [t for t in tokens if t.h1 == "Fioritura_Q" or t.h2 == "q"]
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


def _build_header(meta: MetaPrefix) -> list[str]:
    clef_rh = "*clefG2" if meta.clef_rh == "CLEF_G" else "*clefF4"
    clef_lh = "*clefF4" if meta.clef_lh == "CLEF_F" else "*clefG2"
    key     = _fmt_key(meta.key_sig)
    ts      = _fmt_ts(meta.time_sig)
    tempo   = _fmt_tempo(meta.tempo)

    return [
        "**kern\t**kern",
        "*staff1\t*staff2",
        f"{clef_rh}\t{clef_lh}",
        f"{key}\t{key}",
        f"{ts}\t{ts}",
        f"{tempo}\t{tempo}",
    ]


# ── Conversion principale ──────────────────────────────────────────────────────

def tokens_to_kern(score: TokenizedScore) -> str:
    """
    Convertit un TokenizedScore en chaîne **kern valide.

    Algorithme :
    - Header Humdrum depuis MetaPrefix
    - Parcours des timesteps :
        * BARLINE → ligne "=N\\t=N", reset accidentels
        * (rh, lh) → extraire grace notes, puis ligne principale
    - Footer ==  *-
    """
    timesteps = _group_timesteps(score.tokens)
    tracker   = AccidentalTracker()
    lines     = _build_header(score.meta)

    measure_num = 1

    for ts in timesteps:
        if ts == "BARLINE":
            tracker.reset()
            lines.append(f"={measure_num}\t={measure_num}")
            measure_num += 1
            continue

        rh_toks, lh_toks = ts  # type: ignore[misc]

        rh_grace, rh_main = _split_grace(rh_toks)
        lh_grace, lh_main = _split_grace(lh_toks)

        # Ligne(s) de grace notes (avant la note principale)
        if rh_grace or lh_grace:
            rh_g = _group_to_kern(rh_grace, "RH", tracker) if rh_grace else "."
            lh_g = _group_to_kern(lh_grace, "LH", tracker) if lh_grace else "."
            lines.append(f"{rh_g}\t{lh_g}")

        # Ligne de la note principale
        if rh_main or lh_main:
            rh_s = _group_to_kern(rh_main, "RH", tracker) if rh_main else "."
            lh_s = _group_to_kern(lh_main, "LH", tracker) if lh_main else "."
            lines.append(f"{rh_s}\t{lh_s}")

    lines.append("==")
    lines.append("*-\t*-")

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
