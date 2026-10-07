#!/usr/bin/env python3
"""
multihead_tokenizer.py — Multi-head tokenizer for full-score **kern files (v2).

Converts a 2-spine Humdrum kern file into a TokenizedScore: a MetaPrefix
(key, time, tempo, clef, bracket) followed by a sequence of CompoundToken
objects, one per musical event, with 14 heads as defined in
tokenization_rules_reviewed.md.

Heads:
  H1  Type          Note_Single | Chord_Start | Chord_Cont | Chord_End
                    | Fioritura_Q | Fioritura_q | Rest | Barline
                    | Clef_G2 | Clef_F4 | Clef_G1
  H2  Duration      4 | 8. | 16 … | NULL                           (no grace marker)
  H3  Pitch         cc | G# | r | rr … | NULL
  H4  Ornament      T | M | S | : | Tr] … | NULL
  H5  Articulation  ' | ^ | ~ | ; (fermata) … | NULL
  H6  Tie           Tie_Start | Tie_Continue | Tie_End | NULL
  H7  Slur          Slur_Start | Slur_End | NULL
  H8  Phrase        Phrase_Start | Phrase_End | &Phrase_Start | &Phrase_End | NULL
  H9  Voicing       / | \\ | NULL
  H10 Beam          L | LL | LLL | J | JJ | JJJ | K | k | NULL
  H11 Spine         RH | LH
  H12 Dynamic       pp | mf | < … | NULL                          (sticky)
  H13 Pedal         Ped_Down | Ped_Up | Ped_Change | NULL          (event, RH only)
  H14 Tempo         Rall | Accel | A_Tempo | Rubato | NULL         (event, RH only)

Usage:
    python tokenizers/multihead_tokenizer.py <kern_file>
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

H1_VOCAB = [
    "Note_Single", "Chord_Start", "Chord_Cont", "Chord_End",
    "Fioritura_Q",   # appoggiature (kern uppercase Q, unslashed)
    "Fioritura_q",   # acciaccatura (kern lowercase q, slashed)
    "Rest", "Barline",
    "Clef_G2", "Clef_F4", "Clef_G1",   # mid-piece clef changes
    "SEP",   # melody-prompt → full-score boundary
]

H2_VOCAB = [
    # Standard
    "0", "1", "2", "4", "8", "16", "32", "64",
    # Dotted
    "2.", "4.", "8.", "16.", "4..", "8..",
    # Tuplets
    "3", "6", "12", "24",
    "NULL",
]


def _build_h3_vocab() -> list[str]:
    """Generate ~282 fused pitch tokens: letter(s) + accidental + r / rr."""
    letters = ["c", "d", "e", "f", "g", "a", "b"]
    accidentals = ["", "#", "##", "-", "--"]
    seen: set[str] = set()
    vocab: list[str] = []

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)

    for letter in letters:
        for rep in range(1, 5):          # 1..4 repetitions
            for acc in accidentals:
                _add(letter * rep + acc)         # lowercase: C4 and above
                _add(letter.upper() * rep + acc) # uppercase: below C4
    _add("r")
    _add("rr")
    return vocab


H3_VOCAB = _build_h3_vocab()

H4_VOCAB = [
    "T", "t", "TR", "tR", "Tr]",   # trills + nachschlag + extension-line END
    "M", "m", "W", "w",            # mordents
    "S", "$",                      # turns
    ":", "O", "I",                 # arpeggio, generic ornament, generic
    "NULL",
]

H5_VOCAB = ["'", "`", "~", "^", "I", ";", "NULL"]   # ; = fermata

H6_VOCAB = ["Tie_Start", "Tie_Continue", "Tie_End", "NULL"]

H7_VOCAB = ["Slur_Start", "Slur_End", "NULL"]

H8_VOCAB = ["Phrase_Start", "Phrase_End", "&Phrase_Start", "&Phrase_End", "NULL"]

H9_VOCAB = ["/", "\\", "NULL"]

H10_VOCAB = ["L", "LL", "LLL", "J", "JJ", "JJJ", "K", "k", "NULL"]

H11_VOCAB = ["RH", "LH"]

H12_VOCAB = [
    "pppp", "ppp", "pp", "p", "mp", "mf",
    "f", "ff", "fff", "ffff",
    "fp", "fz", "sfz", "sfp",
    "<", ">", "(", ")",   # hairpins
    "NULL",
]

H13_VOCAB = ["Ped_Down", "Ped_Up", "Ped_Change", "NULL"]

H14_VOCAB = [
    "Rall", "Poco_Rall", "Molto_Rall",
    "Accel", "Poco_Accel", "Molto_Accel",
    "A_Tempo",
    "Rubato", "Poco_Rubato", "Molto_Rubato",
    "NULL",
]

HEADS = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10",
         "h11", "h12", "h13", "h14"]

ALL_HEAD_VOCABS: dict[str, list[str]] = {
    "h1": H1_VOCAB, "h2": H2_VOCAB, "h3": H3_VOCAB,
    "h4": H4_VOCAB, "h5": H5_VOCAB, "h6": H6_VOCAB,
    "h7": H7_VOCAB, "h8": H8_VOCAB, "h9": H9_VOCAB,
    "h10": H10_VOCAB, "h11": H11_VOCAB, "h12": H12_VOCAB,
    "h13": H13_VOCAB, "h14": H14_VOCAB,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetaPrefix:
    key_sig:  str = "K_0_maj"          # e.g. "K_f#c#_maj", "K_b-_min", "K_0_maj"
    time_sig: str = "M_4/4"            # e.g. "M_3/4"
    tempo:    str = "MM_120"           # e.g. "MM_72"
    clef_rh:  str = "CLEF_G"
    clef_lh:  str = "CLEF_F"
    style:    str = ""                 # "ST_chopin" | "ST_liszt" | "" if unknown
    bracket:  str = "BRACKET_PIANO"    # piano accolade (default), or "BRACKET_NONE"


@dataclass
class CompoundToken:
    h1:  str   # Type
    h2:  str   # Duration
    h3:  str   # Pitch
    h4:  str   # Ornament
    h5:  str   # Articulation
    h6:  str   # Tie
    h7:  str   # Slur
    h8:  str   # Phrase
    h9:  str   # Voicing
    h10: str   # Beam
    h11: str   # Spine (RH | LH)
    h12: str   # Dynamic
    h13: str = "NULL"   # Pedal
    h14: str = "NULL"   # Tempo (local tempo-modification direction)


@dataclass
class TokenizedScore:
    meta:   MetaPrefix
    tokens: list[CompoundToken] = field(default_factory=list)


def make_sep() -> CompoundToken:
    """
    SEP token — marks the boundary between melody prompt and full-score generation.
    H1=SEP, all other heads NULL (same convention as Barline).

    Sequence layout:
        [Meta_melody: K, M, MM, CLEF_G]          ← 1-spine context
          → [melody tokens, H11=RH only]
          → SEP
          → [Meta_score: K, M, MM, CLEF_G, CLEF_F]  ← 2-spine context
          → [full score tokens, H11=RH|LH interleaved]

    The second Meta prefix re-conditions the model for 2-spine generation.
    Key/time/tempo can differ between the two (e.g. transposition, tempo change).
    """
    return CompoundToken(
        h1="SEP", h2="NULL", h3="NULL", h4="NULL", h5="NULL",
        h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
        h11="RH", h12="NULL", h13="NULL",
    )


# ---------------------------------------------------------------------------
# Vocabulary encoding / decoding
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Maps string head-values ↔ integer IDs.
    Unknown values at encode time fall back to NULL (or index 0 if no NULL).
    """

    def __init__(self) -> None:
        self._to_id:  dict[str, dict[str, int]] = {}
        self._to_str: dict[str, dict[int, str]] = {}
        for head, vocab in ALL_HEAD_VOCABS.items():
            self._to_id[head]  = {v: i for i, v in enumerate(vocab)}
            self._to_str[head] = {i: v for i, v in enumerate(vocab)}

    def vocab_size(self, head: str) -> int:
        return len(ALL_HEAD_VOCABS[head])

    def all_vocab_sizes(self) -> dict[str, int]:
        return {h: self.vocab_size(h) for h in HEADS}

    def _safe_id(self, head: str, value: str) -> int:
        m = self._to_id[head]
        if value in m:
            return m[value]
        if "NULL" in m:
            return m["NULL"]
        return 0

    def encode_token(self, t: CompoundToken) -> list[int]:
        """Encode one CompoundToken → list of 14 ints."""
        return [
            self._safe_id("h1",  t.h1),
            self._safe_id("h2",  t.h2),
            self._safe_id("h3",  t.h3),
            self._safe_id("h4",  t.h4),
            self._safe_id("h5",  t.h5),
            self._safe_id("h6",  t.h6),
            self._safe_id("h7",  t.h7),
            self._safe_id("h8",  t.h8),
            self._safe_id("h9",  t.h9),
            self._safe_id("h10", t.h10),
            self._safe_id("h11", t.h11),
            self._safe_id("h12", t.h12),
            self._safe_id("h13", t.h13),
            self._safe_id("h14", t.h14),
        ]

    def decode_token(self, ids: list[int]) -> CompoundToken:
        """Decode list of 14 ints → CompoundToken."""
        s = self._to_str
        return CompoundToken(
            h1=s["h1"].get(ids[0], "Note_Single"),
            h2=s["h2"].get(ids[1], "NULL"),
            h3=s["h3"].get(ids[2], "r"),
            h4=s["h4"].get(ids[3], "NULL"),
            h5=s["h5"].get(ids[4], "NULL"),
            h6=s["h6"].get(ids[5], "NULL"),
            h7=s["h7"].get(ids[6], "NULL"),
            h8=s["h8"].get(ids[7], "NULL"),
            h9=s["h9"].get(ids[8], "NULL"),
            h10=s["h10"].get(ids[9], "NULL"),
            h11=s["h11"].get(ids[10], "RH"),
            h12=s["h12"].get(ids[11], "NULL"),
            h13=s["h13"].get(ids[12], "NULL"),
            h14=s["h14"].get(ids[13], "NULL") if len(ids) > 13 else "NULL",
        )

    def encode_score(self, score: TokenizedScore) -> tuple[list[str], list[list[int]]]:
        """
        Returns:
            meta_tokens : list of meta token strings (key_sig, time_sig, tempo, …)
            token_ids   : list of [12 ints] per compound token
        """
        meta_tokens = _meta_to_tokens(score.meta)
        token_ids   = [self.encode_token(t) for t in score.tokens]
        return meta_tokens, token_ids


def _meta_to_tokens(meta: MetaPrefix) -> list[str]:
    tokens = [meta.key_sig, meta.time_sig, meta.tempo, meta.clef_rh, meta.clef_lh, meta.bracket]
    if meta.style:
        tokens.append(meta.style)
    return tokens


# ---------------------------------------------------------------------------
# Spine tracker
# ---------------------------------------------------------------------------

class _SpineTracker:
    """
    Follows Humdrum spine state line by line.
    Handles *^ (split), *v (merge), *x (exchange), *- (terminate).
    Exposes staff1_cols (RH) and staff2_cols (LH) at any moment.
    """

    def __init__(self) -> None:
        self._staves: list[int | None] = []   # staff number per column, or None
        self._types:  list[str]        = []   # "kern" | "dynam" | "other"

    @property
    def staff1_cols(self) -> list[int]:
        return [i for i, s in enumerate(self._staves) if s == 1]

    @property
    def staff2_cols(self) -> list[int]:
        return [i for i, s in enumerate(self._staves) if s == 2]

    @property
    def dynam_cols(self) -> list[int]:
        return [i for i, t in enumerate(self._types) if t == "dynam"]

    @property
    def pedal_cols(self) -> list[int]:
        return [i for i, t in enumerate(self._types) if t == "pedal"]

    def init_from_exclusive(self, fields: list[str]) -> None:
        """Process **kern / **dynam / **pedal / … line (starts with **)."""
        self._staves = [None] * len(fields)
        self._types  = []
        for f in fields:
            fs = f.strip()
            if fs == "**dynam":
                self._types.append("dynam")
            elif fs == "**pedal":
                self._types.append("pedal")
            elif fs.startswith("**kern"):
                self._types.append("kern")
            else:
                self._types.append("other")

    def update(self, line: str) -> None:
        """Process a tandem interpretation line (starts with * but not **)."""
        if not self._staves:
            return
        fields = line.split('\t')

        if '*^' in fields:
            self._do_split(fields)
        elif '*v' in fields:
            self._do_merge(fields)
        elif '*x' in fields:
            self._do_exchange(fields)
        elif '*-' in fields:
            self._do_terminate(fields)
        else:
            for i, f in enumerate(fields):
                if i >= len(self._staves):
                    break
                m = re.match(r'^\*staff(\d+)$', f.strip())
                if m:
                    self._staves[i] = int(m.group(1))

    def _stave(self, i: int) -> int | None:
        return self._staves[i] if i < len(self._staves) else None

    def _type(self, i: int) -> str:
        return self._types[i] if i < len(self._types) else "other"

    def _do_split(self, fields: list[str]) -> None:
        ns, nt = [], []
        for i, f in enumerate(fields):
            if f.strip() == '*^':
                ns.extend([self._stave(i), self._stave(i)])
                nt.extend([self._type(i),  self._type(i)])
            else:
                ns.append(self._stave(i))
                nt.append(self._type(i))
        self._staves, self._types = ns, nt

    def _do_merge(self, fields: list[str]) -> None:
        ns, nt = [], []
        i = 0
        while i < len(fields):
            if (fields[i].strip() == '*v'
                    and i + 1 < len(fields)
                    and fields[i + 1].strip() == '*v'):
                ns.append(self._stave(i))
                nt.append(self._type(i))
                i += 2
            else:
                ns.append(self._stave(i))
                nt.append(self._type(i))
                i += 1
        self._staves, self._types = ns, nt

    def _do_exchange(self, fields: list[str]) -> None:
        for i, f in enumerate(fields):
            if (f.strip() == '*x'
                    and i + 1 < len(fields)
                    and fields[i + 1].strip() == '*x'):
                if i + 1 < len(self._staves):
                    self._staves[i], self._staves[i + 1] = self._staves[i + 1], self._staves[i]
                    self._types[i],  self._types[i + 1]  = self._types[i + 1],  self._types[i]
                break

    def _do_terminate(self, fields: list[str]) -> None:
        ns, nt = [], []
        for i, f in enumerate(fields):
            if f.strip() != '*-':
                ns.append(self._stave(i))
                nt.append(self._type(i))
        self._staves, self._types = ns, nt


# ---------------------------------------------------------------------------
# Note token parser
# ---------------------------------------------------------------------------

# Duration: digits, optional dots, optional grace marker (q acciaccatura | Q appoggiature)
# Backward compat : we also accept the `q` prefix that some old code emitted (`q8cc`).
_DUR_RE   = re.compile(r'^(q?)(\d+)(\.{0,3})([qQ]?)')
# Pitch: repeated letter (case-sensitive) + optional accidentals
_PITCH_RE = re.compile(r'^([A-Ga-g]+)([#\-]{0,2})?')

# Longest-match ordering for multi-char tokens
_BEAM_TOKENS     = ["LLL", "JJJ", "LL", "JJ", "L", "J", "K", "k"]
# Tr] (trill-line END) listed FIRST so longest-match wins over T/t
_ORNAMENT_TOKENS = ["Tr]", "TR", "tR", "T", "t", "M", "m", "W", "w", "S", "$", ":", "O"]

_DYNAMIC_VALUES = frozenset(
    "pppp ppp pp p mp mf f ff fff ffff fp fz sfz sfp".split()
)
_HAIRPIN_CHARS = frozenset("<>()")

_DYN_INTERP_RE    = re.compile(r'^\*(pppp|ppp|pp|p|mp|mf|ffff|fff|ff|f|fp|fz|sfz|sfp)$')
_HAIRPIN_INTERP_RE = re.compile(r'^\*([<>()]+)$')
_KEY_MODE_RE      = re.compile(r'^\*([A-Ga-g][#\-]?):$')

# Tempo-modification tandem interps (H14). Kern token → H14 value.
_TEMPO_INTERP: dict[str, str] = {
    "*rall":        "Rall",
    "*pocorall":    "Poco_Rall",
    "*moltorall":   "Molto_Rall",
    "*accel":       "Accel",
    "*pocoaccel":   "Poco_Accel",
    "*moltoaccel":  "Molto_Accel",
    "*Atempo":      "A_Tempo",
    "*rubato":      "Rubato",
    "*pocorubato":  "Poco_Rubato",
    "*moltorubato": "Molto_Rubato",
}

_STYLE_KEYWORDS: dict[str, str] = {
    "chopin": "ST_chopin",
    "liszt":  "ST_liszt",
}


def _parse_note_token(token: str) -> dict:
    """
    Parse one kern note/rest token into a dict of head values (h2–h10).
    Sets sentinel keys 'h1_is_rest', 'h1_is_fioritura', 'h1_is_acciaccatura'.
    Does NOT set h1, h11, h12, h13.
    """
    t = token.strip()
    heads: dict[str, str] = {
        "h2": "NULL", "h3": "NULL", "h4": "NULL", "h5": "NULL",
        "h6": "NULL", "h7": "NULL", "h8": "NULL", "h9": "NULL", "h10": "NULL",
    }
    is_rest         = False
    is_fioritura    = False
    is_acciaccatura = False   # True if grace marker was lowercase 'q' (slashed)

    if not t or t == ".":
        return heads

    remaining = t

    # ── 1. Prefix markers ────────────────────────────────────────────────────
    while remaining:
        if remaining.startswith("&{"):
            heads["h8"] = "&Phrase_Start"
            remaining = remaining[2:]
        elif remaining.startswith("{"):
            heads["h8"] = "Phrase_Start"
            remaining = remaining[1:]
        elif remaining.startswith("["):
            heads["h6"] = "Tie_Start"
            remaining = remaining[1:]
        elif remaining.startswith("("):
            heads["h7"] = "Slur_Start"
            while remaining.startswith("("):   # consume all nested (
                remaining = remaining[1:]
        elif remaining[0] == ">":
            # accent/sforzando prefix (kern >) — map to H5 accent
            heads["h5"] = "^"
            remaining = remaining[1:]
        elif remaining[0] == "<":
            # portamento / crescendo prefix — consume and ignore
            remaining = remaining[1:]
        else:
            break

    # ── 2. Duration (+ grace marker) ────────────────────────────────────────
    m_dur = _DUR_RE.match(remaining)
    if m_dur:
        prefix_q = m_dur.group(1)           # "q" or "" (legacy `q8cc` form)
        dur_num  = m_dur.group(2)           # e.g. "4"
        dots     = m_dur.group(3)           # "", ".", ".."
        suffix_q = m_dur.group(4) or ""     # "q" (acciaccatura) | "Q" (appoggiature) | ""
        heads["h2"] = dur_num + dots
        if prefix_q or suffix_q == "q":
            is_fioritura, is_acciaccatura = True, True
        elif suffix_q == "Q":
            is_fioritura, is_acciaccatura = True, False
        remaining = remaining[m_dur.end():]
        # Consume optional %N tuplet marker (Humdrum notation, e.g. 64%3ee-)
        if remaining.startswith("%"):
            j = 1
            while j < len(remaining) and remaining[j].isdigit():
                j += 1
            remaining = remaining[j:]

    # ── 3. Pitch ─────────────────────────────────────────────────────────────
    if remaining.startswith("rr"):
        heads["h3"] = "rr"
        remaining = remaining[2:]
        is_rest = True
    elif remaining.startswith("r"):
        heads["h3"] = "r"
        remaining = remaining[1:]
        is_rest = True
    else:
        m_pit = _PITCH_RE.match(remaining)
        if m_pit:
            letters = m_pit.group(1)
            accs    = m_pit.group(2) or ""
            heads["h3"] = letters + accs
            remaining = remaining[m_pit.end():]
            # Skip trailing natural / editorial markers attached to pitch
            while remaining and remaining[0] in ("n", "X", "x"):
                remaining = remaining[1:]

    # ── 4. Trailing Q (rare alternative form: `8ccQ`) ───────────────────────
    if remaining.startswith("Q"):
        is_fioritura, is_acciaccatura = True, False
        remaining = remaining[1:]

    # ── 5. Suffix markers (ornaments, articulations, voicing, beam, closers) ─
    while remaining:
        c = remaining[0]

        # Skip natural / editorial accidental suffixes
        if c in ("n", "X", "x"):
            remaining = remaining[1:]
            continue

        # Beam (longest match first)
        matched_beam = False
        for bt in _BEAM_TOKENS:
            if remaining.startswith(bt):
                heads["h10"] = bt
                remaining = remaining[len(bt):]
                matched_beam = True
                break
        if matched_beam:
            continue

        # Ornament (longest match first)
        matched_orn = False
        for ot in _ORNAMENT_TOKENS:
            if remaining.startswith(ot):
                heads["h4"] = ot
                remaining = remaining[len(ot):]
                matched_orn = True
                break
        if matched_orn:
            continue

        # Single-character markers
        remaining = remaining[1:]
        if   c == "'": heads["h5"] = "'"
        elif c == "`": heads["h5"] = "`"
        elif c == "~": heads["h5"] = "~"
        elif c == "^": heads["h5"] = "^"
        elif c == "I": heads["h5"] = "I"
        elif c == ";": heads["h5"] = ";"     # fermata (point d'orgue)
        elif c == "/": heads["h9"] = "/"
        elif c == "\\": heads["h9"] = "\\"
        elif c == "]":
            # `]` after a note = Tie_End ; if Tie_Start has already been set on
            # this same token, we keep both (rare but legal in kern).
            heads["h6"] = "Tie_End"
        elif c == "_":
            # `_` after a note = Tie_Continue (kern's enclitic "middle of tie" marker)
            heads["h6"] = "Tie_Continue"
        elif c == ")": heads["h7"] = "Slur_End"
        elif c == "}":
            if remaining.startswith("&"):
                heads["h8"] = "&Phrase_End"
                remaining = remaining[1:]
            else:
                heads["h8"] = "Phrase_End"
        elif c == "&" and remaining.startswith("}"):
            heads["h8"] = "&Phrase_End"
            remaining = remaining[1:]
        # # (editorial sharp re-mark) etc. → silently ignored

    heads["h1_is_rest"]         = is_rest          # type: ignore[assignment]
    heads["h1_is_fioritura"]    = is_fioritura     # type: ignore[assignment]
    heads["h1_is_acciaccatura"] = is_acciaccatura  # type: ignore[assignment]
    return heads


# ---------------------------------------------------------------------------
# Spine-level token processing
# ---------------------------------------------------------------------------

def _collect_tokens(fields: list[str], cols: list[int]) -> list[str]:
    """Return non-null kern tokens from the given column indices."""
    result = []
    for col in cols:
        if col < len(fields):
            tok = fields[col].strip()
            if tok and tok != ".":
                result.append(tok)
    return result


def _process_spine_tokens(
    raw_tokens: list[str],
    spine: str,
    dynamic: str,
    pedal:   str = "NULL",
    tempo:   str = "NULL",
) -> list[CompoundToken]:
    """
    Convert raw kern tokens from one spine into CompoundTokens.
    Each raw token may be a kern chord (space-separated notes), single note, or rest.

    H12 (dynamic), H13 (pedal) and H14 (tempo) are attached to the first RH
    token only; callers pass "NULL" for LH.
    """
    result: list[CompoundToken] = []
    dyn_used = False
    ped_used = False
    tmp_used = False

    for raw_tok in raw_tokens:
        notes = raw_tok.split(" ")
        n = len(notes)

        for idx, note_str in enumerate(notes):
            note_str = note_str.strip()
            if not note_str or note_str == ".":
                continue

            parsed = _parse_note_token(note_str)
            is_rest:         bool = parsed.pop("h1_is_rest",         False)  # type: ignore[arg-type]
            is_fioritura:    bool = parsed.pop("h1_is_fioritura",    False)  # type: ignore[arg-type]
            is_acciaccatura: bool = parsed.pop("h1_is_acciaccatura", False)  # type: ignore[arg-type]

            # H1: Type
            if is_rest:
                h1 = "Rest"
            elif is_fioritura:
                h1 = "Fioritura_q" if is_acciaccatura else "Fioritura_Q"
            elif n == 1:
                h1 = "Note_Single"
            elif idx == 0:
                h1 = "Chord_Start"
            elif idx == n - 1:
                h1 = "Chord_End"
            else:
                h1 = "Chord_Cont"

            # Enforce constraints
            if h1 == "Rest":
                # Rests carry no ornament or phrase. Articulation H5 is ALSO
                # forced NULL except for the fermata `;` which is the only
                # legal articulation on a rest (cf. tokenization_rules_reviewed.md).
                parsed["h4"] = "NULL"
                parsed["h8"] = "NULL"
                if parsed["h5"] != ";":
                    parsed["h5"] = "NULL"

            if h1 in ("Fioritura_Q", "Fioritura_q"):
                parsed["h4"] = "NULL"
                parsed["h8"] = "NULL"

            # H12 / H13 / H14: only on the first RH token of this timestep
            h12 = "NULL"
            h13 = "NULL"
            h14 = "NULL"
            if spine == "RH":
                if not dyn_used:
                    h12 = dynamic
                    dyn_used = True
                if not ped_used:
                    h13 = pedal
                    ped_used = True
                if not tmp_used:
                    h14 = tempo
                    tmp_used = True

            result.append(CompoundToken(
                h1=h1,
                h2=parsed["h2"],
                h3=parsed["h3"],
                h4=parsed["h4"],
                h5=parsed["h5"],
                h6=parsed["h6"],
                h7=parsed["h7"],
                h8=parsed["h8"],
                h9=parsed["h9"],
                h10=parsed["h10"],
                h11=spine,
                h12=h12,
                h13=h13,
                h14=h14,
            ))

    return result


# ---------------------------------------------------------------------------
# Main tokenizer
# ---------------------------------------------------------------------------

class MultiHeadTokenizer:
    """
    Converts a 2-spine Humdrum **kern file into a TokenizedScore.

    Design choices:
    - Chords are linearized: space-separated kern notes → Chord_Start … Chord_End
    - Null continuations (`.`) are skipped — only real events are emitted
    - RH events are emitted before LH events at the same timestep
    - Dynamics are sticky: the last seen dynamic carries forward until changed
    - Style token is inferred from the filename (chopin / liszt)
    """

    def tokenize_file(self, path: Path) -> TokenizedScore:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        score = self.tokenize_lines(lines)
        fname = path.stem.lower()
        for kw, st in _STYLE_KEYWORDS.items():
            if kw in fname:
                score.meta.style = st
                break
        return score

    def tokenize_lines(self, lines: list[str]) -> TokenizedScore:
        meta    = MetaPrefix()
        tokens: list[CompoundToken] = []
        tracker = _SpineTracker()
        mode           = "maj"    # updated by *X: / *x: tandem interps
        current_dyn    = "NULL"   # sticky dynamic
        pending_pedal  = "NULL"   # consumed by the NEXT data line's first RH token
        pending_tempo  = "NULL"   # idem — tempo-modification direction (H14)
        header_done    = False    # True after first Barline / data line — clefs become Clef_* tokens

        # Bracket detection : !!!system-decoration: {(s1,s2)} → BRACKET_PIANO (default)
        for raw in lines:
            if raw.startswith("!!!system-decoration:"):
                meta.bracket = "BRACKET_PIANO" if "{(s1" in raw else "BRACKET_NONE"
                break

        def _clef_token(spec: str, h11: str) -> CompoundToken | None:
            """spec = 'G2', 'F4', 'G1' → Clef_* token. Returns None if unsupported."""
            name = f"Clef_{spec}"
            if name not in H1_VOCAB:
                return None
            return CompoundToken(
                h1=name, h2="NULL", h3="NULL", h4="NULL", h5="NULL",
                h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
                h11=h11, h12="NULL", h13="NULL",
            )

        for raw_line in lines:
            line = raw_line.rstrip("\r\n")

            if not line or line.startswith("!"):
                continue

            # ── Exclusive interpretation (**kern, **dynam, **pedal, …) ─────
            if line.startswith("**"):
                tracker.init_from_exclusive(line.split("\t"))
                continue

            # ── Tandem interpretation (*M3/4, *k[f#], *staff1, …) ──────────
            if line.startswith("*"):
                fields = line.split("\t")

                for col, f in enumerate(fields):
                    fs = f.strip()

                    # Key signature
                    if fs.startswith("*k[") and fs.endswith("]"):
                        notes = fs[3:-1]
                        meta.key_sig = f"K_{notes or '0'}_{mode}"

                    # Mode interpretation (*C: major, *a: minor, …)
                    m_mode = _KEY_MODE_RE.match(fs)
                    if m_mode:
                        tonic = m_mode.group(1)
                        mode  = "min" if tonic[0].islower() else "maj"
                        meta.key_sig = meta.key_sig.rsplit("_", 1)[0] + f"_{mode}"

                    # Time signature
                    elif fs.startswith("*M") and "/" in fs:
                        meta.time_sig = f"M_{fs[2:]}"

                    # Tempo
                    elif fs.startswith("*MM") and len(fs) > 3:
                        try:
                            meta.tempo = f"MM_{int(float(fs[3:]))}"
                        except ValueError:
                            pass

                    # Clef — header → MetaPrefix ; mid-piece → Clef_* token
                    elif fs.startswith("*clef"):
                        spec = fs[5:]            # e.g. "G2", "F4", "G1"
                        is_lh = col in tracker.staff2_cols
                        h11   = "LH" if is_lh else "RH"
                        if not header_done:
                            # initial clef → MetaPrefix
                            if spec.startswith("G"):
                                if is_lh: meta.clef_lh = "CLEF_G"
                                else:     meta.clef_rh = "CLEF_G"
                            elif spec.startswith("F"):
                                if is_lh: meta.clef_lh = "CLEF_F"
                                else:     meta.clef_rh = "CLEF_F"
                        else:
                            # mid-piece change → Clef_* token
                            tok = _clef_token(spec, h11)
                            if tok is not None:
                                tokens.append(tok)

                    # Dynamic as tandem interp (*pp, *mf, …)
                    m_dyn = _DYN_INTERP_RE.match(fs)
                    if m_dyn:
                        current_dyn = m_dyn.group(1)
                    m_hp = _HAIRPIN_INTERP_RE.match(fs)
                    if m_hp:
                        current_dyn = m_hp.group(1)

                    # Pedal as tandem interp (*ped = down, *Xped = up).
                    # Stored in pending_pedal so the NEXT data line's first RH
                    # token can carry it as H13.
                    if fs == "*ped":
                        pending_pedal = "Ped_Down"
                    elif fs == "*Xped":
                        pending_pedal = "Ped_Up"

                    # Tempo direction as tandem interp (*rall, *accel, …).
                    # Stored in pending_tempo so the NEXT data line's first RH
                    # token can carry it as H14.
                    if fs in _TEMPO_INTERP:
                        pending_tempo = _TEMPO_INTERP[fs]

                tracker.update(line)
                continue

            # ── Barline ────────────────────────────────────────────────────
            if line.startswith("="):
                tokens.append(CompoundToken(
                    h1="Barline", h2="NULL", h3="NULL", h4="NULL", h5="NULL",
                    h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
                    h11="RH", h12="NULL", h13="NULL",
                ))
                header_done = True
                continue

            # ── Data line (notes / rests) ──────────────────────────────────
            header_done = True
            fields = line.split("\t")

            rh_cols  = tracker.staff1_cols or [0]
            lh_cols  = tracker.staff2_cols or ([1] if len(fields) > 1 else [])
            dyn_cols = tracker.dynam_cols
            ped_cols = tracker.pedal_cols

            # Extract dynamic from **dynam spine if present
            for dc in dyn_cols:
                if dc < len(fields):
                    dv = fields[dc].strip()
                    if dv and dv != ".":
                        if dv in _DYNAMIC_VALUES or dv in _HAIRPIN_CHARS:
                            current_dyn = dv

            rh_raw = _collect_tokens(fields, rh_cols)
            lh_raw = _collect_tokens(fields, lh_cols)

            # Pedal / tempo events are carried by the FIRST RH token of the
            # timestep. If this data line has no RH token (LH-only line), keep
            # them pending so a later RH token picks them up — resetting here
            # would silently drop the event (cf. *Atempo followed by a
            # `.\t8e-\t.` line).
            pedal_event = pending_pedal
            tempo_event = pending_tempo
            for pc in ped_cols:
                if pc < len(fields):
                    pv = fields[pc].strip()
                    if pv and pv != ".":
                        if   pv == "D": pedal_event = "Ped_Down"
                        elif pv == "U": pedal_event = "Ped_Up"
                        elif pv == "C": pedal_event = "Ped_Change"
            if rh_raw:
                pending_pedal = "NULL"
                pending_tempo = "NULL"

            new_rh = _process_spine_tokens(rh_raw, "RH", current_dyn,
                                           pedal_event, tempo_event)
            new_lh = _process_spine_tokens(lh_raw, "LH", "NULL", "NULL", "NULL")
            tokens.extend(new_rh)
            tokens.extend(new_lh)

            # Émettre un séparateur SEP après chaque ligne kern qui a produit
            # au moins un événement, pour interdire à `_group_timesteps` de
            # marier un RH d'une ligne avec un LH d'une ligne plus tardive
            # (cas typique : passage de doubles-croches RH au-dessus
            # d'arpèges LH plus lents — kern asymétrique).
            if new_rh or new_lh:
                tokens.append(CompoundToken(
                    h1="SEP", h2="NULL", h3="NULL", h4="NULL", h5="NULL",
                    h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
                    h11="RH", h12="NULL", h13="NULL",
                ))

        return TokenizedScore(meta=meta, tokens=tokens)


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <kern_file>")
        sys.exit(1)

    _path = Path(sys.argv[1])
    if not _path.exists():
        print(f"File not found: {_path}")
        sys.exit(1)

    _tok   = MultiHeadTokenizer()
    _score = _tok.tokenize_file(_path)
    _vocab = Vocabulary()

    print("Meta:")
    print(f"  key_sig  = {_score.meta.key_sig}")
    print(f"  time_sig = {_score.meta.time_sig}")
    print(f"  tempo    = {_score.meta.tempo}")
    print(f"  clef_rh  = {_score.meta.clef_rh}")
    print(f"  clef_lh  = {_score.meta.clef_lh}")
    print(f"  style    = {_score.meta.style!r}")
    print()
    print(f"Total tokens: {len(_score.tokens)}")
    print()

    _hdr = (f"{'H1':<14} {'H2':<6} {'H3':<8} {'H4':<5} {'H5':<5} "
            f"{'H6':<14} {'H7':<12} {'H8':<14} {'H9':<5} {'H10':<6} {'H11':<4} "
            f"{'H12':<5} {'H13':<11} H14")
    print(_hdr)
    print("-" * len(_hdr))
    for _t in _score.tokens[:30]:
        print(f"{_t.h1:<14} {_t.h2:<6} {_t.h3:<8} {_t.h4:<5} {_t.h5:<5} "
              f"{_t.h6:<14} {_t.h7:<12} {_t.h8:<14} {_t.h9:<5} {_t.h10:<6} {_t.h11:<4} "
              f"{_t.h12:<5} {_t.h13:<11} {_t.h14}")

    print()
    _meta_toks, _ids = _vocab.encode_score(_score)
    print(f"Meta tokens : {_meta_toks}")
    print(f"Encoded     : ({len(_ids)}, {len(HEADS)})")
    print(f"Vocab sizes : {_vocab.all_vocab_sizes()}")
