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
  H7  Slur          Slur_Start | Slur_End | Slur_EndStart | NULL
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
    "Chord_Atomic",   # v3 — accord harmonique entier en 1 compound token (H15+H16)
    "Fioritura_Q",   # appoggiature (kern uppercase Q, unslashed)
    "Fioritura_q",   # acciaccatura (kern lowercase q, slashed)
    "Rest", "Barline",
    "Clef_G2", "Clef_F4", "Clef_G1",   # mid-piece clef changes
    "Sustain",   # v3 Phase 2 — Staff_1/V1 tient son '.' (ancre de ligne). Tous heads NULL sauf H11/H17.
    "SEP",   # melody-prompt → full-score boundary (PLUS utilise comme delimiteur de ligne en v3)
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
    "S", "s", "$",                 # turns (S=turn, s=inverted turn)
    ":", "O", "I",                 # arpeggio, generic ornament, generic
    "NULL",
]

H5_VOCAB = ["'", "`", "~", "^", "I", ";", "NULL"]   # ; = fermata

H6_VOCAB = ["Tie_Start", "Tie_Continue", "Tie_End", "NULL"]

H7_VOCAB = ["Slur_Start", "Slur_End", "Slur_EndStart", "NULL"]

H8_VOCAB = ["Phrase_Start", "Phrase_End", "&Phrase_Start", "&Phrase_End", "NULL"]

H9_VOCAB = ["/", "\\", "NULL"]

H10_VOCAB = ["L", "LL", "LLL", "J", "JJ", "JJJ", "K", "k", "NULL"]

# v3 — multi-staff jusqu'a 4 portees (cf. todo §"Support multi-staff").
# Staff_1 = staff numerote 1 par le fichier kern (souvent treble/RH au piano,
# soprano dans un quatuor, melodie principale). Staff_4 = stave numerote 4
# (souvent bass dans un quatuor, ou pedale piano dans une pièce d'orgue).
# La semantique de chaque stave est apprise par le modele depuis le contexte
# (clefs Meta, registre des notes).
H11_VOCAB = ["Staff_1", "Staff_2", "Staff_3", "Staff_4"]

H12_VOCAB = [
    "pppp", "ppp", "pp", "p", "mp", "mf",
    "f", "ff", "fff", "ffff",
    "fp", "fz", "sfz", "sfp",
    "<", ">", "(", ")",   # hairpins
    "NULL",
]

H13_VOCAB = ["Ped_Down", "Ped_Up", "Ped_Change", "Una_Corda", "Tre_Corde", "NULL"]

H14_VOCAB = [
    "Rall", "Poco_Rall", "Molto_Rall",
    "Accel", "Poco_Accel", "Molto_Accel",
    "A_Tempo",
    "Rubato", "Poco_Rubato", "Molto_Rubato",
    "Calando", "Poco_Calando",
    "NULL",
]

# v3 — H15 (chord quality) et H16 (chord voicing), actifs uniquement quand H1=Chord_Atomic.
# Listes valid es empiriquement sur Chopin+KernScores (cf. accords_atomiques.md).
H15_VOCAB = [
    "Maj", "min", "dim", "aug",
    "dom7", "dim7", "half_dim7",
    "sus4",
    "NULL",
]

H16_VOCAB = [
    "close_root_nodb", "close_inv1_nodb", "close_inv2_nodb", "close_inv3_nodb",
    "open_root_nodb", "open_root_dbR",
    "open_inv1_nodb", "open_inv1_dbX",
    "open_inv2_nodb", "open_inv2_dbX",
    "NULL",
]

# v3 — H17 (voix dans la portée). Polyphonie intra-stave (Chopin : 28% des fichiers
# atteignent 3 voix sonnantes sur une portée). Axe ORTHOGONAL à H11 (portée) : le
# modèle apprend "voix interne" une seule fois, partagé entre portées. Caps :
# V1-3 sur portées 1-2, V1-2 sur portées 3-4 (~99% du corpus).
H17_VOCAB = ["V1", "V2", "V3", "NULL"]

# v3 — H18 (caractère / expression). Marques de toucher et caractère expressif global
# (dolce, cantabile, espr…), orthogonales à H12 (dynamique) : un même token peut
# porter H12=p ET H18=Dolce (ex. `p dolce` Chopin). Event-based comme H12.
# Fusions : appassionato/con passione/con anima → Espr ; scherzando/giocoso → Leggiero.
# Kern : tandems *dolce, *cantabile, *espr, *leggiero, *marcato, *sostenuto,
#        *tranquillo, *agitato, *sottovoce, *pesante.
H18_VOCAB = [
    "Dolce", "Cantabile", "Espr", "Leggiero", "Marcato_P",
    "Sostenuto", "Tranquillo", "Agitato", "Sotto_Voce", "Pesante",
    "NULL",
]

HEADS = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10",
         "h11", "h12", "h13", "h14", "h15", "h16", "h17", "h18"]

ALL_HEAD_VOCABS: dict[str, list[str]] = {
    "h1": H1_VOCAB, "h2": H2_VOCAB, "h3": H3_VOCAB,
    "h4": H4_VOCAB, "h5": H5_VOCAB, "h6": H6_VOCAB,
    "h7": H7_VOCAB, "h8": H8_VOCAB, "h9": H9_VOCAB,
    "h10": H10_VOCAB, "h11": H11_VOCAB, "h12": H12_VOCAB,
    "h13": H13_VOCAB, "h14": H14_VOCAB,
    "h15": H15_VOCAB, "h16": H16_VOCAB, "h17": H17_VOCAB, "h18": H18_VOCAB,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetaPrefix:
    key_sig:  str = "K_0_maj"          # e.g. "K_f#c#_maj", "K_b-_min", "K_0_maj"
    time_sig: str = "M_4/4"            # e.g. "M_3/4"
    tempo:    str = "MM_120"           # e.g. "MM_72"
    # v3 — clefs par stave (jusqu'a 4 staves). clefs[0] = Staff_1, etc.
    # CLEF_G / CLEF_F / CLEF_G1 / CLEF_C (vocab restreint pour le moment).
    # En 2-staves piano par defaut : ['CLEF_G', 'CLEF_F'] = treble + bass.
    clefs:    list[str] = field(default_factory=lambda: ["CLEF_G", "CLEF_F"])
    n_staves: int = 2                  # nombre de staves actifs (2-4)
    style:    str = ""                 # "ST_chopin" | "ST_liszt" | "" if unknown
    bracket:  str = "BRACKET_PIANO"    # piano accolade (default), or "BRACKET_NONE"

    # Compat retro avec l'ancienne API : on expose clef_rh / clef_lh comme
    # alias en lecture/ecriture sur clefs[0] et clefs[1].
    @property
    def clef_rh(self) -> str:
        return self.clefs[0] if self.clefs else "CLEF_G"

    @clef_rh.setter
    def clef_rh(self, v: str) -> None:
        while len(self.clefs) < 1: self.clefs.append("CLEF_G")
        self.clefs[0] = v

    @property
    def clef_lh(self) -> str:
        return self.clefs[1] if len(self.clefs) >= 2 else "CLEF_F"

    @clef_lh.setter
    def clef_lh(self, v: str) -> None:
        while len(self.clefs) < 2: self.clefs.append("CLEF_F")
        self.clefs[1] = v


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
    h13: str = "NULL"   # Pedal (Ped_Down/Up/Change | Una_Corda | Tre_Corde)
    h14: str = "NULL"   # Tempo (Rall/Accel/A_Tempo/Rubato/Calando…)
    h15: str = "NULL"   # v3 — Chord Quality (actif uniquement si H1=Chord_Atomic)
    h16: str = "NULL"   # v3 — Chord Voicing (actif uniquement si H1=Chord_Atomic)
    h17: str = "NULL"   # v3 — Voix dans la portée (V1..V3 ; NULL pour barline/clef/SEP)
    h18: str = "NULL"   # v3 — Caractère/Expression (Dolce/Cantabile/Espr…)


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
        h11="Staff_1", h12="NULL", h13="NULL",
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
        """Encode one CompoundToken → list of 18 ints (H1-H18)."""
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
            self._safe_id("h15", t.h15),
            self._safe_id("h16", t.h16),
            self._safe_id("h17", t.h17),
            self._safe_id("h18", t.h18),
        ]

    def decode_token(self, ids: list[int]) -> CompoundToken:
        """Decode list of ints → CompoundToken (heads manquants → NULL pour compat)."""
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
            h11=s["h11"].get(ids[10], "Staff_1"),
            h12=s["h12"].get(ids[11], "NULL"),
            h13=s["h13"].get(ids[12], "NULL"),
            h14=s["h14"].get(ids[13], "NULL") if len(ids) > 13 else "NULL",
            h15=s["h15"].get(ids[14], "NULL") if len(ids) > 14 else "NULL",
            h16=s["h16"].get(ids[15], "NULL") if len(ids) > 15 else "NULL",
            h17=s["h17"].get(ids[16], "NULL") if len(ids) > 16 else "NULL",
            h18=s["h18"].get(ids[17], "NULL") if len(ids) > 17 else "NULL",
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
    # v3 — emet les clefs de chaque stave actif + le token N_STAVES_X (conditionne
    # le mod\xe8le sur le nombre de staves attendu, evite les Staff_3 generes en mode 2-staves)
    n = max(2, min(4, meta.n_staves))
    tokens = [meta.key_sig, meta.time_sig, meta.tempo]
    for i in range(n):
        tokens.append(meta.clefs[i] if i < len(meta.clefs) else "CLEF_G")
    tokens.append(f"N_STAVES_{n}")
    tokens.append(meta.bracket)
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

    def staff_cols(self, n: int) -> list[int]:
        """v3 — colonnes **kern du stave n (1..4). Filtre par TYPE : un `**fing`
        ou `**text` peut porter un marqueur `*staffN` mais n'est PAS une colonne de
        notes — l'inclure ferait lire des doigtés/paroles comme des notes (h3=NULL)."""
        return [i for i, s in enumerate(self._staves)
                if s == n and self._type(i) == "kern"]

    @property
    def staff1_cols(self) -> list[int]:
        return self.staff_cols(1)

    @property
    def staff2_cols(self) -> list[int]:
        return self.staff_cols(2)

    @property
    def active_staves(self) -> list[int]:
        """Liste tri\xe9e des staves r\xe9ellement attribues (1..N)."""
        return sorted({s for s in self._staves if s is not None and s >= 1})

    @property
    def kern_cols(self) -> list[int]:
        """Toutes les colonnes de type **kern (ordre gauche→droite). Pour le fallback
        quand le fichier n'a aucun marqueur *staff."""
        return [i for i, t in enumerate(self._types) if t == "kern"]

    @property
    def dynam_cols(self) -> list[int]:
        return [i for i, t in enumerate(self._types) if t == "dynam"]

    @property
    def dynam_cols_by_staff(self) -> "dict[int, list[int]]":
        """Retourne {staff_num: [col_indices]} pour les colonnes **dynam qui ont
        un marqueur *staffN. Si aucune colonne dynam n'est tagguée, retourne {}
        (fallback : mode global, toutes les colonnes dynam fusionnées)."""
        result: dict[int, list[int]] = {}
        for i, t in enumerate(self._types):
            if t == "dynam":
                s = self._staves[i] if i < len(self._staves) else None
                if s is not None:
                    result.setdefault(s, []).append(i)
        return result

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
        # Humdrum : N colonnes consecutives marquees `*v` fusionnent en UNE seule.
        # Le tracker doit consommer ces N entrees et n'emettre qu'une fois la
        # stave/type de la premiere. La version pair-only laissait passer le 3eme
        # `*v` d'un trio comme une colonne normale -> decalage de toutes les cols
        # apres et donc mauvaise attribution staff/type (cf. 002-1-Hat.krn L148).
        ns, nt = [], []
        i = 0
        while i < len(fields):
            if fields[i].strip() == '*v':
                # Trouver la longueur du run consecutif de '*v'
                j = i
                while j < len(fields) and fields[j].strip() == '*v':
                    j += 1
                run = j - i
                if run >= 2:
                    # Fusion de `run` colonnes en 1
                    ns.append(self._stave(i))
                    nt.append(self._type(i))
                    i = j
                    continue
                # `*v` isole : passer comme une colonne normale (rare/erreur kern)
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
_ORNAMENT_TOKENS = ["Tr]", "TR", "tR", "T", "t", "M", "m", "W", "w", "S", "s", "$", ":", "O"]

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
    # Calando = ralentir ET diminuer (geste unifié, distinct de rall+dim séparés).
    # Sources : calando, perdendosi, smorzando. Kern : *calando / *pococal.
    "*calando":     "Calando",
    "*pococal":     "Poco_Calando",
}

# Caractère/expression tandem interps (H18). Kern token → H18 value.
_CARAC_INTERP: dict[str, str] = {
    "*dolce":      "Dolce",
    "*cantabile":  "Cantabile",
    "*espr":       "Espr",
    "*leggiero":   "Leggiero",
    "*marcato":    "Marcato_P",
    "*sostenuto":  "Sostenuto",
    "*tranquillo": "Tranquillo",
    "*agitato":    "Agitato",
    "*sottovoce":  "Sotto_Voce",
    "*pesante":    "Pesante",
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
        elif remaining[0] in ("\\", "/"):
            # hampe (stem) parfois placée AVANT la durée dans certaines éditions
            # (ex. `(>\16D#LL`) — la consommer ici comme H9 sinon le parse de durée
            # casse au `\` et la note est perdue (h3=NULL -> 'c' inventé).
            heads["h9"] = "\\" if remaining[0] == "\\" else "/"
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

    # Filet de robustesse : si le parse n'a extrait aucune hauteur (marqueur
    # inattendu) mais que le token en contient une, la récupérer — sinon la note
    # devient h3=NULL et la reconstruction invente un 'c'.
    if heads["h3"] == "NULL" and not is_rest:
        _ps = _extract_pitch_str(t)
        if _ps:
            heads["h3"] = _ps

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
        elif c == ")":
            # `)` ferme un slur. Si ce token a AUSSI ouvert un slur (préfixe `(`
            # → Slur_Start), la note ferme le slur précédent ET en ouvre un nouveau
            # (slurs chaînés / legato note-à-note) → Slur_EndStart.
            heads["h7"] = "Slur_EndStart" if heads.get("h7") == "Slur_Start" else "Slur_End"
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
    carac:   str = "NULL",
    voice:   str = "V1",
) -> list[CompoundToken]:
    """
    Convert raw kern tokens from one spine into CompoundTokens.
    Each raw token may be a kern chord (space-separated notes), single note, or rest.

    H12 (dynamic), H13 (pedal), H14 (tempo) and H18 (carac) are stamped on EVERY
    token — they reflect the active musical state at this moment regardless of
    voice/stave. The postprocess reads them only from the Staff_1/V1 anchor.
    H17 (voice) = `voice` for every emitted token.
    """
    result: list[CompoundToken] = []

    for raw_tok in raw_tokens:
        notes = raw_tok.split(" ")
        n = len(notes)

        # ── v3 — Atomisation des accords ────────────────────────────────────
        # Si l'accord (n>=3) matche le schema (quality + voicing valides, aucune
        # decoration per-note, H9/H10 communs), on emet UN seul compound token
        # H1=Chord_Atomic. Sinon, on retombe sur la sequence Chord_Start/Cont/End.
        if n >= 3:
            clean_notes = [s.strip() for s in notes if s.strip() and s.strip() != "."]
            if len(clean_notes) >= 3:
                parsed_members = [_parse_note_token(s) for s in clean_notes]
                if not any(p.get("h1_is_rest") or p.get("h1_is_fioritura")
                           for p in parsed_members):
                    atom = _try_atomize_chord(parsed_members, clean_notes)
                    if atom is not None:
                        h2, h3, h4, h5, h9, h10, h15, h16 = atom

                        # H4/H5 = decoration commune (relaxation 2026-06) ;
                        # H6/H7/H8 toujours NULL (tie/slur/phrase -> fallback) ;
                        # H10 = beam commun (convention kern : ecrit sur un seul membre).
                        result.append(CompoundToken(
                            h1="Chord_Atomic",
                            h2=h2, h3=h3,
                            h4=h4, h5=h5, h6="NULL", h7="NULL", h8="NULL",
                            h9=h9, h10=h10,
                            h11=spine,
                            h12=dynamic, h13=pedal, h14=tempo,
                            h15=h15, h16=h16, h17=voice, h18=carac,
                        ))
                        continue

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

            # H12/H13/H14/H18 : moment musical commun, portés par tous les tokens.
            # Le postprocess lit ces heads uniquement depuis l'ancre Staff_1/V1.
            h12 = dynamic
            h13 = pedal
            h14 = tempo

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
                h17=voice,
                h18=carac,
            ))

    return result


# ---------------------------------------------------------------------------
# Chord atomization (v3) — detection qualite + voicing
# ---------------------------------------------------------------------------
# Logique portee depuis projet_a_pt_etre_reprendre/chord_shape_stats.py, restreinte
# aux 8 qualites H15 et 10 voicings H16 valides empiriquement (cf. accords_atomiques.md).

# Qualites harmoniques : nom -> intervalles depuis la fondamentale (mod 12).
# Restreint a la liste H15 (8 entrees). Les qualites plus rares (Maj7, min7, dom9...)
# tombent en fallback sequentiel.
_QUALITY_INTERVALS: dict[str, frozenset[int]] = {
    "Maj":       frozenset((0, 4, 7)),
    "min":       frozenset((0, 3, 7)),
    "dim":       frozenset((0, 3, 6)),
    "aug":       frozenset((0, 4, 8)),
    "dom7":      frozenset((0, 4, 7, 10)),
    "dim7":      frozenset((0, 3, 6, 9)),
    "half_dim7": frozenset((0, 3, 6, 10)),
    "sus4":      frozenset((0, 5, 7)),
}

_LETTER_TO_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _kern_pitch_to_midi(tok: str) -> int | None:
    """Parse un token kern (lettre(s) + accidentels, sans duree) -> MIDI."""
    if not tok:
        return None
    m = re.match(r"^([A-Ga-g]+)(#+|-+|n)?$", tok)
    if not m:
        return None
    letters, acc = m.group(1), m.group(2) or ""
    letter = letters[0].lower()
    count = len(letters)
    if letters[0].isupper():
        octave = 4 - count
    else:
        octave = 3 + count
    pc = _LETTER_TO_PC.get(letter)
    if pc is None:
        return None
    if acc.startswith("#"):
        pc += len(acc)
    elif acc.startswith("-"):
        pc -= len(acc)
    return (octave + 1) * 12 + pc   # C4 = 60


def _extract_pitch_str(note_token: str) -> str | None:
    """Extrait le segment 'pitch+accidental' (ex 'cc#', 'EE-') d'un token kern."""
    m = re.search(r"([A-Ga-g]+[#\-]{0,2})", note_token)
    return m.group(1) if m else None


def _detect_chord_quality(midi_pitches: list[int]) -> tuple[int, str] | None:
    """
    Cherche une qualite harmonique correspondante parmi _QUALITY_INTERVALS.
    Retourne (root_midi, quality_name) ou None si aucune ne matche.
    Prefere le candidat dont la fondamentale est en basse (position fondamentale).
    """
    if len(midi_pitches) < 3:
        return None  # dyades : pas de triade -> jamais atomise
    pcs = sorted({p % 12 for p in midi_pitches})
    if len(pcs) < 3:
        return None  # juste 2 PC distinctes -> pas une triade reelle

    candidates: list[tuple[int, str]] = []   # (root_pc, qname)
    for root_pc in pcs:
        intervals = frozenset((p - root_pc) % 12 for p in pcs)
        for qname, q_intervals in _QUALITY_INTERVALS.items():
            if q_intervals == intervals:
                candidates.append((root_pc, qname))
    if not candidates:
        return None

    bass_pc = min(midi_pitches) % 12
    # 1. Position fondamentale prioritaire
    for root_pc, qname in candidates:
        if root_pc == bass_pc:
            root_midi = min(p for p in midi_pitches if p % 12 == root_pc)
            return (root_midi, qname)
    # 2. Sinon premier candidat (inversion)
    root_pc, qname = candidates[0]
    root_midi = min(p for p in midi_pitches if p % 12 == root_pc)
    return (root_midi, qname)


def _detect_chord_voicing(midi_pitches: list[int], root_midi: int) -> str | None:
    """
    Identifie le voicing parmi la liste H16. Retourne une string de H16_VOCAB
    ou None si le voicing n'y figure pas (fallback sequentiel).
    """
    if not midi_pitches:
        return None
    sorted_p = sorted(midi_pitches)
    spread = sorted_p[-1] - sorted_p[0]

    if spread < 12:
        spread_cat = "close"
    elif spread < 24:
        spread_cat = "open"
    else:
        return None    # wide -> pas dans H16

    root_pc = root_midi % 12
    bass_pc = sorted_p[0] % 12
    interval_from_root = (bass_pc - root_pc) % 12

    if interval_from_root == 0:
        inversion = "root"
    elif interval_from_root in (3, 4):
        inversion = "inv1"
    elif interval_from_root in (6, 7, 8):
        inversion = "inv2"
    elif interval_from_root in (10, 11):
        inversion = "inv3"
    else:
        return None    # 2nde/4te/6te en basse -> hors-liste

    # Doublures (par classe de hauteur)
    pcs = [p % 12 for p in midi_pitches]
    pc_counts: dict[int, int] = {}
    for pc in pcs:
        pc_counts[pc] = pc_counts.get(pc, 0) + 1

    if all(c == 1 for c in pc_counts.values()):
        doubling = "nodb"
    elif pc_counts.get(root_pc, 0) >= 2 and sum(1 for c in pc_counts.values() if c >= 2) == 1:
        doubling = "dbR"
    elif sum(1 for c in pc_counts.values() if c >= 2) == 1 and pc_counts.get(root_pc, 0) < 2:
        doubling = "dbX"
    else:
        return None    # dbMulti -> hors-liste

    label = f"{spread_cat}_{inversion}_{doubling}"
    if label not in H16_VOCAB:
        return None
    return label


# ── Reconstruction d'accord atomique (epellation kern pure) ───────────────────
# Partagee avec postprocess.py. Sert (a) a la verification "lossless" lors de
# l'atomisation (on n'atomise QUE si la reconstruction reproduit exactement les
# notes ecrites -> zero changement de partition), (b) a la detokenisation.
_QUALITY_DEGREES: dict[str, list[tuple[int, int]]] = {
    "Maj":       [(0, 0), (2, 4), (4, 7)],
    "min":       [(0, 0), (2, 3), (4, 7)],
    "dim":       [(0, 0), (2, 3), (4, 6)],
    "aug":       [(0, 0), (2, 4), (4, 8)],
    "dom7":      [(0, 0), (2, 4), (4, 7), (6, 10)],
    "dim7":      [(0, 0), (2, 3), (4, 6), (6, 9)],
    "half_dim7": [(0, 0), (2, 3), (4, 6), (6, 10)],
    "sus4":      [(0, 0), (3, 5), (4, 7)],
}
_CHORD_LETTERS = "cdefgab"


def _parse_root_kern(h3: str):
    """'eee-' -> ('e', -1, 6) = (lettre, accidentel en demi-tons, octave)."""
    m = re.match(r"^([A-Ga-g]+)(#+|-+)?$", h3 or "")
    if not m:
        return None
    letters, accs = m.group(1), m.group(2) or ""
    letter = letters[0].lower()
    nrep = len(letters)
    octave = (3 + nrep) if letters[0].islower() else (4 - nrep)
    acc = len(accs) if accs.startswith("#") else (-len(accs) if accs.startswith("-") else 0)
    return letter, acc, octave


def _render_kern_pitch(letter: str, acc: int, octave: int) -> str:
    """('e', -1, 6) -> 'eee-'. Octave via repetition de lettre (convention kern)."""
    s = letter * (octave - 3) if octave >= 4 else letter.upper() * (4 - octave)
    return s + ("#" * acc if acc > 0 else "-" * (-acc) if acc < 0 else "")


def _atomic_chord_pitches(h15: str, h16: str, h3: str) -> list[str]:
    """(qualite, voicing, fondamentale-kern) -> liste de pitch-strings kern (ascendant).
    100% kern : octave en diatonique (lettre), jamais en pitch-class. La fondamentale
    reste a son octave H3 ; aucune note n'est re-epelee enharmoniquement."""
    degrees = _QUALITY_DEGREES.get(h15)
    parsed  = _parse_root_kern(h3)
    if degrees is None or parsed is None:
        return []
    rl, ra, ro = parsed
    rpc = (_LETTER_TO_PC[rl] + ra) % 12
    rli = _CHORD_LETTERS.index(rl)
    tones: list[tuple[str, int]] = []
    for d, iv in degrees:
        tl   = _CHORD_LETTERS[(rli + d) % 7]
        tpc  = (rpc + iv) % 12
        acc  = (tpc - _LETTER_TO_PC[tl]) % 12
        if acc > 6:
            acc -= 12
        tones.append((tl, acc))
    parts     = (h16 or "").split("_")
    spread    = parts[0] if len(parts) > 0 else "close"
    inversion = parts[1] if len(parts) > 1 else "root"
    doubling  = parts[2] if len(parts) > 2 else "nodb"
    inv = min({"root": 0, "inv1": 1, "inv2": 2, "inv3": 3}.get(inversion, 0), len(tones) - 1)
    dsteps = [d for d, _ in degrees]
    placed = [
        [tones[i][0], tones[i][1],
         ro + (rli + dsteps[i]) // 7 - (1 if (inv > 0 and i >= inv) else 0)]
        for i in range(len(tones))
    ]
    _h = lambda p: p[2] * 7 + _CHORD_LETTERS.index(p[0])
    placed.sort(key=_h)

    def _foct(letter: str, acc: int) -> int:
        for p in placed:
            if p[0] == letter and p[1] == acc:
                return p[2]
        return placed[0][2]

    if doubling == "dbR":
        placed.append([rl, ra, _foct(rl, ra) + 1])
    elif doubling == "dbX" and len(tones) >= 2:
        # Doubler la note de BASSE (= tone de l'inversion), pas systematiquement la
        # 3ce : en pratique on double le plus souvent la basse (la 3ce en inv1, la
        # 5te en inv2). dbX n'apparait qu'avec inv1/inv2 (jamais en position fonda).
        bt = tones[inv] if inv > 0 else tones[1]
        placed.append([bt[0], bt[1], _foct(bt[0], bt[1]) + 1])
    elif spread == "open" and len(placed) >= 2:
        # Ouvrir = monter d'une octave la plus basse note NON-fondamentale au-dessus
        # de la basse (3ce en position fonda, 5te en inv1, 3ce en inv2 ...). La fonda
        # n'est jamais deplacee -> reste ancree a son octave H3 (fixed point), et le
        # registre colle au voicing "ouvert" classique des partitions.
        for i in range(1, len(placed)):
            if (placed[i][0], placed[i][1]) != (rl, ra):
                placed[i][2] += 1
                break
    placed.sort(key=_h)
    return [_render_kern_pitch(l, a, o) for l, a, o in placed]


def _atomic_chord_decoration(h4: str, h5: str) -> str:
    """Suffixe kern (ornement H4 puis articulation H5) re-emis sur CHAQUE membre
    d'un Chord_Atomic portant une decoration commune a tout l'accord. Partage
    entre la garde lossless du tokeniseur et la reconstruction (postprocess) pour
    garantir l'idempotence. L'ordre ornement->articulation est sans incidence sur
    le re-parsing (_parse_note_token gere les suffixes dans n'importe quel ordre)."""
    s = ""
    if h4 and h4 != "NULL":
        s += h4
    if h5 and h5 != "NULL":
        s += h5
    return s


def _try_atomize_chord(
    parsed_members: list[dict],
    note_strings:   list[str],
) -> tuple[str, str, str, str, str, str, str] | None:
    """
    Tente d'atomiser un accord en (h2, h3, h4, h5, h9, h15, h16).
    Retourne None si l'accord ne matche pas le schema (fallback sequentiel).

    Conditions de fallback (cf. accords_atomiques.md) :
      1. H6/H7/H8 non-NULL sur un membre ; OU H4/H5 differents entre membres
      2. Les membres different sur H9 (voicing stems) ou H10 (beam)
      3. La qualite harmonique n'est pas dans la liste des 8 H15
      4. Le voicing n'est pas dans la liste des 10 H16
    """
    n = len(parsed_members)
    if n < 3:
        return None

    # 1. Decorations per-note.
    #    - H6 (tie), H7 (slur), H8 (phrase) : BLOQUANTES si non-NULL sur un membre.
    #      Un tie lie une hauteur PRECISE a l'evenement suivant (per-note
    #      heterogene) ; slur/phrase sont portes par UN seul membre -> on ne
    #      saurait pas lequel apres reordonnancement a la reconstruction. Fallback.
    #    - H4 (ornement) / H5 (articulation) : RELAXATION (2026-06). Atomisables si
    #      la valeur est COMMUNE a tous les membres (ornement/articulation de tout
    #      l'accord, ex. staccato sur le bloc entier). On la porte sur le token
    #      atomique et on la re-emet sur chaque membre -> meme multiset
    #      (hauteur, decoration), donc lossless. Valeurs differentes -> fallback.
    for p in parsed_members:
        if (p.get("h6", "NULL") != "NULL" or p.get("h7", "NULL") != "NULL"
                or p.get("h8", "NULL") != "NULL"):
            return None
    if len({p.get("h4", "NULL") for p in parsed_members}) > 1:
        return None
    if len({p.get("h5", "NULL") for p in parsed_members}) > 1:
        return None
    common_h4 = parsed_members[0].get("h4", "NULL")
    common_h5 = parsed_members[0].get("h5", "NULL")

    # 2. H9 / H10 doivent etre communs aux membres
    h9_vals = {p.get("h9", "NULL") for p in parsed_members}
    if len(h9_vals) > 1:
        return None
    # H10 (beam) : en kern le marqueur L/J est ecrit sur UNE SEULE note de l'accord
    # par convention (pas repete sur tous les membres). On prend la valeur non-NULL
    # comme valeur commune ; on ne bloque que si deux valeurs non-NULL differentes
    # coexistent (cas contradictoire).
    h10_non_null = {p.get("h10", "NULL") for p in parsed_members
                    if p.get("h10", "NULL") not in ("NULL", None, "")}
    if len(h10_non_null) > 1:
        return None
    common_h10 = next(iter(h10_non_null)) if h10_non_null else "NULL"
    common_h9  = parsed_members[0].get("h9",  "NULL")

    # 3. Duree commune (toujours vraie en kern mais on verifie)
    durations = {p.get("h2", "NULL") for p in parsed_members}
    if len(durations) > 1:
        return None
    common_h2 = parsed_members[0].get("h2", "NULL")
    if common_h2 == "NULL":
        return None

    # 4. Calcul des MIDI par membre
    midi_pitches: list[int] = []
    for ns in note_strings:
        pstr = _extract_pitch_str(ns)
        if pstr is None:
            return None
        m = _kern_pitch_to_midi(pstr)
        if m is None:
            return None
        midi_pitches.append(m)

    # 5. Qualite harmonique
    q = _detect_chord_quality(midi_pitches)
    if q is None:
        return None
    root_midi, qname = q
    if qname not in H15_VOCAB:
        return None

    # 6. Voicing
    voicing = _detect_chord_voicing(midi_pitches, root_midi)
    if voicing is None:
        return None

    # 7. Pitch kern de la fondamentale : on reprend l'orthographe enharmonique
    #    du membre dont le MIDI = root_midi (preserve D# vs E-)
    root_pitch_str: str | None = None
    for ns, midi in zip(note_strings, midi_pitches):
        if midi == root_midi:
            root_pitch_str = _extract_pitch_str(ns)
            break
    if root_pitch_str is None:
        return None

    # 8. Verification LOSSLESS : n'atomiser que si la reconstruction reproduit
    #    EXACTEMENT les notes ecrites (orthographe + octave). Sinon -> None ->
    #    chemin sequentiel, qui preserve l'ecriture telle quelle. Garantit qu'aucun
    #    accord atomise ne change la partition (enharmonie, symetrie, voicing).
    recon = sorted(_atomic_chord_pitches(qname, voicing, root_pitch_str))
    orig  = sorted(p for p in (_extract_pitch_str(ns) for ns in note_strings) if p)
    if recon != orig:
        return None

    # 8b. Verification LOSSLESS (decoration H4/H5) : le suffixe re-emis doit se
    #     re-parser EXACTEMENT vers (common_h4, common_h5). Sinon -> fallback.
    #     (valeur commune emise sur chaque membre => meme multiset decoration.)
    if common_h4 != "NULL" or common_h5 != "NULL":
        suffix = _atomic_chord_decoration(common_h4, common_h5)
        probe  = _parse_note_token(common_h2 + "c" + suffix)
        if probe.get("h4", "NULL") != common_h4 or probe.get("h5", "NULL") != common_h5:
            return None

    return (common_h2, root_pitch_str, common_h4, common_h5, common_h9, common_h10, qname, voicing)


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
        pending_dyn    = "NULL"   # dynamique tandem (*p/*<) consommée par la prochaine
                                  # ligne de données. H12 est EVENT-based (pas sticky) :
                                  # porté uniquement à la ligne du marquage explicite,
                                  # sinon NULL → préserve les re-statements (mf mf) et
                                  # les hairpins répétés (plus de déduplication lossy).
        pending_pedal  = "NULL"   # consumed by the NEXT data line's first RH token
        pending_tempo  = "NULL"   # idem — tempo-modification direction (H14)
        pending_carac  = "NULL"   # idem — caractère/expression (H18)
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

        def _staff_name(n: int) -> str:
            """1..4 -> Staff_1..Staff_4 ; au-dela clipped a Staff_4."""
            return f"Staff_{min(max(n, 1), 4)}"

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
                        # v3 — quel stave porte ce clef (1..4) ?
                        stave_num = None
                        for n in (1, 2, 3, 4):
                            if col in tracker.staff_cols(n):
                                stave_num = n
                                break
                        if stave_num is None:
                            continue   # clef sur un spine non-staff (rare/erreur)
                        h11 = _staff_name(stave_num)
                        if not header_done:
                            # initial clef \xe2\x86\x92 MetaPrefix.clefs[stave_num-1]
                            clef_value = "CLEF_G" if spec.startswith("G") else (
                                         "CLEF_F" if spec.startswith("F") else "CLEF_G")
                            while len(meta.clefs) < stave_num:
                                meta.clefs.append("CLEF_G")
                            meta.clefs[stave_num - 1] = clef_value
                        else:
                            tok = _clef_token(spec, h11)
                            if tok is not None:
                                tokens.append(tok)

                    # Dynamic as tandem interp (*pp, *mf, …)
                    m_dyn = _DYN_INTERP_RE.match(fs)
                    if m_dyn:
                        pending_dyn = m_dyn.group(1)
                    m_hp = _HAIRPIN_INTERP_RE.match(fs)
                    if m_hp:
                        pending_dyn = m_hp.group(1)

                    # Pedal as tandem interp (*ped = down, *Xped = up,
                    # *unacorda / *trecorde = soft pedal).
                    if fs == "*ped":
                        pending_pedal = "Ped_Down"
                    elif fs == "*Xped":
                        pending_pedal = "Ped_Up"
                    elif fs == "*unacorda":
                        pending_pedal = "Una_Corda"
                    elif fs == "*trecorde":
                        pending_pedal = "Tre_Corde"

                    # Tempo direction as tandem interp (*rall, *calando, …).
                    if fs in _TEMPO_INTERP:
                        pending_tempo = _TEMPO_INTERP[fs]

                    # Caractère/expression as tandem interp (*dolce, *espr, …).
                    if fs in _CARAC_INTERP:
                        pending_carac = _CARAC_INTERP[fs]

                tracker.update(line)
                continue

            # ── Barline ────────────────────────────────────────────────────
            if line.startswith("="):
                tokens.append(CompoundToken(
                    h1="Barline", h2="NULL", h3="NULL", h4="NULL", h5="NULL",
                    h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
                    h11="Staff_1", h12="NULL", h13="NULL",
                ))
                header_done = True
                continue

            # ── Data line (notes / rests) ──────────────────────────────────
            header_done = True
            fields = line.split("\t")

            dyn_cols = tracker.dynam_cols
            ped_cols = tracker.pedal_cols

            # v3 \xe2\x80\x94 lire les colonnes pour chacun des 4 staves possibles
            # v3 voix : garder l'index de voix (position dans les sous-colonnes
            # du stave). Cap V1-3 sur portees 1-2, V1-2 sur 3-4 ; au-dela -> fusion
            # en accord (degradation locale).
            staff_voices: dict[int, dict[int, str]] = {}
            for n in (1, 2, 3, 4):
                cols = tracker.staff_cols(n)
                if not cols:
                    continue
                cap = 3 if n <= 2 else 2
                for vi, col in enumerate(cols, start=1):
                    if col >= len(fields):
                        continue
                    tok = fields[col].strip()
                    if not tok or tok == ".":
                        continue
                    v = min(vi, cap)
                    if n in staff_voices and v in staff_voices[n]:
                        staff_voices[n][v] = staff_voices[n][v] + " " + tok
                    else:
                        staff_voices.setdefault(n, {})[v] = tok

            # Fallback : si aucun stave attribu\xe9 (fichier sans *staff), prendre
            # les 2 premieres colonnes comme Staff_1 (col 0) et Staff_2 (col 1).
            if not staff_voices:
                # Fichier sans marqueur *staff : assigner chaque colonne **kern à une
                # portée par ordre (capé à 4). Avant : seules les colonnes 0 et 1
                # étaient lues -> perte de toutes les portées 3+ (et lecture d'une
                # **dynam si elle était en col 1).
                for idx, col in enumerate(tracker.kern_cols[:4]):
                    if col < len(fields):
                        tokv = fields[col].strip()
                        if tokv and tokv != ".":
                            staff_voices[idx + 1] = {1: tokv}

            # Dynamique de CETTE ligne (event-based). Deux modes :
            # — Per-staff (kern v3 avec **dynam *staffN) : chaque portée lit sa
            #   propre colonne dynam → dyn_events_by_staff[N] pour la portée N.
            # — Global (kern ancien, une seule **dynam sans *staffN) : toutes les
            #   colonnes dynam fusionnées en un seul dyn_event (fallback).
            dynam_by_staff = tracker.dynam_cols_by_staff
            if dynam_by_staff:
                dyn_events_by_staff: "dict[int, str]" = {}
                for sn, cols in dynam_by_staff.items():
                    ev = pending_dyn
                    for dc in cols:
                        if dc < len(fields):
                            dv = fields[dc].strip()
                            if dv and dv != ".":
                                if dv in _DYNAMIC_VALUES or dv in _HAIRPIN_CHARS:
                                    ev = dv
                                    break
                    dyn_events_by_staff[sn] = ev
                dyn_event = dyn_events_by_staff.get(1, pending_dyn)
            else:
                dyn_events_by_staff = {}
                dyn_event = pending_dyn
                for dc in dyn_cols:
                    if dc < len(fields):
                        dv = fields[dc].strip()
                        if dv and dv != ".":
                            if dv in _DYNAMIC_VALUES or dv in _HAIRPIN_CHARS:
                                dyn_event = dv
                                break

            # P\xe9dale event de la ligne (pris en compte sur Staff_1, ou pending si
            # Staff_1 n'a rien sur cette ligne).
            pedal_event = pending_pedal
            tempo_event = pending_tempo
            for pc in ped_cols:
                if pc < len(fields):
                    pv = fields[pc].strip()
                    if pv and pv != ".":
                        if   pv == "D": pedal_event = "Ped_Down"
                        elif pv == "U": pedal_event = "Ped_Up"
                        elif pv == "C": pedal_event = "Ped_Change"

            staff1_active = 1 in staff_voices
            # pending_* sont portés par l'ancre Staff_1/V1 (TOUJOURS émise).
            # → reset INCONDITIONNEL. Avant : reset seulement si staff1 actif
            # → sur-émission sur chaque ligne tant que staff1 tenait une longue note.
            pending_pedal = "NULL"
            pending_tempo = "NULL"
            pending_dyn   = "NULL"
            carac_event   = pending_carac
            pending_carac = "NULL"

            # Mise a jour du compteur n_staves (max stave seen during fichier)
            for n in staff_voices:
                if n > meta.n_staves:
                    meta.n_staves = min(n, 4)

            # Emission par stave dans l'ordre croissant (Staff_1 d'abord).
            # v3 Phase 2 — ANCRE Staff_1/V1 toujours emise en PREMIER (evenement si
            # active, sinon token Sustain). Elle delimite la ligne (plus de SEP). Les
            # autres (staff, voix) emettent seulement si actives ; leurs '.' sont skippes.
            # H12/H13/H14 portes par TOUS les tokens (moment musical commun).
            if staff_voices:
                if 1 in staff_voices and 1 in staff_voices[1]:
                    tokens.extend(_process_spine_tokens(
                        [staff_voices[1][1]], "Staff_1",
                        dyn_event, pedal_event, tempo_event, carac_event, voice="V1"))
                else:
                    tokens.append(CompoundToken(
                        h1="Sustain", h2="NULL", h3="NULL", h4="NULL", h5="NULL",
                        h6="NULL", h7="NULL", h8="NULL", h9="NULL", h10="NULL",
                        h11="Staff_1", h12=dyn_event, h13=pedal_event, h14=tempo_event,
                        h15="NULL", h16="NULL", h17="V1", h18=carac_event))
                for n in sorted(staff_voices):
                    # Per-staff dyn : utiliser la colonne dynam de la portée n
                    # (si kern per-staff) ; sinon le dyn_event global (fallback).
                    n_dyn = dyn_events_by_staff.get(n, dyn_event) if dyn_events_by_staff else dyn_event
                    for v in sorted(staff_voices[n]):
                        if n == 1 and v == 1:
                            continue
                        tokens.extend(_process_spine_tokens(
                            [staff_voices[n][v]], _staff_name(n),
                            n_dyn, pedal_event, tempo_event, "NULL", voice=f"V{v}"))

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
            f"{'H12':<5} {'H13':<11} {'H14':<14} {'H15':<10} H16")
    print(_hdr)
    print("-" * len(_hdr))
    for _t in _score.tokens[:30]:
        print(f"{_t.h1:<14} {_t.h2:<6} {_t.h3:<8} {_t.h4:<5} {_t.h5:<5} "
              f"{_t.h6:<14} {_t.h7:<12} {_t.h8:<14} {_t.h9:<5} {_t.h10:<6} {_t.h11:<4} "
              f"{_t.h12:<5} {_t.h13:<11} {_t.h14:<14} {_t.h15:<10} {_t.h16}")

    print()
    _meta_toks, _ids = _vocab.encode_score(_score)
    print(f"Meta tokens : {_meta_toks}")
    print(f"Encoded     : ({len(_ids)}, {len(HEADS)})")
    print(f"Vocab sizes : {_vocab.all_vocab_sizes()}")
