#!/usr/bin/env python3
"""
chord_shape_stats.py — Analyse la distribution des accords dans un corpus de fichiers
.krn, pour decider de la liste atomique (qualite + voicing) du tokeniseur v2.

Pour chaque accord (token kern contenant un espace, ex "4c 4e 4g") :
  1. Extrait les pitches MIDI de chaque membre
  2. Essaie de matcher contre une liste de qualites harmoniques (Maj, min, dim, ...)
     en testant les 12 rotations possibles de fondamentale
  3. Identifie le voicing pattern (close/open, inversion, doublures)
  4. Detecte la presence d'ornements/articulations/ties per-note (qui forceraient
     un fallback sequentiel dans le tokeniseur v2)

Sorties :
  - chord_stats.csv : une ligne par accord avec quality_match, voicing_pattern,
                      n_notes, has_per_note_decoration, root_pitch
  - Resume console : top qualites, top voicings, % atomisable

Usage :
  python chord_shape_stats.py --input <dir> [--output stats.csv] [--sample N]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Qualites harmoniques : nom -> intervalles depuis la fondamentale (mod 12)
# ---------------------------------------------------------------------------

QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "Maj":         (0, 4, 7),
    "min":         (0, 3, 7),
    "dim":         (0, 3, 6),
    "aug":         (0, 4, 8),
    "sus4":        (0, 5, 7),
    "sus2":        (0, 2, 7),
    "Maj7":        (0, 4, 7, 11),
    "min7":        (0, 3, 7, 10),
    "dom7":        (0, 4, 7, 10),
    "half_dim7":   (0, 3, 6, 10),
    "dim7":        (0, 3, 6, 9),
    "dom9":        (0, 4, 7, 10, 2),
    "minMaj7":     (0, 3, 7, 11),  # bonus : extension utile en romantique
    "aug7":        (0, 4, 8, 10),  # bonus
}


# ---------------------------------------------------------------------------
# Conversion pitch kern -> MIDI
# ---------------------------------------------------------------------------

_LETTER_TO_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def kern_pitch_to_midi(tok: str) -> int | None:
    """
    Parse un token de pitch kern (sans la duree, ex "cc#", "EE-", "ccc")
    et retourne le numero MIDI. None si parse impossible.

    Convention kern :
      C4 = "c", C5 = "cc", C6 = "ccc"
      C3 = "C", C2 = "CC", C1 = "CCC"
      Octave boundary : entre B et C
    """
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

    # accidentals (kern : # diese, - bemol)
    if acc.startswith("#"):
        pc += len(acc)
    elif acc.startswith("-"):
        pc -= len(acc)
    # "n" (natural) : pas de modification

    midi = (octave + 1) * 12 + pc  # C4 = 60 ; C-1 = 0
    return midi


# ---------------------------------------------------------------------------
# Parse une note kern (avec duree, decorations) -> (pitch_midi, has_decoration)
# ---------------------------------------------------------------------------

# Caracteres kern qui indiquent une decoration per-note (ornement, articulation,
# tie, slur). Si l'un d'eux est present, l'accord ne peut pas etre atomise.
_DECORATION_CHARS = set("TtMmWwSs:;'`~^IJK_[](){}\\/")
# Note : "/" et "\" sont les stems (H9 Voicing), techniquement pas une decoration
#   musicale, mais on les compte car ils sont per-note et bloqueraient l'atomisation
#   dans la spec v2 stricte. On peut les exclure pour une version plus permissive.
_HARD_DECORATION_CHARS = set("TtMmWwSs:;'`~^I[]()")


def parse_kern_note(token: str) -> tuple[int | None, bool, bool]:
    """
    Parse un token de note kern (ex "4c", "8cc#T", "[4d-").
    Retourne (midi_pitch, has_hard_decoration, has_any_decoration).
    Hard decoration = ornement, articulation, tie, slur (bloque l'atomisation).
    Any decoration = inclut aussi stems et beams (info supplementaire).
    """
    has_hard = any(c in _HARD_DECORATION_CHARS for c in token)
    has_any = any(c in _DECORATION_CHARS for c in token)
    # Extraire la partie pitch : strip duration prefix + decorations
    # Approche simple : trouver le segment de lettres+accidentals
    m = re.search(r"([A-Ga-g]+[#-]*n?)", token)
    if not m:
        return (None, has_hard, has_any)
    pitch_part = m.group(1)
    midi = kern_pitch_to_midi(pitch_part)
    return (midi, has_hard, has_any)


# ---------------------------------------------------------------------------
# Detection de qualite harmonique
# ---------------------------------------------------------------------------

def detect_quality(midi_pitches: list[int]) -> tuple[int, str] | None:
    """
    Essaie de matcher l'ensemble de pitches contre les qualites connues.
    Pour chaque membre candidat-fondamentale :
      - calcule les intervalles mod 12 depuis ce membre
      - compare au set d'intervalles de chaque qualite (modulo permutations)
    Retourne (root_midi, quality_name) du premier match, ou None.
    Prefere la note la plus basse comme fondamentale si plusieurs matchs.
    """
    if not midi_pitches or len(midi_pitches) < 2:
        return None

    # Pitch classes uniques (les doublures d'octave sont ignorees pour la qualite)
    pcs = sorted({p % 12 for p in midi_pitches})
    if len(pcs) < 2:
        return None  # juste une octave doublee, pas un accord

    # Tester chaque pitch class comme fondamentale candidate
    candidates: list[tuple[int, str]] = []  # (root_pc, quality)
    for root_pc in pcs:
        intervals_set = frozenset((p - root_pc) % 12 for p in pcs)
        for qname, q_intervals in QUALITY_INTERVALS.items():
            if frozenset(q_intervals) == intervals_set:
                candidates.append((root_pc, qname))

    if not candidates:
        return None

    # Preferer le candidat dont la fondamentale est la note la plus basse jouee
    bass_pc = min(midi_pitches) % 12
    for root_pc, qname in candidates:
        if root_pc == bass_pc:
            # Trouver le pitch MIDI le plus bas qui a cette PC
            root_midi = min(p for p in midi_pitches if p % 12 == root_pc)
            return (root_midi, qname)

    # Sinon prendre le premier (root pas en basse = inversion)
    root_pc, qname = candidates[0]
    root_midi = min(p for p in midi_pitches if p % 12 == root_pc)
    return (root_midi, qname)


# ---------------------------------------------------------------------------
# Detection de voicing pattern
# ---------------------------------------------------------------------------

def detect_voicing(midi_pitches: list[int], root_midi: int) -> str:
    """
    Identifie un voicing pattern compact pour cet accord.
    Retourne une string descriptive (close_root_nodb, open_root_5up, ...).

    On encode :
      - spread : 'close' (max-min < 12), 'open' (max-min < 24), 'wide' (>=24)
      - inversion : 'root', 'inv1', 'inv2', 'inv3' (selon quelle note est la basse,
                    par rapport aux intervalles MAJ/min standard)
      - doublures : 'nodb', 'dbR' (root doublee), 'dbX' (autre doublure)
    """
    if not midi_pitches:
        return "empty"

    sorted_p = sorted(midi_pitches)
    spread = sorted_p[-1] - sorted_p[0]

    if spread < 12:
        spread_cat = "close"
    elif spread < 24:
        spread_cat = "open"
    else:
        spread_cat = "wide"

    # Inversion : quelle PC est la basse par rapport a la root_pc
    root_pc = root_midi % 12
    bass_pc = sorted_p[0] % 12
    interval_from_root = (bass_pc - root_pc) % 12

    if interval_from_root == 0:
        inversion = "root"
    elif interval_from_root in (3, 4):
        inversion = "inv1"  # 3ce a la basse
    elif interval_from_root in (6, 7, 8):
        inversion = "inv2"  # 5te (ou tritone) a la basse
    elif interval_from_root in (10, 11):
        inversion = "inv3"  # 7e a la basse
    else:
        inversion = "other"  # rare : 2nde, 4te, 6te a la basse

    # Doublures
    pcs = [p % 12 for p in midi_pitches]
    pc_counts = Counter(pcs)
    if all(c == 1 for c in pc_counts.values()):
        doubling = "nodb"
    elif pc_counts.get(root_pc, 0) >= 2:
        doubling = "dbR"
    else:
        # Une note non-root est doublee
        doubled_pcs = [pc for pc, c in pc_counts.items() if c >= 2]
        if len(doubled_pcs) == 1:
            doubling = "dbX"
        else:
            doubling = "dbMulti"

    return f"{spread_cat}_{inversion}_{doubling}"


# ---------------------------------------------------------------------------
# Parse d'un fichier kern : extraire tous les accords
# ---------------------------------------------------------------------------

def extract_chord_events(kern_path: Path) -> list[dict]:
    """
    Pour chaque ligne kern qui contient un accord (token avec espace), retourne
    un dict avec les infos extraites.
    """
    events: list[dict] = []
    try:
        text = kern_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return events

    for line in text.splitlines():
        # Ignorer commentaires, en-tetes, interpretations
        if not line or line.startswith("!") or line.startswith("*") or line.startswith("="):
            continue

        # Une ligne kern : tokens separes par TAB (un par spine)
        for tok in line.split("\t"):
            tok = tok.strip()
            if " " not in tok or tok == ".":
                continue
            # C'est un accord : split sur espace
            members = tok.split(" ")
            if len(members) < 2:
                continue

            parsed = [parse_kern_note(m) for m in members]
            pitches = [p for p, _, _ in parsed if p is not None]
            if len(pitches) < 2:
                continue

            has_hard = any(h for _, h, _ in parsed)
            has_any = any(a for _, _, a in parsed)

            quality_match = detect_quality(pitches)
            if quality_match is not None:
                root_midi, qname = quality_match
                voicing = detect_voicing(pitches, root_midi)
            else:
                root_midi, qname, voicing = None, None, None

            events.append({
                "n_notes": len(members),
                "n_distinct_pcs": len({p % 12 for p in pitches}),
                "quality": qname or "UNMATCHED",
                "root_pc": (root_midi % 12) if root_midi is not None else -1,
                "voicing": voicing or "UNMATCHED",
                "has_hard_decoration": int(has_hard),
                "has_any_decoration": int(has_any),
                "atomizable": int(qname is not None and not has_hard),
            })

    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PC_TO_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Repertoire contenant les .krn (recherche recursive)")
    ap.add_argument("--output", type=Path, default=None,
                    help="CSV de sortie (defaut: <input>/chord_stats.csv)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Limiter a N fichiers (0 = tous)")
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f"Erreur : {args.input} n'est pas un dossier.")
        sys.exit(1)

    files = sorted(args.input.rglob("*.krn"))
    if args.sample > 0:
        import random
        random.seed(42)
        files = random.sample(files, min(args.sample, len(files)))

    print(f"  {len(files)} fichier(s) a analyser ...")

    output_csv = args.output or (args.input / "chord_stats.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_chords_total = 0
    n_atomizable = 0
    quality_counter: Counter[str] = Counter()
    voicing_counter: Counter[str] = Counter()
    n_notes_counter: Counter[int] = Counter()
    decoration_counter = {"hard": 0, "any_only": 0, "none": 0}
    quality_voicing_counter: Counter[tuple[str, str]] = Counter()

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "n_notes", "n_distinct_pcs", "quality",
                         "root_pc", "voicing", "has_hard_decoration",
                         "has_any_decoration", "atomizable"])

        for i, fpath in enumerate(files, 1):
            events = extract_chord_events(fpath)
            for ev in events:
                writer.writerow([fpath.name, ev["n_notes"], ev["n_distinct_pcs"],
                                 ev["quality"], ev["root_pc"], ev["voicing"],
                                 ev["has_hard_decoration"],
                                 ev["has_any_decoration"], ev["atomizable"]])

                n_chords_total += 1
                n_atomizable += ev["atomizable"]
                quality_counter[ev["quality"]] += 1
                voicing_counter[ev["voicing"]] += 1
                n_notes_counter[ev["n_notes"]] += 1
                if ev["has_hard_decoration"]:
                    decoration_counter["hard"] += 1
                elif ev["has_any_decoration"]:
                    decoration_counter["any_only"] += 1
                else:
                    decoration_counter["none"] += 1
                if ev["quality"] != "UNMATCHED" and ev["voicing"] != "UNMATCHED":
                    quality_voicing_counter[(ev["quality"], ev["voicing"])] += 1

            if i % 500 == 0 or i == len(files):
                print(f"  ... {i}/{len(files)}  chords={n_chords_total}  "
                      f"atomizable={n_atomizable}", flush=True)

    # ============================================================
    # Resume
    # ============================================================
    if n_chords_total == 0:
        print("\nAucun accord trouve.")
        return

    print(f"\n=== RESUME ===")
    print(f"Total accords          : {n_chords_total:>10,}")
    print(f"Atomisables            : {n_atomizable:>10,} ({100*n_atomizable/n_chords_total:.1f}%)")
    print(f"Non-atomisables :")
    n_unmatched = quality_counter.get("UNMATCHED", 0)
    n_decorated = decoration_counter["hard"]
    print(f"  qualite non matchee  : {n_unmatched:>10,} ({100*n_unmatched/n_chords_total:.1f}%)")
    print(f"  decoration per-note  : {n_decorated:>10,} ({100*n_decorated/n_chords_total:.1f}%)")
    print(f"  (overlap possible)")

    print(f"\nDistribution par nombre de notes :")
    for n in sorted(n_notes_counter):
        c = n_notes_counter[n]
        print(f"  {n} notes : {c:>9,} ({100*c/n_chords_total:.1f}%)")

    print(f"\nTop 15 qualites :")
    for q, c in quality_counter.most_common(15):
        print(f"  {q:<14} {c:>9,} ({100*c/n_chords_total:.1f}%)")

    print(f"\nTop 30 voicings :")
    for v, c in voicing_counter.most_common(30):
        print(f"  {v:<28} {c:>9,} ({100*c/n_chords_total:.1f}%)")

    print(f"\nTop 30 (qualite, voicing) :")
    for (q, v), c in quality_voicing_counter.most_common(30):
        print(f"  {q:<14} {v:<28} {c:>9,} ({100*c/n_chords_total:.1f}%)")

    print(f"\nCSV ecrit : {output_csv}")


if __name__ == "__main__":
    main()
