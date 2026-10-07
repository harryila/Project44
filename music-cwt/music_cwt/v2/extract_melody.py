#!/usr/bin/env python3
"""
extract_melody.py — Extrait la melodie d'un fichier kern 2-spines piano.

Deux methodes au choix :
  - top_spine : prend la spine RH (premiere colonne **kern), reduit les accords au top note
  - skyline  : prend la note la plus aigue toutes spines confondues a chaque offset

Sortie : un fichier kern monophonique (1 spine) preservant l'en-tete meta
(clef, key, time, tempo, barlines, rests). Pas de chords, pas de spine LH.

Usage :
  python extract_melody.py --input <kern_file>
  python extract_melody.py --input-dir <dir> --output-dir <dir> [--method top_spine|skyline]

Utilise par finetune_harmonize.py pour construire les paires (melodie, score complet).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pitch utility — kern pitch -> MIDI (pour skyline)
# ---------------------------------------------------------------------------

_LETTER_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def kern_to_midi(tok: str) -> int | None:
    """Parse un token kern (juste la partie pitch) en numero MIDI. None si rest/parse fail."""
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
    pc = _LETTER_PC.get(letter)
    if pc is None:
        return None
    if acc.startswith("#"):
        pc += len(acc)
    elif acc.startswith("-"):
        pc -= len(acc)
    return (octave + 1) * 12 + pc


def extract_pitch_part(token: str) -> str | None:
    """Extrait la partie pitch d'un token kern (apres duration, avant decorations)."""
    m = re.search(r"([A-Ga-g]+[#-]*n?)", token)
    return m.group(1) if m else None


def is_rest(token: str) -> bool:
    return bool(re.search(r"[rR]", token)) and not extract_pitch_part(token)


def split_chord(cell: str) -> list[str]:
    """Un cellule kern peut etre une note simple `4c` ou un accord `4c 4e 4g`."""
    cell = cell.strip()
    if not cell or cell == ".":
        return []
    return cell.split()


def top_note_of_chord(notes: list[str]) -> str | None:
    """Renvoie le token de la note la plus aigue de l'accord (par pitch MIDI)."""
    if not notes:
        return None
    if len(notes) == 1:
        return notes[0]
    best = (None, -1)
    for n in notes:
        p = extract_pitch_part(n)
        if p is None:
            continue
        midi = kern_to_midi(p)
        if midi is None:
            continue
        if midi > best[1]:
            best = (n, midi)
    return best[0] if best[0] else notes[0]


# ---------------------------------------------------------------------------
# Methode 1 — top-spine extraction
# ---------------------------------------------------------------------------

def _detect_rh_column(kern_text: str) -> int | None:
    """
    Detecte la colonne **kern correspondant a la RH (= la spine MELODIE).
    Heuristique :
      1. Si une declaration *clef est presente : preferer la colonne en clefG2 / clefG1
         (treble) plutot que clefF4 (bass).
      2. Fallback : computer le pitch moyen de chaque colonne kern et prendre la plus aigue.
    """
    kern_indices: list[int] = []
    clef_per_col: dict[int, str] = {}

    # 1. Trouver les colonnes **kern + leurs clefs declarees
    for line in kern_text.split("\n"):
        if not line or line.startswith("!"):
            continue
        if line.startswith("**"):
            cols = line.split("\t")
            kern_indices = [i for i, c in enumerate(cols) if c.strip() == "**kern"]
            continue
        if line.startswith("*clef"):
            cols = line.split("\t")
            for idx in kern_indices:
                if idx < len(cols):
                    val = cols[idx].strip()
                    if val.startswith("*clef"):
                        clef_per_col[idx] = val
        # Si on a a la fois les kern indices et au moins une clef → on peut decider
        if kern_indices and clef_per_col:
            # Critere 1 : preferer treble (G2 / G1)
            treble_cols = [i for i, c in clef_per_col.items()
                           if "G2" in c or "G1" in c or ("G" in c and "F" not in c)]
            bass_cols = [i for i, c in clef_per_col.items() if "F4" in c or "F3" in c]
            if treble_cols and not bass_cols:
                # Toutes treble : prendre la 1ere
                return treble_cols[0]
            if treble_cols and bass_cols:
                # Mix : prendre la 1ere treble (= la RH)
                return treble_cols[0]
            if bass_cols and not treble_cols:
                # Bizarre, tout bass : prendre la 1ere
                return bass_cols[0]
            break  # on a vu les clefs, decision faite

    # 2. Fallback : pitch moyen par colonne
    if not kern_indices:
        return None

    if len(kern_indices) == 1:
        return kern_indices[0]

    sums = {i: 0 for i in kern_indices}
    counts = {i: 0 for i in kern_indices}
    for line in kern_text.split("\n"):
        if not line or line.startswith(("!", "*", "=")):
            continue
        cols = line.split("\t")
        for idx in kern_indices:
            if idx >= len(cols):
                continue
            cell = cols[idx].strip()
            if not cell or cell == ".":
                continue
            for note in split_chord(cell):
                p = extract_pitch_part(note)
                if p is None:
                    continue
                midi = kern_to_midi(p)
                if midi is not None:
                    sums[idx] += midi
                    counts[idx] += 1

    means = {i: (sums[i] / counts[i] if counts[i] else 0) for i in kern_indices}
    # La colonne au pitch moyen le plus haut = la RH
    return max(means, key=lambda i: means[i])


def extract_top_spine(kern_text: str) -> str:
    """
    Extrait la melodie = spine RH (detectee par clef ou pitch moyen),
    avec les accords reduits au top note.
    Preserve les meta (commentaires, interpretations, barlines, rests).
    """
    rh_col = _detect_rh_column(kern_text)
    if rh_col is None:
        return ""

    out_lines: list[str] = []
    seen_kern_decl = False

    for line in kern_text.split("\n"):
        line = line.rstrip("\n")
        if not line:
            continue

        if line.startswith("!"):
            out_lines.append(line.split("\t")[0])
            continue

        if line.startswith("**"):
            out_lines.append("**kern")
            seen_kern_decl = True
            continue

        if not seen_kern_decl:
            continue

        if line.startswith("*") or line.startswith("="):
            cols = line.split("\t")
            if rh_col < len(cols):
                out_lines.append(cols[rh_col])
            else:
                out_lines.append("*")
            continue

        cols = line.split("\t")
        if rh_col >= len(cols):
            continue
        cell = cols[rh_col].strip()
        if not cell or cell == ".":
            out_lines.append(".")
            continue
        notes = split_chord(cell)
        if len(notes) <= 1:
            out_lines.append(cell)
        else:
            top = top_note_of_chord(notes)
            out_lines.append(top if top else cell)

    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# Methode 2 — skyline (note la plus aigue toutes spines confondues)
# ---------------------------------------------------------------------------

def extract_skyline(kern_text: str) -> str:
    """
    A chaque ligne de donnees, prend la note la plus aigue parmi TOUTES les
    colonnes **kern (toutes mains confondues). Permet de capturer une melodie
    qui change de main en cours de morceau (jazz, certains Liszt).
    """
    out_lines: list[str] = []
    kern_indices: list[int] = []
    header_col_taken = None  # index pour clef/key/etc., on prend la 1ere kern col

    for line in kern_text.split("\n"):
        line = line.rstrip("\n")
        if not line:
            continue

        if line.startswith("!"):
            out_lines.append(line.split("\t")[0])
            continue

        if line.startswith("*") or line.startswith("=") or "**kern" in line:
            cols = line.split("\t")
            if "**kern" in line:
                kern_indices = [i for i, c in enumerate(cols) if c.strip() == "**kern"]
                if not kern_indices:
                    continue
                header_col_taken = kern_indices[0]
                out_lines.append("**kern")
                continue
            if header_col_taken is not None and header_col_taken < len(cols):
                out_lines.append(cols[header_col_taken])
            else:
                out_lines.append(cols[0] if cols else "*")
            continue

        # Ligne de donnees
        if not kern_indices:
            continue
        cols = line.split("\t")
        candidates: list[tuple[int, str]] = []   # (midi, token)
        any_rest = False
        for idx in kern_indices:
            if idx >= len(cols):
                continue
            cell = cols[idx].strip()
            if not cell or cell == ".":
                continue
            for note in split_chord(cell):
                p = extract_pitch_part(note)
                if p is None:
                    any_rest = True   # token contient 'r' mais pas de pitch
                    continue
                midi = kern_to_midi(p)
                if midi is not None:
                    candidates.append((midi, note))
        if candidates:
            # Note la plus aigue
            best = max(candidates, key=lambda x: x[0])
            out_lines.append(best[1])
        elif any_rest:
            # Si toutes les colonnes sont rests ou null, propage un rest
            # On utilise la duree la plus courante (= 4 par defaut, peu importe ici)
            out_lines.append("4r")
        else:
            out_lines.append(".")

    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(in_path: Path, method: str = "top_spine") -> str | None:
    try:
        text = in_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if method == "top_spine":
        return extract_top_spine(text)
    if method == "skyline":
        return extract_skyline(text)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="Fichier kern unique")
    p.add_argument("--input-dir", type=Path, help="Dossier de kern files (recursif)")
    p.add_argument("--output", type=Path, help="Sortie pour fichier unique")
    p.add_argument("--output-dir", type=Path, help="Dossier sortie pour batch")
    p.add_argument("--method", choices=["top_spine", "skyline"], default="top_spine")
    args = p.parse_args()

    if args.input:
        if not args.input.is_file():
            print(f"ERR: {args.input} introuvable"); sys.exit(1)
        out = process_file(args.input, method=args.method)
        if out is None:
            print(f"ERR: parse fail"); sys.exit(1)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(out, encoding="utf-8")
            print(f"Ecrit : {args.output}  ({len(out)} bytes)")
        else:
            print(out)
        return

    if args.input_dir:
        if not args.output_dir:
            print("ERR: --output-dir requis pour batch"); sys.exit(1)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(args.input_dir.rglob("*.krn"))
        n_ok = n_err = 0
        for f in files:
            out = process_file(f, method=args.method)
            if out is None:
                n_err += 1; continue
            dst = args.output_dir / f.name
            try:
                dst.write_text(out, encoding="utf-8")
                n_ok += 1
            except Exception:
                n_err += 1
        print(f"OK : {n_ok}, ERR : {n_err}, total : {len(files)}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
