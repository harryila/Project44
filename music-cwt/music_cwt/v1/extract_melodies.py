#!/usr/bin/env python3
"""
extract_melodies.py — Prépare le dataset de mélodies pour heuristic_melody_baseline.

Deux étapes enchaînées :
  1. Copie plate   : kernscores_composers/**/*.krn  →  kern_data/{ComposerDir}_{file}.krn
  2. Extraction    : kern_data/*.krn                →  melody_data/*.krn   (1 spine, note la plus haute)

Usage :
    python extract_melodies.py [--skip-copy]

Options :
    --skip-copy   Sauter l'étape de copie (kern_data/ déjà rempli).

Prérequis : Python 3.10+, aucune dépendance tierce.
"""

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

COMPOSERS_DIR = ROOT / "kern_scrapping" / "kernscores_composers"
KERN_DATA_DIR = HERE / "kern_data"
MELODY_DATA_DIR = HERE / "melody_data"

# Import de l'heuristique d'extraction depuis melody_test/
sys.path.insert(0, str(ROOT / "melody_test"))
from extract_melody import extract_melody  # noqa: E402


# ---------------------------------------------------------------------------
# Étape 1 — Copie plate vers kern_data/
# ---------------------------------------------------------------------------

def copy_kern_data() -> int:
    """
    Parcourt kernscores_composers/ récursivement et copie chaque .krn dans
    kern_data/ sous le nom {ComposerDir}_{original}.krn.
    Gère les rares collisions de noms avec un suffixe numérique (_2, _3…).
    Retourne le nombre de fichiers copiés.
    """
    KERN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    n_copied = n_skip = 0

    for src in sorted(COMPOSERS_DIR.rglob("*.krn")):
        composer_dir = src.parent.name
        base_name = f"{composer_dir}_{src.name}"

        # Résolution de collisions
        dest_name = base_name
        counter = 2
        while dest_name in used_names:
            stem = src.stem
            dest_name = f"{composer_dir}_{stem}_{counter}.krn"
            counter += 1

        used_names.add(dest_name)
        dst = KERN_DATA_DIR / dest_name

        if dst.exists():
            n_skip += 1
            continue

        shutil.copy2(src, dst)
        n_copied += 1

    print(f"  Copie : {n_copied} fichiers copiés, {n_skip} déjà présents.")
    return n_copied + n_skip


# ---------------------------------------------------------------------------
# Étape 2 — Extraction des mélodies vers melody_data/
# ---------------------------------------------------------------------------

def extract_all_melodies() -> tuple[int, int]:
    """
    Traite tous les .krn de kern_data/ et écrit les mélodies dans melody_data/.
    Retourne (n_ok, n_err).
    """
    MELODY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    krn_files = sorted(KERN_DATA_DIR.glob("*.krn"))

    if not krn_files:
        print("  Aucun fichier .krn trouvé dans kern_data/")
        return 0, 0

    n_ok = n_err = 0
    for src in krn_files:
        dst = MELODY_DATA_DIR / src.name
        if dst.exists():
            n_ok += 1
            continue
        try:
            raw_lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            result_lines = extract_melody(raw_lines)
            dst.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
            n_ok += 1
        except Exception as exc:
            print(f"  ERR  {src.name} → {exc}")
            n_err += 1

    return n_ok, n_err


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    skip_copy = "--skip-copy" in sys.argv

    print(f"Répertoire de travail : {HERE}")
    print(f"Source kern           : {COMPOSERS_DIR}")
    print(f"kern_data/            : {KERN_DATA_DIR}")
    print(f"melody_data/          : {MELODY_DATA_DIR}")
    print()

    # ── Étape 1 ──────────────────────────────────────────────────────────────
    if skip_copy:
        existing = len(list(KERN_DATA_DIR.glob("*.krn")))
        print(f"[1/2] Copie ignorée (--skip-copy). {existing} fichiers dans kern_data/.")
    else:
        print("[1/2] Copie des fichiers kern vers kern_data/ …")
        total = copy_kern_data()
        print(f"      Total dans kern_data/ : {total} fichiers.")

    print()

    # ── Étape 2 ──────────────────────────────────────────────────────────────
    print("[2/2] Extraction des mélodies vers melody_data/ …")
    n_ok, n_err = extract_all_melodies()
    total = n_ok + n_err
    print(f"\n  {total} fichier(s) traité(s) — {n_ok} OK, {n_err} erreur(s).")

    if n_err > 0:
        print(f"  Taux de succès : {100 * n_ok / total:.1f}%")
    else:
        print("  Toutes les extractions ont réussi.")


if __name__ == "__main__":
    main()
