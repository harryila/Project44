"""Reproduit le split val du fine-tuning (seed=42) pour identifier les fichiers val."""
import os
import sys
import random
from pathlib import Path
import torch
from torch.utils.data import random_split

def is_not_pdmx(path: Path) -> bool:
    return not path.stem.startswith("Qm")

def walk_krn(root: Path) -> list[Path]:
    out: list[Path] = []
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            if n.endswith(".krn"):
                out.append(Path(d) / n)
    return out

# Pour reproduire le split serveur, on imite la structure ~/taff/kern_all/
# (asap, chopin_first, kernscores — pas besoin de pdmx qui est filtre)
SOURCES = {
    # Corpus roots: set CWT_KERN_SCRAPPING to the folder holding asap_kern/, humdrum-chopin-first-editions/
    # and kernscores_composers/ (the original hard-coded the author's Windows paths here).
    "asap":         os.path.join(os.environ.get("CWT_KERN_SCRAPPING", "kern_scrapping"), "asap_kern"),
    "chopin_first": os.path.join(os.environ.get("CWT_KERN_SCRAPPING", "kern_scrapping"), "humdrum-chopin-first-editions", "kern"),
    "kernscores":   os.path.join(os.environ.get("CWT_KERN_SCRAPPING", "kern_scrapping"), "kernscores_composers"),
}

# Sur le serveur : `_walk_krn(~/taff/kern_all/)` puis sorted -> ordre lex sur chemin complet
# qui = "kern_all/asap/X.krn", "kern_all/chopin_first/Y.krn", "kern_all/kernscores/Z.krn"
# Localement on a 3 dossiers ; pour avoir le MEME ordre relatif (alphabétique des sous-dossiers),
# on traite les 3 sources dans l'ordre alphabetique de leur nom de sous-dossier.
all_files: list[Path] = []
for sub, path in sorted(SOURCES.items()):  # asap, chopin_first, kernscores
    files = sorted(walk_krn(Path(path)))
    print(f"  {sub:<14} {len(files):>6} fichiers", file=sys.stderr)
    all_files.extend(files)

# Filter PDMX (pas presents localement mais on applique la regle pour matcher)
files = [f for f in all_files if is_not_pdmx(f)]
print(f"  TOTAL apres filtre : {len(files)}", file=sys.stderr)

n = len(files)
n_val = max(1, int(n * 0.05))
n_train = n - n_val

# Reproduire random_split avec seed=42
gen = torch.Generator().manual_seed(42)
indices = torch.randperm(n, generator=gen).tolist()
val_indices = indices[n_train:]
val_files = [files[i] for i in val_indices]

print(f"\n=== {len(val_files)} fichiers val (seed=42) ===", file=sys.stderr)
# Tri par source pour lisibilite
by_source = {"chopin": [], "asap": [], "kernscores": [], "other": []}
for f in val_files:
    p = str(f).lower()
    if "humdrum-chopin" in p: by_source["chopin"].append(f)
    elif "asap" in p:         by_source["asap"].append(f)
    elif "kernscores" in p:   by_source["kernscores"].append(f)
    else:                     by_source["other"].append(f)

for src, lst in by_source.items():
    print(f"\n--- {src} ({len(lst)}) ---", file=sys.stderr)
    for f in lst[:25]:
        print(f"  {f.name}")
    if len(lst) > 25:
        print(f"  ... +{len(lst)-25} autres", file=sys.stderr)
