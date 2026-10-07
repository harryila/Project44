"""
train.py — Entraînement du Compound Word Transformer sur kern_data/.

Usage :
    python train.py [options]

Exemples :
    # Premier run (défauts recommandés)
    python train.py --kern-dir kern_data --checkpoint-dir ~/taff/checkpoints

    # Reprise depuis checkpoint
    python train.py --kern-dir kern_data --checkpoint-dir ~/taff/checkpoints

    # GPU limité (batch plus petit, grad accum pour compenser)
    python train.py --batch-size 2 --grad-accum 16

Spec : archi.md
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path


def _walk_krn(root: Path) -> list[Path]:
    """
    Liste recursivement les .krn dans root EN SUIVANT les symlinks de dossier.
    Python 3.9 : Path.rglob() ne suit pas les symlinks de dir, donc on utilise
    os.walk(followlinks=True). Indispensable quand le corpus est organise avec
    des symlinks (ex: kern_all/pdmx -> pdmx_scrapper/pdmx_data/kern).
    """
    out: list[Path] = []
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            if n.endswith(".krn"):
                out.append(Path(d) / n)
    return out

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ── Imports locaux ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import MultiHeadTokenizer, Vocabulary  # noqa: E402

from cwt_model import CompoundWordTransformer, ModelConfig, HEAD_NAMES  # noqa: E402

# ── Dataset ────────────────────────────────────────────────────────────────────

class KernDataset(Dataset):
    """
    Lazy-loading dataset sur les fichiers .krn d'un répertoire.
    Tokenisation à la volée dans __getitem__ (pas de preload en mémoire).
    - Séquences > max_seq_len : crop aléatoire.
    - Séquences < min_len     : remplacées par une séquence nulle (filtrées en pratique).
    """

    def __init__(self, kern_dir: Path, max_seq_len: int, min_len: int = 4):
        # Recursif avec suivi des symlinks (kern_dir/pdmx peut etre un symlink)
        self.files       = sorted(_walk_krn(kern_dir))
        self.tokenizer   = MultiHeadTokenizer()
        self.vocab       = Vocabulary()
        self.max_seq_len = max_seq_len
        self.min_len     = min_len
        print(f"[Dataset] {len(self.files)} fichiers (recursif+symlinks) dans {kern_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            score = self.tokenizer.tokenize_file(self.files[idx])
            ids   = [self.vocab.encode_token(t) for t in score.tokens]
        except Exception:
            ids = []

        if len(ids) < self.min_len:
            # Séquence trop courte — séquence nulle, filtrée par le masque
            ids = [[0] * len(HEAD_NAMES) for _ in range(self.min_len)]

        # Crop aléatoire si trop long
        if len(ids) > self.max_seq_len:
            start = random.randint(0, len(ids) - self.max_seq_len)
            ids   = ids[start : start + self.max_seq_len]

        T = len(ids)
        tokens = {h: torch.zeros(T, dtype=torch.long) for h in HEAD_NAMES}
        for t, tok_ids in enumerate(ids):
            for hi, h in enumerate(HEAD_NAMES):
                tokens[h][t] = tok_ids[hi]

        return tokens, T   # longueur réelle pour le masque de padding


def collate_fn(batch):
    """Pad chaque séquence du batch à max(lengths)."""
    tokens_list, lengths = zip(*batch)
    max_len = max(lengths)

    padded = {h: torch.zeros(len(batch), max_len, dtype=torch.long) for h in HEAD_NAMES}
    for i, (tok, L) in enumerate(zip(tokens_list, lengths)):
        for h in HEAD_NAMES:
            padded[h][i, :L] = tok[h]

    return padded, torch.tensor(lengths, dtype=torch.long)


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: CompoundWordTransformer, loader: DataLoader, device: torch.device,
             use_amp: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for tokens, lengths in loader:
        tokens  = {h: v.to(device) for h, v in tokens.items()}
        lengths = lengths.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = model.compute_loss(tokens, pad_lengths=lengths)
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, step: int, ckpt_dir: Path,
                    tag: str = "") -> None:
    suffix = f"_{tag}" if tag else ""
    name   = f"ckpt_ep{epoch + 1:03d}_step{step:06d}{suffix}.pt"
    path   = ckpt_dir / name
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       model.cfg,
    }, path)
    print(f"  [ckpt] {path}", flush=True)


def cleanup_checkpoints(ckpt_dir: Path, glob_pattern: str = "ckpt_ep*.pt",
                        keep_every: int = 15, keep_recent: int = 3) -> None:
    """
    Purge les checkpoints reguliers (matching glob_pattern, excluant _mid) :
      - garde les keep_recent derniers (par numero d'epoch)
      - garde ceux dont l'epoch est multiple de keep_every
      - supprime le reste
    Pour les _mid (sauvegardes intermediaires) : garde uniquement le plus recent.

    Note : utilise un glob_pattern specifique pour ne pas toucher aux ckpt
    d'autres phases (ex : `ckpt_ep*.pt` n'affecte pas `ckpt_ft_ep*.pt`).
    """
    import re
    all_ckpts = list(ckpt_dir.glob(glob_pattern))
    regular = [p for p in all_ckpts if "_mid" not in p.name]
    mids    = [p for p in all_ckpts if "_mid" in p.name]

    def epoch_of(p):
        m = re.search(r"ep(\d+)", p.name)
        return int(m.group(1)) if m else -1

    regular.sort(key=epoch_of)
    keep = set()
    keep.update(regular[-keep_recent:])
    for p in regular:
        if epoch_of(p) > 0 and epoch_of(p) % keep_every == 0:
            keep.add(p)

    for p in regular:
        if p not in keep:
            try:
                p.unlink()
                print(f"  [cleanup] {p.name}", flush=True)
            except OSError:
                pass

    mids.sort(key=lambda p: p.stat().st_mtime)
    for p in mids[:-1]:
        try:
            p.unlink()
            print(f"  [cleanup] {p.name}", flush=True)
        except OSError:
            pass


def load_latest_checkpoint(ckpt_dir: Path, model, optimizer, device):
    # Trier par date de modification — plus robuste que le nom avec les _mid
    checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        print("[resume] Aucun checkpoint trouvé — démarrage à froid.")
        return 0, 0

    # Essayer du plus récent au plus ancien (un _mid corrompu ne bloque pas)
    for ckpt_path in reversed(checkpoints):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            epoch = ckpt.get("epoch", 0)
            step  = ckpt.get("step",  0)
            print(f"[resume] Reprise depuis {ckpt_path.name}  (epoch {epoch + 1}, step {step})")
            return epoch, step
        except Exception as e:
            print(f"[resume] Checkpoint corrompu ignoré : {ckpt_path.name}  ({e})")

    print("[resume] Tous les checkpoints sont corrompus — démarrage à froid.")
    return 0, 0


# ── Boucle d'entraînement ──────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[config] device={device}  amp={use_amp}")

    # Modèle
    cfg = ModelConfig(
        d_model      = args.d_model,
        n_layers     = args.n_layers,
        n_attn_heads = args.n_attn_heads,
        d_ff         = args.d_ff,
        max_seq_len  = args.max_seq_len,
        dropout      = args.dropout,
    )
    model = CompoundWordTransformer(cfg).to(device)
    n_params = model.count_params()
    print(f"[config] paramètres : {n_params:,}  (~{n_params / 1e6:.1f} M)")

    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Dataset
    kern_dir = Path(args.kern_dir).expanduser()
    dataset  = KernDataset(kern_dir, max_seq_len=args.max_seq_len)

    n_val   = max(1, int(len(dataset) * 0.02))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[config] train={n_train}  val={n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=use_amp, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Répertoire des checkpoints
    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reprise depuis checkpoint
    start_epoch, global_step = load_latest_checkpoint(ckpt_dir, model, optimizer, device)

    last_ckpt_time = time.time()
    CKPT_INTERVAL  = args.ckpt_every * 60   # en secondes

    # ── Epochs ────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        n_steps    = 0
        t_epoch    = time.time()

        for step, (tokens, lengths) in enumerate(train_loader):
            tokens  = {h: v.to(device) for h, v in tokens.items()}
            lengths = lengths.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.compute_loss(tokens, pad_lengths=lengths)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            n_steps    += 1

            # Log toutes les 200 itérations (compatible tmux)
            if (step + 1) % 200 == 0:
                avg   = epoch_loss / n_steps
                speed = n_steps / (time.time() - t_epoch)
                print(
                    f"  ep{epoch + 1:03d} | step {step + 1:5d}/{len(train_loader)} "
                    f"| loss {avg:.4f} | {speed:.1f} it/s",
                    flush=True,
                )

            # Checkpoint intermédiaire toutes les N minutes
            if time.time() - last_ckpt_time > CKPT_INTERVAL:
                save_checkpoint(model, optimizer, epoch, global_step, ckpt_dir, tag="mid")
                last_ckpt_time = time.time()

        # Validation en fin d'epoch
        val_loss   = evaluate(model, val_loader, device, use_amp)
        epoch_time = time.time() - t_epoch
        print(
            f"[ep {epoch + 1:03d}/{args.epochs}] "
            f"train={epoch_loss / n_steps:.4f}  val={val_loss:.4f}  "
            f"time={epoch_time:.0f}s",
            flush=True,
        )

        save_checkpoint(model, optimizer, epoch, global_step, ckpt_dir)
        cleanup_checkpoints(ckpt_dir, glob_pattern="ckpt_ep*.pt",
                            keep_every=15, keep_recent=3)
        last_ckpt_time = time.time()

    print("[done] Entraînement terminé.", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train CWT piano model")

    # Données
    p.add_argument("--kern-dir",        default="kern_data",     help="Dossier .krn")
    p.add_argument("--checkpoint-dir",  default="checkpoints",   help="Dossier checkpoints (sur Polytechnique : ~/taff/checkpoints — NFS permanent)")

    # Entraînement
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=4,    help="Séquences par batch GPU")
    p.add_argument("--grad-accum", type=int,   default=8,    help="Accumulation de gradient (batch effectif = batch-size × grad-accum)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--ckpt-every", type=int,   default=30,   help="Checkpoint toutes les N minutes")
    p.add_argument("--num-workers",type=int,   default=4)

    # Modèle (défauts v2 = config ~16M params, alignés sur ModelConfig)
    p.add_argument("--max-seq-len",    type=int,   default=2048)
    p.add_argument("--d-model",        type=int,   default=320)
    p.add_argument("--n-layers",       type=int,   default=10)
    p.add_argument("--n-attn-heads",   type=int,   default=8)
    p.add_argument("--d-ff",           type=int,   default=1600)
    p.add_argument("--dropout",        type=float, default=0.1)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
