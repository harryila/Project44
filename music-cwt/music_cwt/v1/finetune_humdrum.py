"""
finetune_humdrum.py — Fine-tuning du CWT sur les fichiers Humdrum uniquement.

Les fichiers Humdrum sont identifiés par la présence d'un record `!!!COM:`
(compositeur encodé manuellement = qualité maximale).

Usage :
    python3 finetune_humdrum.py --kern-dir kern_data --checkpoint-dir checkpoints

Options :
    --kern-dir        Dossier kern (défaut : kern_data)
    --checkpoint-dir  Dossier checkpoints (défaut : checkpoints)
    --epochs          Epochs de fine-tuning (défaut : 50)
    --lr              Learning rate (défaut : 1e-4, plus bas qu'en pre-training)
    --batch-size      Batch size (défaut : 4)
    --grad-accum      Accumulation de gradient (défaut : 8)
    --ckpt-every      Checkpoint toutes les N minutes (défaut : 30)
    --num-workers     Workers DataLoader (défaut : 4)
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import MultiHeadTokenizer, Vocabulary  # noqa: E402
from cwt_model import CompoundWordTransformer, ModelConfig, HEAD_NAMES  # noqa: E402


# ── Filtre Humdrum ─────────────────────────────────────────────────────────────

def is_humdrum(path: Path) -> bool:
    """Retourne True si le fichier contient un record !!!COM: (encodage Humdrum)."""
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 30:          # record !!!COM: toujours dans les 30 premières lignes
                    break
                if line.startswith("!!!COM:"):
                    return True
    except Exception:
        pass
    return False


# ── Dataset ────────────────────────────────────────────────────────────────────

class KernDataset(Dataset):
    def __init__(self, kern_dir: Path, max_seq_len: int, min_len: int = 4):
        all_files     = sorted(kern_dir.glob("*.krn"))
        self.files    = [f for f in all_files if is_humdrum(f)]
        self.tokenizer   = MultiHeadTokenizer()
        self.vocab       = Vocabulary()
        self.max_seq_len = max_seq_len
        self.min_len     = min_len
        print(f"[Dataset] {len(all_files)} fichiers total → {len(self.files)} fichiers Humdrum retenus")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            score = self.tokenizer.tokenize_file(self.files[idx])
            ids   = [self.vocab.encode_token(t) for t in score.tokens]
        except Exception:
            ids = []

        if len(ids) < self.min_len:
            ids = [[0] * 12] * self.min_len

        if len(ids) > self.max_seq_len:
            start = random.randint(0, len(ids) - self.max_seq_len)
            ids   = ids[start : start + self.max_seq_len]

        T = len(ids)
        tokens = {h: torch.zeros(T, dtype=torch.long) for h in HEAD_NAMES}
        for t, tok_ids in enumerate(ids):
            for hi, h in enumerate(HEAD_NAMES):
                tokens[h][t] = tok_ids[hi]
        return tokens, T


def collate_fn(batch):
    tokens_list, lengths = zip(*batch)
    max_len = max(lengths)
    padded = {h: torch.zeros(len(batch), max_len, dtype=torch.long) for h in HEAD_NAMES}
    for i, (tok, L) in enumerate(zip(tokens_list, lengths)):
        for h in HEAD_NAMES:
            padded[h][i, :L] = tok[h]
    return padded, torch.tensor(lengths, dtype=torch.long)


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total, n = 0.0, 0
    for tokens, lengths in loader:
        tokens  = {h: v.to(device) for h, v in tokens.items()}
        lengths = lengths.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = model.compute_loss(tokens, pad_lengths=lengths)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, step, ckpt_dir, tag=""):
    suffix = f"_{tag}" if tag else ""
    name   = f"ckpt_ft_ep{epoch + 1:03d}_step{step:06d}{suffix}.pt"
    path   = ckpt_dir / name
    torch.save({
        "epoch": epoch, "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": model.cfg,
    }, path)
    print(f"  [ckpt] {path}", flush=True)


def load_latest_checkpoint(ckpt_dir, model, optimizer, device):
    checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        print("[resume] Aucun checkpoint — démarrage à froid.")
        return 0, 0
    for ckpt_path in reversed(checkpoints):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            epoch = ckpt.get("epoch", 0)
            step  = ckpt.get("step",  0)
            print(f"[resume] {ckpt_path.name}  (epoch {epoch + 1}, step {step})")
            return epoch, step
        except Exception as e:
            print(f"[resume] Corrompu, ignoré : {ckpt_path.name}  ({e})")
    print("[resume] Tous corrompus — démarrage à froid.")
    return 0, 0


# ── Entraînement ───────────────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[config] device={device}  amp={use_amp}  lr={args.lr}")

    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Charger config depuis dernier checkpoint pour reconstruire le bon modèle
    checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    cfg = ModelConfig()
    for cp in reversed(checkpoints):
        try:
            ckpt = torch.load(cp, map_location="cpu", weights_only=False)
            if "cfg" in ckpt and isinstance(ckpt["cfg"], ModelConfig):
                cfg = ckpt["cfg"]
            break
        except Exception:
            continue

    model     = CompoundWordTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, global_step = load_latest_checkpoint(ckpt_dir, model, optimizer, device)

    # Reconfigurer lr après chargement (fine-tuning lr, pas pre-training lr)
    for g in optimizer.param_groups:
        g["lr"] = args.lr

    kern_dir = Path(args.kern_dir).expanduser()
    dataset  = KernDataset(kern_dir, max_seq_len=cfg.max_seq_len)

    if len(dataset) == 0:
        print("[error] Aucun fichier Humdrum trouvé dans", kern_dir)
        sys.exit(1)

    n_val   = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[config] train={n_train}  val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=use_amp, persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    last_ckpt_time = time.time()
    CKPT_INTERVAL  = args.ckpt_every * 60

    # Fine-tuning epochs numérotées à partir de 0 pour ne pas écraser les checkpoints pre-training
    for ft_epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss, n_steps = 0.0, 0
        t_epoch = time.time()

        for step, (tokens, lengths) in enumerate(train_loader):
            tokens  = {h: v.to(device) for h, v in tokens.items()}
            lengths = lengths.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.compute_loss(tokens, pad_lengths=lengths) / args.grad_accum

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

            if (step + 1) % 200 == 0:
                avg   = epoch_loss / n_steps
                speed = n_steps / (time.time() - t_epoch)
                print(f"  ft{ft_epoch + 1:03d} | step {step + 1:5d}/{len(train_loader)} "
                      f"| loss {avg:.4f} | {speed:.1f} it/s", flush=True)

            if time.time() - last_ckpt_time > CKPT_INTERVAL:
                save_checkpoint(model, optimizer, ft_epoch, global_step, ckpt_dir, tag="mid")
                last_ckpt_time = time.time()

        val_loss = evaluate(model, val_loader, device, use_amp)
        print(f"[ft {ft_epoch + 1:03d}/{args.epochs}] "
              f"train={epoch_loss / n_steps:.4f}  val={val_loss:.4f}  "
              f"time={time.time() - t_epoch:.0f}s", flush=True)

        save_checkpoint(model, optimizer, ft_epoch, global_step, ckpt_dir)
        last_ckpt_time = time.time()

    print("[done] Fine-tuning terminé.", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fine-tuning CWT sur fichiers Humdrum")
    p.add_argument("--kern-dir",       default="kern_data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--ckpt-every",     type=int,   default=30)
    p.add_argument("--num-workers",    type=int,   default=4)
    train(p.parse_args())

if __name__ == "__main__":
    main()
