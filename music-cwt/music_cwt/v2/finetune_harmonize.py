"""
finetune_harmonize.py — Fine-tuning du CWT pour la tache d'harmonisation.

Apprend la conditionnelle P(score complet | melodie) en construisant des paires :
    [melodie monophonique RH only] + [SEP] + [score complet RH+LH]

Le modele apprend a generer la suite a partir de la melodie en prefix.

Strategie :
  - Partir du meilleur checkpoint v2 (ckpt_ft_ep015_step359042.pt = best val)
  - Filtre is_not_pdmx (corpus qualite seulement, ~5931 fichiers)
  - Pour chaque fichier : tokenize full score + extract melody (top_spine) + tokenize melody
  - Build sequence : melody_tokens + SEP + score_tokens
  - Loss masquee sur le prefix melody (le modele n'apprend pas a predire la melodie,
    seulement le score complet apres le SEP)
  - lr=5e-5 (plus bas que le ft humdrum pour ne pas detruire la capacite de pretrain)
  - epochs=20 (typiquement suffisant pour adapter a une tache narrow)

Usage :
    python3 finetune_harmonize.py --kern-dir ../kern_all \
        --base-ckpt checkpoints/ckpt_ft_ep015_step359042.pt \
        --checkpoint-dir checkpoints_harmonize \
        --epochs 20 --lr 5e-5
"""

import argparse
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import (  # noqa: E402
    MultiHeadTokenizer, Vocabulary, make_sep, HEADS as HEAD_NAMES,
)
from cwt_model import CompoundWordTransformer, ModelConfig  # noqa: E402
from extract_melody import extract_top_spine  # noqa: E402


def _walk_krn(root: Path) -> list[Path]:
    out: list[Path] = []
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            if n.endswith(".krn"):
                out.append(Path(d) / n)
    return out


def is_not_pdmx(path: Path) -> bool:
    return not path.stem.startswith("Qm")


# ── Dataset paires harmonisation ─────────────────────────────────────────────

class HarmonizeDataset(Dataset):
    """
    Pour chaque fichier .krn (corpus qualite) :
      1. Tokenise le score complet -> score_tokens
      2. Extrait la melodie (top_spine) -> melody_kern_string
      3. Tokenise la melodie -> melody_tokens (monophonique, H11=RH only)
      4. Concatene : melody_tokens + SEP + score_tokens
      5. Retourne (tokens, total_length, prefix_length)
         prefix_length = len(melody_tokens) + 1 (= position du 1er token du score)

    Filtres :
      - is_not_pdmx (pas de PDMX)
      - melody minimum 5 tokens (sinon trop court pour avoir un sens)
      - score minimum 10 tokens
      - sequence totale <= max_seq_len (sinon crop sur le score uniquement, preserve la melodie)
    """

    def __init__(self, kern_dir: Path, max_seq_len: int, min_len: int = 4):
        all_files = sorted(_walk_krn(kern_dir))
        self.files = [f for f in all_files if is_not_pdmx(f)]
        self.tokenizer = MultiHeadTokenizer()
        self.vocab     = Vocabulary()
        self.max_seq_len = max_seq_len
        self.min_len     = min_len
        self.sep_token   = make_sep()
        self.sep_ids     = self.vocab.encode_token(self.sep_token)
        print(f"[Dataset] {len(all_files)} fichiers total -> "
              f"{len(self.files)} fichiers non-PDMX retenus pour l'harmonisation")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        kern_path = self.files[idx]
        try:
            # 1. Tokenize full score
            full_score = self.tokenizer.tokenize_file(kern_path)
            score_ids  = [self.vocab.encode_token(t) for t in full_score.tokens]
            if len(score_ids) < 10:
                raise ValueError("score too short")

            # 2. Extract melody (top-spine) -> kern string
            kern_text = kern_path.read_text(encoding="utf-8", errors="ignore")
            melody_kern = extract_top_spine(kern_text)
            if not melody_kern.strip():
                raise ValueError("melody empty")

            # 3. Tokenize melody (via fichier temporaire)
            with tempfile.NamedTemporaryFile(suffix=".krn", delete=False,
                                              mode="w", encoding="utf-8") as tmp:
                tmp.write(melody_kern)
                tmp_path = Path(tmp.name)
            try:
                melody_score = self.tokenizer.tokenize_file(tmp_path)
                melody_ids   = [self.vocab.encode_token(t) for t in melody_score.tokens]
            finally:
                try: tmp_path.unlink()
                except OSError: pass

            if len(melody_ids) < 5:
                raise ValueError("melody too short")

            # 4. Build paired sequence
            ids = melody_ids + [self.sep_ids] + score_ids
            prefix_len = len(melody_ids) + 1   # melody + SEP

            # 5. Si sequence trop longue : crop sur le score uniquement
            if len(ids) > self.max_seq_len:
                budget_score = self.max_seq_len - prefix_len
                if budget_score < 20:
                    # melodie deja trop longue, on crop tout proportionnellement
                    half = self.max_seq_len // 2
                    melody_ids = melody_ids[:half - 1]
                    score_ids  = score_ids[:half]
                    ids        = melody_ids + [self.sep_ids] + score_ids
                    prefix_len = len(melody_ids) + 1
                else:
                    # On garde la melodie complete + on crop le debut du score
                    # (Alt : crop ailleurs ; pour l'instant on garde le debut du score
                    #  qui correspond rythmiquement au debut de la melodie)
                    score_ids = score_ids[:budget_score]
                    ids       = melody_ids + [self.sep_ids] + score_ids

        except Exception:
            # Sequence nulle : sera filtree par le masque (longueur min)
            ids        = [[0] * len(HEAD_NAMES) for _ in range(self.min_len)]
            prefix_len = 1

        T = len(ids)
        tokens = {h: torch.zeros(T, dtype=torch.long) for h in HEAD_NAMES}
        for t, tok_ids in enumerate(ids):
            for hi, h in enumerate(HEAD_NAMES):
                tokens[h][t] = tok_ids[hi]
        return tokens, T, prefix_len


def collate_fn(batch):
    tokens_list, lengths, prefix_lens = zip(*batch)
    max_len = max(lengths)
    padded = {h: torch.zeros(len(batch), max_len, dtype=torch.long) for h in HEAD_NAMES}
    for i, (tok, L) in enumerate(zip(tokens_list, lengths)):
        for h in HEAD_NAMES:
            padded[h][i, :L] = tok[h]
    return (
        padded,
        torch.tensor(lengths,     dtype=torch.long),
        torch.tensor(prefix_lens, dtype=torch.long),
    )


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    total, n = 0.0, 0
    for tokens, lengths, prefix_lens in loader:
        tokens  = {h: v.to(device) for h, v in tokens.items()}
        lengths = lengths.to(device)
        prefix_lens = prefix_lens.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = model.compute_loss(tokens, pad_lengths=lengths,
                                       prefix_lengths=prefix_lens)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, step, ckpt_dir):
    name = f"ckpt_harm_ep{epoch + 1:03d}_step{step:06d}.pt"
    path = ckpt_dir / name
    torch.save({
        "epoch": epoch, "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": model.cfg,
    }, path)
    print(f"  [ckpt] {path}", flush=True)


def cleanup_checkpoints(ckpt_dir: Path, glob_pattern: str = "ckpt_harm_ep*.pt",
                        keep_every: int = 5, keep_recent: int = 3) -> None:
    all_ckpts = list(ckpt_dir.glob(glob_pattern))
    regular = [p for p in all_ckpts if "_mid" not in p.name]

    def epoch_of(p):
        m = re.search(r"ep(\d+)", p.name)
        return int(m.group(1)) if m else -1

    regular.sort(key=epoch_of)
    keep = set(regular[-keep_recent:])
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


# ── Entrainement ─────────────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[config] device={device}  amp={use_amp}  lr={args.lr}")

    base_ckpt = Path(args.base_ckpt).expanduser()
    if not base_ckpt.is_file():
        print(f"ERR: base checkpoint introuvable : {base_ckpt}"); sys.exit(1)

    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reprend depuis un checkpoint harm si present, sinon depuis le base ckpt v2
    harm_ckpts = sorted(ckpt_dir.glob("ckpt_harm_ep*.pt"), key=lambda p: p.stat().st_mtime)
    resume_path = harm_ckpts[-1] if harm_ckpts else base_ckpt
    print(f"[init] Chargement depuis {resume_path}")

    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"] if isinstance(ckpt.get("cfg"), ModelConfig) else ModelConfig()
    model = CompoundWordTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[init] {model.count_params():,} parametres")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    global_step = 0
    if resume_path != base_ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except Exception: pass
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step",  0)
    # Re-impose le lr (les ckpts peuvent avoir un lr different)
    for g in optimizer.param_groups:
        g["lr"] = args.lr

    kern_dir = Path(args.kern_dir).expanduser()
    dataset  = HarmonizeDataset(kern_dir, max_seq_len=cfg.max_seq_len)
    if len(dataset) == 0:
        print("[error] dataset vide"); sys.exit(1)

    n_val   = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    # Seed DIFFERENT du finetune_humdrum pour eviter le meme val split
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(13))
    print(f"[config] train={n_train}  val={n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=use_amp,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    last_ckpt_time = time.time()
    CKPT_INTERVAL  = args.ckpt_every * 60

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss, n_steps = 0.0, 0
        t_epoch = time.time()

        for step, (tokens, lengths, prefix_lens) in enumerate(train_loader):
            tokens      = {h: v.to(device) for h, v in tokens.items()}
            lengths     = lengths.to(device)
            prefix_lens = prefix_lens.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.compute_loss(tokens, pad_lengths=lengths,
                                           prefix_lengths=prefix_lens) / args.grad_accum

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

            if (step + 1) % 100 == 0:
                avg   = epoch_loss / n_steps
                speed = n_steps / (time.time() - t_epoch)
                print(f"  harm{epoch + 1:03d} | step {step + 1:5d}/{len(train_loader)} "
                      f"| loss {avg:.4f} | {speed:.1f} it/s", flush=True)

        val_loss = evaluate(model, val_loader, device, use_amp)
        print(f"[harm {epoch + 1:03d}/{args.epochs}] "
              f"train={epoch_loss / n_steps:.4f}  val={val_loss:.4f}  "
              f"time={time.time() - t_epoch:.0f}s", flush=True)

        save_checkpoint(model, optimizer, epoch, global_step, ckpt_dir)
        cleanup_checkpoints(ckpt_dir)

    print("[done] Fine-tune harmonisation termine.", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fine-tuning CWT pour l'harmonisation")
    p.add_argument("--kern-dir",       default="../kern_all")
    p.add_argument("--base-ckpt",      required=True,
                   help="checkpoint v2 a fine-tuner (ex: checkpoints/ckpt_ft_ep015_step359042.pt)")
    p.add_argument("--checkpoint-dir", default="checkpoints_harmonize")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--ckpt-every",     type=int,   default=30)
    p.add_argument("--num-workers",    type=int,   default=4)
    train(p.parse_args())


if __name__ == "__main__":
    main()
