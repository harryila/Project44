"""Calibration A — does our custom trainer destabilize a WORKING checkpoint?

Warm-start from the released MIDI2ScoreTF.ckpt (which generates fine), run a short
fine-tune on the synthetic PDMX pairs with our trainer, and check generation BEFORE
and AFTER. If the model still emits non-empty scores after training, the warm-start
fine-tune path for Track C is de-risked (the cold-start collapse is sidestepped). If
it collapses, our trainer actively destroys generation and must be fixed first.

The trainer's known issues (from the bug analysis) are toggleable here so we can test
as-is vs. the recommended fix (unconditional_dropout 0 for paired-only data):
    --unconditional-dropout 0.5   (trainer default; zeroes encoder MIDI 50% — suspect)
    --unconditional-dropout 0.0   (recommended fix for paired-only fine-tune)

Forces CPU (MPS breaks this model). Run from the repo root.
"""
import argparse, glob, os, sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")

REPO = Path("/Users/harry/Desktop/temp/musicML")
TF = REPO / "MIDI2ScoreTransformer/midi2scoretransformer"
sys.path.insert(0, str(TF))
os.chdir(REPO)  # synthetic pair paths + cache are relative to repo root

import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from train import load_pretrained_init, make_pdmx_loaders  # noqa: E402
from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer, eval as eval_pair  # noqa: E402

RELEASED = "MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt"
# short ASAP test piece for the generation check
BACH_DIR = REPO / "MIDI2ScoreTransformer/data/asap-dataset/Bach/Fugue/bwv_846"


def find_test_piece():
    gt = BACH_DIR / "xml_score.musicxml"
    mids = [m for m in glob.glob(str(BACH_DIR / "*.mid")) if "score" not in os.path.basename(m).lower()]
    return mids[0], str(gt)


def gen_check(model, test_midi, gt, label):
    model.eval()
    with torch.no_grad():
        x = MultistreamTokenizer.tokenize_midi(test_midi)
        y_hat = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
    pad = y_hat["pad"].squeeze().float()
    n_nonpad = int((pad > 0.5).sum())
    total = int(pad.numel())
    mer = None
    try:
        sim = eval_pair(y_hat, gt)
        mer = (sim.get("muster") or {}).get("MeanER")
    except Exception as e:
        mer = f"score-err: {e}"
    status = "GENERATES" if n_nonpad > total * 0.05 else "COLLAPSED (empty)"
    print(f"  [{label}] non-pad notes={n_nonpad}/{total}  MeanER={mer}  -> {status}", flush=True)
    return {"n_nonpad": n_nonpad, "total": total, "MeanER": mer, "status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--input-dropout", type=float, default=0.75)
    ap.add_argument("--unconditional-dropout", type=float, default=0.5)
    ap.add_argument("--manifest", default="data/pairs/_manifest.csv")
    ap.add_argument("--out", default="MIDI2ScoreTransformer/checkpoints/calib_a.ckpt")
    args = ap.parse_args()

    test_midi, gt = find_test_piece()
    print(f"Test piece: {os.path.relpath(test_midi, REPO)}")
    print(f"Warm-start from: {RELEASED}")
    print(f"Config: steps={args.steps} lr={args.lr} batch={args.batch_size} "
          f"input_dropout={args.input_dropout} unconditional_dropout={args.unconditional_dropout}\n")

    model = load_pretrained_init(RELEASED)
    assert model is not None, "warm-start failed"
    model._input_dropout = args.input_dropout
    model._unconditional_dropout = args.unconditional_dropout
    model.to("cpu")

    print("PRE-TRAIN generation check (the released model, warm-started):")
    pre = gen_check(model, test_midi, gt, "pre")

    print(f"\nFine-tuning {args.steps} steps (replicates training_step)...")
    train_loader, _ = make_pdmx_loaders(args.manifest, batch_size=args.batch_size, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.2)
    model.train()
    t0 = time.time(); step = 0; last = None
    for batch in train_loader:
        if step >= args.steps:
            break
        inp, out = batch
        inp = model._maybe_drop_input(inp)          # training_step line 165
        dec_in = model._maybe_drop_decoder(out)     # training_step line 168
        pred = model.forward(input_streams=inp, output_streams=dec_in)
        loss, _ = model._compute_loss(pred, out)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # fit() gradient_clip_val=1.0
        opt.step()
        last = float(loss.detach())
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss={last:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        step += 1
    print(f"  trained {step} steps in {time.time()-t0:.0f}s, final loss={last:.4f}")

    print("\nPOST-TRAIN generation check (after our trainer touched it):")
    post = gen_check(model, test_midi, gt, "post")

    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"\nSaved fine-tuned weights: {args.out}")

    print("\n" + "=" * 64)
    print("CALIBRATION A VERDICT")
    print("=" * 64)
    print(f"  pre : {pre['status']}  (non-pad {pre['n_nonpad']}/{pre['total']}, MeanER {pre['MeanER']})")
    print(f"  post: {post['status']}  (non-pad {post['n_nonpad']}/{post['total']}, MeanER {post['MeanER']})")
    if post["n_nonpad"] > post["total"] * 0.05:
        print("  => Trainer PRESERVES generation. Warm-start fine-tune path is DE-RISKED.")
    else:
        print("  => Trainer DESTROYS generation. Must fix trainer (try --unconditional-dropout 0) before GPU.")


if __name__ == "__main__":
    main()
