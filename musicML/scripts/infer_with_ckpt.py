"""Run MIDI2ScoreTransformer inference with a chosen checkpoint on a single MIDI.

Mirrors the transformer-backend inference path in `transcribe.py` but lets us
pass the checkpoint path explicitly, so we can A/B the released baseline vs
the synthetic pretrain checkpoint without touching the production CLI.
"""

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TF_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF_DIR))

import torch  # noqa: E402

# Patch like transcribe.py does
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer  # noqa: E402
from score_utils import postprocess_score  # noqa: E402
from models.roformer import Roformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi", type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pad-threshold", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()

    device = "cpu"
    log.info("Loading checkpoint: %s", args.ckpt)
    model = Roformer.load_from_checkpoint(
        str(args.ckpt), map_location=device, weights_only=False,
    )
    model.eval()
    model.to(device)

    log.info("Tokenizing %s", args.midi)
    x = MultistreamTokenizer.tokenize_midi(str(args.midi))
    log.info("  %d notes", x["pitch"].shape[0])

    log.info("Running inference (chunk=%d, overlap=%d, top_k=%d, T=%.2f)",
             args.chunk_size, args.overlap, args.top_k, args.temperature)
    y_hat = infer(x, model, verbose=False, kv_cache=True,
                  overlap=args.overlap, chunk=args.chunk_size,
                  top_k=args.top_k, temperature=args.temperature)

    log.info("Detokenizing (pad_threshold=%.2f)", args.pad_threshold)
    diagnostics = []
    mxl = MultistreamTokenizer.detokenize_mxl(
        y_hat, _diagnostics=diagnostics, pad_threshold=args.pad_threshold,
    )
    mxl = postprocess_score(mxl, inPlace=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mxl.write("musicxml", fp=str(args.out))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
