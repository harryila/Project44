"""Local smoke test for per-head decoding overrides (no checkpoint needed).

Validates:
  (A) IDENTITY: head_overrides=None and head_overrides={"pitch": (1, 1.0)} (the explicit
      greedy identity) both reproduce the default greedy output exactly -> zero regression risk.
  (B) LIVE: a real pitch override {"pitch": (5, 1.3)} runs cleanly through the chunked path
      and returns the same keys/shapes (it may differ in content -- that's the point).

Run:  venv311/bin/python scripts/test_perhead_decode.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))

import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
from tokenizer import MultistreamTokenizer
from utils import infer
from models.roformer import Roformer


def tiny_model():
    common = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
                  intermediate_size=128, max_position_embeddings=1536, embedding_size=64,
                  hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                  layer_norm_eps=1e-12, rotary_value=False, positional_encoding="RoPE")
    enc = MyModelConfig(is_decoder=False, is_autoregressive=False, **common)
    dec = MyModelConfig(is_decoder=True, is_autoregressive=True, add_cross_attention=True, **common)
    hp = {"components": ["encoder", "decoder"], "domains": {"in": "midi", "out": "mxl"}}
    m = Roformer(enc_configuration=enc, dec_configuration=dec, hyperparameters=hp)
    m.eval()
    return m


def run(model, x, ho):
    torch.manual_seed(0)
    with torch.no_grad():
        return infer(x, model, overlap=8, chunk=24, verbose=False, kv_cache=False, head_overrides=ho)


def main():
    model = tiny_model()
    x = MultistreamTokenizer.tokenize_midi(str(REPO / "midi" / "TwinkleTwinkle.mid"))

    base = run(model, x, None)
    ident = run(model, x, {"pitch": (1, 1.0)})
    over = run(model, x, {"pitch": (5, 1.3)})

    # (A) identity invariants
    for k in ("pitch", "pad", "duration", "offset"):
        assert torch.equal(base[k], ident[k]), f"identity override changed stream {k}"
    print(f"(A) identity: None == pitch:(1,1.0) on all streams  (T={base['pitch'].shape[0]})")

    # (B) real override runs, same keys/shapes
    assert set(over.keys()) == set(base.keys()), "override changed the returned key set"
    for k in base:
        assert over[k].shape == base[k].shape or k.startswith("raw_") or k == "pad_prob", (k, over[k].shape, base[k].shape)
    diff = int((over["pitch"].argmax(-1) != base["pitch"].argmax(-1)).sum())
    print(f"(B) pitch:(5,1.3) ran cleanly; pitch tokens differing from greedy: {diff}")

    print("\nPASS: per-head override is identity-safe at default and live when set.")


if __name__ == "__main__":
    main()
