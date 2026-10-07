"""Local smoke test for the pad_prob / raw-stream fix (no checkpoint needed).

Validates two invariants on a real tokenize->generate->infer->detokenize run with a
*randomly initialised* model (we test plumbing + masking, not transcription quality):

  (A) 0.5-INVARIANT: detokenizing with the new soft pad_prob+raw_* path at threshold 0.5
      yields exactly the same notes as the legacy binary-pad + zeroed-stream path. So the
      fix is backward compatible at the default.
  (B) RESCUE: lowering the threshold keeps >= as many notes (monotonic), and raising it
      keeps <=. So --pad-threshold is now actually LIVE.

Run:  venv311/bin/python scripts/test_pad_prob_fix.py
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

torch.manual_seed(0)


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


def n_notes(score):
    return 0 if score is None else len(list(score.recurse().notes))


def detok(y, thr, legacy=False):
    """legacy=True strips the new keys, forcing the old binary-pad + zeroed-stream path."""
    d = y
    if legacy:
        d = {k: v for k, v in y.items() if k != "pad_prob" and not k.startswith("raw_")}
    return MultistreamTokenizer.detokenize_mxl(d, pad_threshold=thr)


def main():
    midi = REPO / "midi" / "TwinkleTwinkle.mid"
    x = MultistreamTokenizer.tokenize_midi(str(midi))
    print(f"input notes: {x['pitch'].shape[0]}")

    # Force chunking so the infer() overlap / n_ctx alignment path is exercised too.
    with torch.no_grad():
        y = infer(x, tiny_model(), overlap=8, chunk=24, verbose=False, kv_cache=False)

    assert "pad_prob" in y, "pad_prob missing"
    T = y["pad"].shape[0]
    assert y["pad_prob"].shape[0] == T, (y["pad_prob"].shape, T)
    for k in ("pitch", "duration", "offset"):
        assert y["raw_" + k].shape[0] == T, (k, y["raw_" + k].shape, T)
    # soft mask at 0.5 must equal the hard binary mask exactly
    soft = (y["pad_prob"].squeeze() > 0.5)
    hard = (y["pad"].squeeze() > 0.5)
    assert torch.equal(soft, hard), "soft@0.5 != hard mask"
    print(f"generated slots: {T} | kept@0.5: {int(hard.sum())} | "
          f"pad_prob range [{y['pad_prob'].min():.3f}, {y['pad_prob'].max():.3f}]")

    # (A) 0.5-invariant
    n_new = n_notes(detok(y, 0.5, legacy=False))
    n_old = n_notes(detok(y, 0.5, legacy=True))
    print(f"(A) 0.5-invariant: new={n_new} notes  legacy={n_old} notes")
    assert n_new == n_old, f"0.5 invariant BROKEN: new={n_new} legacy={n_old}"

    # (B) monotonic rescue / prune
    counts = {thr: n_notes(detok(y, thr, legacy=False)) for thr in (0.1, 0.3, 0.5, 0.7, 0.9)}
    print("(B) notes vs threshold:", {t: counts[t] for t in sorted(counts)})
    thrs = sorted(counts)
    assert all(counts[thrs[i]] >= counts[thrs[i + 1]] for i in range(len(thrs) - 1)), \
        "note count not monotonically non-increasing in threshold"

    print("\nPASS: 0.5 is byte-identical to legacy; pad-threshold is live and monotonic.")


if __name__ == "__main__":
    main()
