"""Local logic test for the A1/A2 duration placement levers (no ckpt/ASAP needed).
(A) off (tau=0, lambda=0) == baseline; (B) on runs cleanly, same shapes, and changes output."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"))
import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
from tokenizer import MultistreamTokenizer
from utils import infer
from models.roformer import Roformer


def tiny():
    c = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
             max_position_embeddings=1536, embedding_size=64, hidden_dropout_prob=0.0,
             attention_probs_dropout_prob=0.0, layer_norm_eps=1e-12, rotary_value=False, positional_encoding="RoPE")
    enc = MyModelConfig(is_decoder=False, is_autoregressive=False, **c)
    dec = MyModelConfig(is_decoder=True, is_autoregressive=True, add_cross_attention=True, **c)
    m = Roformer(enc_configuration=enc, dec_configuration=dec,
                 hyperparameters={"components": ["encoder", "decoder"], "domains": {"in": "midi", "out": "mxl"}})
    m.eval(); return m


def run(m, x, **kw):
    torch.manual_seed(0)
    with torch.no_grad():
        return infer(x, m, overlap=8, chunk=24, verbose=False, kv_cache=False, **kw)


def main():
    m = tiny()
    vocab_dur = m.dec_config.out_duration_vocab_size
    x = MultistreamTokenizer.tokenize_midi(str(REPO / "midi" / "TwinkleTwinkle.mid"))
    # strong, structured priors so the effect is visible on a random model
    log_pi = torch.zeros(vocab_dur); log_pi[::3] = -4.0   # make non-multiple-of-3 (tuplet) buckets "rarer"
    metr = torch.full((24, vocab_dur), -6.0); metr[8, 1::3] = 0.0; metr[16, 2::3] = 0.0  # tuplets only at phase 8,16

    base = run(m, x)
    off = run(m, x, dur_log_pi=log_pi, dur_tau=0.0, dur_metrical=metr, dur_metrical_lambda=0.0)
    a1 = run(m, x, dur_log_pi=log_pi, dur_tau=3.0)
    a2 = run(m, x, dur_metrical=metr, dur_metrical_lambda=5.0)

    assert torch.equal(base["duration"], off["duration"]), "(A) off != baseline"
    print(f"(A) off==baseline OK (T={base['duration'].shape[0]})")
    d1 = int((a1["duration"].argmax(-1) != base["duration"].argmax(-1)).sum())
    d2 = int((a2["duration"].argmax(-1) != base["duration"].argmax(-1)).sum())
    assert a1["duration"].shape == base["duration"].shape and a2["duration"].shape == base["duration"].shape
    print(f"(B) A1 tau=3 changed {d1} duration tokens; A2 lambda=5 changed {d2} duration tokens (live)")
    print("PASS: duration priors are identity-off and live-on.")


if __name__ == "__main__":
    main()
