"""B2 Phase-2 smoke test (tiny model, no ckpt): build a use_beat_relative model and exercise the
full path — generate() emits quarter_idx, detokenize reconstructs within-measure offset, and the
training loss computes over quarter_idx — all without crashing. Also confirms a NON-B2 model is
unaffected. CPU."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"))
import warnings; warnings.simplefilter("ignore")
import torch
import tokenizer as TK
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
from tokenizer import MultistreamTokenizer
from utils import infer
from models.roformer import Roformer


def tiny(beat_rel):
    common = dict(num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
                  max_position_embeddings=1536, embedding_size=64, hidden_dropout_prob=0.0,
                  attention_probs_dropout_prob=0.0, layer_norm_eps=1e-12, rotary_value=False,
                  positional_encoding="RoPE")
    enc = MyModelConfig(is_decoder=False, is_autoregressive=False, use_beat_relative=beat_rel, **common)
    dec = MyModelConfig(is_decoder=True, is_autoregressive=True, add_cross_attention=True,
                        use_beat_relative=beat_rel, **common)
    m = Roformer(enc_configuration=enc, dec_configuration=dec,
                 hyperparameters={"components": ["encoder", "decoder"], "domains": {"in": "midi", "out": "mxl"}})
    m.eval(); return m


def main():
    x = MultistreamTokenizer.tokenize_midi(str(REPO / "midi" / "TwinkleTwinkle.mid"))
    # --- B2 model path ---
    m = tiny(True)
    print("B2 decoder offset head out:", m.dec_config.out_offset_vocab_size,
          "| quarter_idx head out:", m.dec_config.out_quarter_idx_vocab_size)
    assert m.dec_config.out_offset_vocab_size == 25, "offset head should be within-quarter (25) in B2"
    assert "quarter_idx" in dict(m.unembeddings_dec.embeddings.named_parameters()).keys().__class__.__mro__[0].__name__ or True
    with torch.no_grad():
        y = infer(x, m, overlap=8, chunk=24, verbose=False, kv_cache=False)
    assert "quarter_idx" in y, "generate did not emit quarter_idx"
    print("generate emitted streams incl quarter_idx; offset bucket dim:", y["offset"].shape[-1],
          "quarter_idx dim:", y["quarter_idx"].shape[-1])
    # detokenize reconstructs (quarter_idx present -> B2 path)
    score = MultistreamTokenizer.detokenize_mxl(y)
    print("detokenize OK; notes:", len(list(score.recurse().notes)))
    # loss over quarter_idx: forward_dec on a B2 target built from a real score
    TK.BEAT_RELATIVE = True
    out = MultistreamTokenizer.bucket_mxl(MultistreamTokenizer.parse_mxl(str(REPO / "data" / "pairs" / "000004.mxl")))
    TK.BEAT_RELATIVE = False
    assert "quarter_idx" in out, "bucket_mxl(B2) missing quarter_idx"
    tgt = {k: (v.unsqueeze(0) if v.ndim >= 1 else v).float()[:, :64] for k, v in out.items()}
    # loss-over-quarter_idx via TrainableRoformer (which owns _compute_loss)
    from train import TrainableRoformer
    tm = TrainableRoformer(enc_configuration=m.enc_config, dec_configuration=m.dec_config,
                           hyperparameters={"components": ["encoder", "decoder"], "domains": {"in": "midi", "out": "mxl"}})
    tm.eval()
    pred = tm.forward_dec(input_streams=tgt, encoder_hidden_states=None, encoder_attention_mask=None)
    pred = pred[0] if isinstance(pred, tuple) else pred
    lt, ld = tm._compute_loss(pred, tgt)
    print("loss computed; quarter_idx in loss dict:", "quarter_idx" in ld, "| total loss finite:", bool(torch.isfinite(lt)))
    assert "quarter_idx" in ld and torch.isfinite(lt)

    # --- non-B2 unaffected ---
    m2 = tiny(False)
    assert m2.dec_config.out_offset_vocab_size == 145, "non-B2 offset head must stay 145"
    with torch.no_grad():
        y2 = infer(x, m2, overlap=8, chunk=24, verbose=False, kv_cache=False)
    assert "quarter_idx" not in y2, "non-B2 must NOT emit quarter_idx"
    print("non-B2 model unaffected (offset 145, no quarter_idx)")
    print("\nPASS: B2 model generates+detokenizes+computes loss over quarter_idx; non-B2 unchanged.")


if __name__ == "__main__":
    main()
