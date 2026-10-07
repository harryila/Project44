"""Direct generation check: does a checkpoint GENERATE non-empty scores, or collapse to
all-pad (the from-zero bug)? Bypasses MUSTER (which crashes on empty scores)."""
import warnings, sys, os
warnings.simplefilter("ignore")
sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")
os.chdir("MIDI2ScoreTransformer")
import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer
from utils import infer
from score_utils import postprocess_score
sys.path.insert(0, "../benchmark")
from eval_tier1_asap import load_any_checkpoint, collect_paths
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import device as _device

ck = sys.argv[1]
m = load_any_checkpoint(ck, "cuda")
m.eval().to(_device())
for comp in ["Mozart", "Bach", "Scriabin"]:
    cands = [x for x in collect_paths("test") if comp in x["composer"]]
    if not cands:
        continue
    p = cands[0]
    x = MultistreamTokenizer.tokenize_midi(p["midi"])
    y = infer(x, m, overlap=64, chunk=512, verbose=False, kv_cache=True)
    pad = y["pad"]
    kept = int((pad > 0.5).sum()) if pad.dtype.is_floating_point else int(pad.sum())
    sc = postprocess_score(MultistreamTokenizer.detokenize_mxl(y), inPlace=True)
    nn = len(list(sc.recurse().notes)) if sc is not None else 0
    verdict = "GENERATES" if nn > 10 else "COLLAPSED"
    print(f"{comp}: input={x['pitch'].shape[0]} kept_notes={kept} detok_notes={nn} -> {verdict}", flush=True)
