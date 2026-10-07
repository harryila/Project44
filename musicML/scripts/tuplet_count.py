"""MUSTER-free tuplet check: for each test piece, generate -> detokenize -> count tuplets in
the predicted score vs the GT. Answers "does the model PRODUCE tuplets?" without the slow/hangy
MUSTER scorer. Usage: tuplet_count.py <ckpt> <Composer1> <Composer2> ..."""
import warnings, sys, os, signal
warnings.simplefilter("ignore")
sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")
os.chdir("MIDI2ScoreTransformer")
import torch


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)  # per-piece guard against music21 hangs
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer
from utils import infer
from score_utils import postprocess_score
sys.path.insert(0, "../benchmark")
from eval_tier1_asap import load_any_checkpoint, collect_paths
from music21 import converter
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import device as _device

ck = sys.argv[1]
filters = sys.argv[2:]
m = load_any_checkpoint(ck, _device()).eval().to(_device())
paths = collect_paths("test")


def n_tup(score):
    if score is None:
        return 0, 0
    t = n = 0
    for nt in score.recurse().notes:
        n += 1
        try:
            if nt.duration.tuplets:
                t += 1
        except Exception:
            pass
    return t, n


for f in filters:
    cands = [p for p in paths if f.lower() in (p["composer"] + "/" + p["piece"]).lower()]
    if not cands:
        print(f"{f}: no match"); continue
    p = sorted(cands, key=lambda q: 0)[0]
    try:
        signal.alarm(240)
        x = MultistreamTokenizer.tokenize_midi(p["midi"])
        y = infer(x, m, overlap=64, chunk=512, verbose=False, kv_cache=True)
        pred = postprocess_score(MultistreamTokenizer.detokenize_mxl(y), inPlace=True)
        pt, pn = n_tup(pred)
        gt, gn = n_tup(converter.parse(p["score"]))
        signal.alarm(0)
        ratio = pt / gt if gt else None
        print(f"{f:11s} pred_tup={pt:5d}/{pn:5d}  gt_tup={gt:5d}/{gn:5d}  ratio={ratio}", flush=True)
    except _Timeout:
        signal.alarm(0)
        print(f"{f:11s} TIMEOUT (music21 hang) — skipped", flush=True)
    except Exception as e:
        signal.alarm(0)
        print(f"{f:11s} ERROR {type(e).__name__}: {e}", flush=True)
