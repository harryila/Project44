"""Pre-build the ASAP parse cache in parallel. Uses split='all' because the
'train' split's __getitem__ ignores the index (random multinomial resample), so
indexing it wouldn't deterministically cache every item. 'all' uses the real idx,
and the cache pkl is keyed by sample_path+id (augmentation-independent), so it
serves the train/val loaders too.
"""
import os, sys, warnings, time
warnings.simplefilter("ignore")
sys.path.insert(0, "midi2scoretransformer")
from joblib import Parallel, delayed
from dataset import ASAPDataset

ds = ASAPDataset("./data/", "all", seq_length=512, padding="per-beat", augmentations={})
n = len(ds)
print(f"[all] {n} ASAP performances — building cache (32 workers)...", flush=True)

def touch(i):
    try:
        ds[i]
        return 1
    except Exception as e:
        return f"ERR {i}: {type(e).__name__}: {e}"

t0 = time.time()
res = Parallel(n_jobs=32, verbose=5)(delayed(touch)(i) for i in range(n))
ok = sum(1 for r in res if r == 1)
errs = [r for r in res if r != 1]
print(f"DONE cached {ok}/{n}, errors {len(errs)} in {time.time()-t0:.0f}s", flush=True)
for e in errs[:8]:
    print("  ", e)
print(f"cache pkls now: {len(os.listdir('data/cache'))}")
