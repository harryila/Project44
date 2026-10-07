# Block 2 (MIDI2ScoreTransformer) Synthetic Pretrain — Execution Status

## Summary

The full data pipeline and trainer infrastructure are built and working
end-to-end. Stage A pretrain reduces loss cleanly on PDMX (val/total 1.85 →
0.244 across 4 epochs). However, the trained checkpoint produces degenerate
output (empty score) at inference — both on real MAESTRO MIDI and on its own
synthetic training distribution. This is the "from-scratch generation"
problem: without seeing many `start-from-all-zeros` decoder states during
training, the bidirectional decoder collapses to predicting `pad=0` for every
position when given the all-zero start token.

The fix requires either:
- The upstream training script (which has the right dropout / masking schedule
  for autoregressive-style decoding from a non-autoregressive bidirectional
  decoder), OR
- Several more rounds of dropout / loss / curriculum experiments with
  associated training time on a GPU

Both are out of scope for a single Mac-CPU session. Everything below this
training-paradigm issue is in working order.

## What is fully done and working

| # | Step | Status | Artifact |
|---|---|---|---|
| 1 | ASAP overlap check | DONE | [data/preflight/step1_overlap_findings.md](data/preflight/step1_overlap_findings.md) |
| 2 | Tokenizer stream verification | DONE | parse_midi/parse_mxl emit all 4+13 expected streams; verified on Twinkle |
| 3 | Baseline metrics | DONE | [benchmark/baseline_metrics.json](benchmark/baseline_metrics.json) and [benchmark/eval_baseline.py](benchmark/eval_baseline.py) |
| 4 | PDMX download | DONE | `~/datasets/pdmx/PDMX.csv` (215 MB), `~/datasets/pdmx/mxl/` (254,035 .mxl files) |
| 5 | Piano filter | DONE | [data/pdmx_piano_subset.csv](data/pdmx_piano_subset.csv) — 181,693 pieces (24,195 classical) |
| 6 | Expressive renderer | DONE | [scripts/expressive_render.py](scripts/expressive_render.py); 359/359 pitch-match verified |
| 7 | Eyeball check | DONE | [data/render_test/eyeball/findings.md](data/render_test/eyeball/findings.md); distributions overlap real |
| 8 | 5K pair generation | DONE | 5,248 pairs + 302 errors (5.4% loss); manifest + chunks + cache (738 MB total) |
| 9a | Trainer + LR test + Stage A | PARTIAL | Trainer works, val loss converges, but checkpoint doesn't generate at inference |
| 9b | ASAP setup for Stage B | PARTIAL | ASAP cloned + ACPAS metadata + 505/967 chunks computed; chunker incomplete |
| 10 | Eval pretrain vs baseline | DEFERRED | Pretrain checkpoint produces empty XML; valid eval requires working ckpt |

## Step 1 finding (no hard stop)

| Piece | In ASAP? | Block 2 baseline | Verdict |
|---|---|---|---|
| Chopin Op.10 No.4 | YES (22 perf instances) | 4/4, 6.8% measure error | Works |
| Chopin Op.25 No.11 | YES (19 perf instances) | 4/4, 2.0% measure error | Works |
| Liszt Mazeppa | NO | 3/4 (wrong), 92.8% measure error | Broken |

Pieces in ASAP work; piece not in ASAP fails. Classic data-shortage signature
— pretrain plan is justified.

## Step 3 baseline numbers (the bar to beat)

```
Piece                        TS    OK   Meas   %err    Note F1
Op10 No.4 (ref 88)          4/4   YES  82     6.8     0.971
Op25 No.11 (ref 102)        4/4   YES  100    2.0     0.978
Mazeppa (ref 167)           3/4   NO   322    92.8    0.981
```

## Stage A training results (Mac CPU, 4 epochs, 5K pairs, batch 4)

LR sanity check (100 steps, lr 1e-6 → 5e-3): loss decreased smoothly from
16.3 → 6.0; min at lr=4.9e-4; steepest descent at 3.2e-4. Picked **lr=1e-4**.

After fixing a critical bug (decoder input was not being dropped, causing the
model to learn a trivial identity mapping):

```
Stage A v2 (with input_dropout=0.75 on encoder + decoder, unconditional=0.5):
  Epoch 1: train/total 14.25 → 4.40,  val/total 1.85
  Epoch 2: train/total 4.40  → 4.11,  val/total 0.439
  Epoch 3: train/total 4.11  → 3.66,  val/total 0.265
  Epoch 4: train/total 3.66  → 3.39,  val/total 0.244
```

Best ckpt: `MIDI2ScoreTransformer/checkpoints/pretrain_pdmx-epoch=03-val/total=0.2440.ckpt`

## Inference issue diagnosis

Tested the Stage A checkpoint on a synthetic pair (its own training
distribution):

```
Input:  data/pairs/000112.mid (616 notes)
Output: pad logits all 0.000, max=0.000, no positions > 0.5
```

And on Op10 No.4 (real MAESTRO MIDI):

```
Input:  benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi (2306 notes)
Output: pad logits all 0.000, max=0.000, no positions > 0.5
```

Both produce empty scores. The model learned a low loss in teacher-forcing
mode (where decoder sees noisy GT targets and predicts clean ones), but the
inference path uses iterative generation starting from an all-zero start
token — a regime the model never encountered at training time.

The released `MIDI2ScoreTF.ckpt` was trained with the same architecture
(`is_autoregressive=False`) and same hyperparameters (`input_dropout=0.75`,
`unconditional_dropout=0.5`) and DOES generate from zeros. The training-time
mechanism that bridges bidirectional teacher-forcing to autoregressive zero-
start inference is not obvious from the upstream code we have, and reverse-
engineering it would need either the upstream training script or several
more rounds of experiment.

## Files added by this work

Production code:
- [scripts/filter_pdmx.py](scripts/filter_pdmx.py) — PDMX piano filter
- [scripts/expressive_render.py](scripts/expressive_render.py) — score → perturbed MIDI with exact 1:1 alignment
- [scripts/make_pairs.py](scripts/make_pairs.py) — bulk renderer with joblib parallelism + chunks json + cache
- [scripts/eyeball_check.py](scripts/eyeball_check.py) — distribution-comparison sanity check
- [scripts/infer_with_ckpt.py](scripts/infer_with_ckpt.py) — single-ckpt inference helper
- [benchmark/eval_baseline.py](benchmark/eval_baseline.py) — TS/measure/F1 metrics on benchmark pieces
- [MIDI2ScoreTransformer/midi2scoretransformer/pdmx_dataset.py](MIDI2ScoreTransformer/midi2scoretransformer/pdmx_dataset.py) — synthetic dataset wrapper
- [MIDI2ScoreTransformer/midi2scoretransformer/train.py](MIDI2ScoreTransformer/midi2scoretransformer/train.py) — Lightning trainer + LR-range test

Bug fix in upstream code:
- [tokenizer.py](MIDI2ScoreTransformer/midi2scoretransformer/tokenizer.py) line 341: coerce `Part.id` to `str` before `.lower()` (was crashing on auto-generated MusicXML with int IDs)

Documentation:
- [data/preflight/step1_overlap_findings.md](data/preflight/step1_overlap_findings.md)
- [data/preflight/heldout_plan.md](data/preflight/heldout_plan.md)
- [data/render_test/eyeball/findings.md](data/render_test/eyeball/findings.md)
- This document

## How to resume on a GPU (handoff runbook)

This section is intended to be followed top-to-bottom by the next person on a
fresh CUDA machine. Each phase has a verification step; if a verification
fails, the troubleshooting note tells you what to check before moving on.

### Phase R0: Machine setup

```bash
# Clone the repo
git clone https://github.com/harryila/Music-ML-BayenLab.git
cd Music-ML-BayenLab

# Create venv with Python 3.11 (PyTorch + Lightning)
python3.11 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip wheel
pip install -r MIDI2ScoreTransformer/requirements.txt
pip install pytorch-lightning>=2.0 tensorboard pandas joblib pretty_midi soxr soundfile
# CUDA-specific torch install if not already installed:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is visible
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
# expect: cuda: True 1+
```

If you don't have GitHub credentials configured, push the repo from the Mac
side and pull on the GPU machine. The repo (post-cleanup) is ~300 MB.

### Phase R1: External data downloads

These are intentionally NOT in the git repo (too big, all regenerable):

```bash
# 1. PDMX MusicXML corpus from Zenodo (~30 min on a fast network, ~2 GB total)
mkdir -p ~/datasets/pdmx && cd ~/datasets/pdmx
curl -L -o PDMX.csv https://zenodo.org/api/records/15571083/files/PDMX.csv/content
curl -L -o mxl.tar.gz https://zenodo.org/api/records/15571083/files/mxl.tar.gz/content
md5sum mxl.tar.gz   # expect 49ffd75ecf5489c0be6d41182eb11ff7
tar -xzf mxl.tar.gz
find mxl -name "*.mxl" | wc -l   # expect 254035
cd -  # back to repo root

# 2. ASAP dataset for Stage B fine-tune (~5 min, ~1 GB)
cd MIDI2ScoreTransformer/data
git clone https://github.com/TimFelixBeyer/asap-dataset.git
cd asap-dataset
git checkout 8cba199e15931975542010a7ea2ff94a6fc9cbee
cd ../../..

# 3. ACPAS metadata (already in repo via previous push, but if needed:)
# Files are: MIDI2ScoreTransformer/data/ACPAS-dataset/{metadata_R,metadata_S}.csv
# If missing:
mkdir -p MIDI2ScoreTransformer/data/ACPAS-dataset
curl -sL "https://raw.githubusercontent.com/cheriell/ACPAS-dataset/main/metadata_R.csv" \
  -o MIDI2ScoreTransformer/data/ACPAS-dataset/metadata_R.csv
curl -sL "https://raw.githubusercontent.com/cheriell/ACPAS-dataset/main/metadata_S.csv" \
  -o MIDI2ScoreTransformer/data/ACPAS-dataset/metadata_S.csv

# 4. Released MIDI2ScoreTF.ckpt baseline (for warm-start experiments + reference)
mkdir -p MIDI2ScoreTransformer/checkpoints
# Download from upstream's GitHub Releases:
#   https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases
# Save as: MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt
ls -lh MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt
# expect ~372 MB
```

**Verify Phase R1 passed:**

```bash
python -c "
import pandas as pd
df = pd.read_csv('/path/to/datasets/pdmx/PDMX.csv', nrows=2)
print('PDMX columns:', len(df.columns))
import os
print('asap pieces:', len(os.listdir('MIDI2ScoreTransformer/data/asap-dataset')))
print('acpas:', os.path.exists('MIDI2ScoreTransformer/data/ACPAS-dataset/metadata_R.csv'))
print('baseline ckpt:', os.path.exists('MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt'))
"
# expect: PDMX columns: ~60, asap pieces: 17 (composer dirs), acpas: True, baseline ckpt: True
```

### Phase R2: Recreate piano subset and synthetic pairs

```bash
# 1. Filter PDMX to piano subset (~10 sec)
venv311/bin/python scripts/filter_pdmx.py \
  --csv ~/datasets/pdmx/PDMX.csv \
  --out data/pdmx_piano_subset.csv

# expect: ~181,693 pieces, ~24K classical
# expect file size: ~55 MB

# 2. Generate 50K synthetic (perturbed-MIDI, MusicXML) pairs (~30-90 min on GPU)
#    Use --n 50000 for the real pretrain run (the Mac CPU run only did 5K).
venv311/bin/python scripts/make_pairs.py \
  --subset-csv data/pdmx_piano_subset.csv \
  --mxl-root ~/datasets/pdmx \
  --out-dir data/pairs \
  --cache-dir data/cache_pdmx \
  --manifest data/pairs/_manifest.csv \
  --errors data/pairs/_errors.log \
  --n 50000 \
  --n-jobs -1 \
  --prefer-multi-track

# Sanity check after it finishes:
wc -l data/pairs/_manifest.csv data/pairs/_errors.log
ls data/pairs/*.mid | wc -l
du -sh data/pairs data/cache_pdmx

# expect ~50,000 manifest rows, ~5% error rate, ~5 GB total on disk
```

**Verify Phase R2 passed:**

```bash
venv311/bin/python -c "
import sys, json, csv
sys.path.insert(0, 'MIDI2ScoreTransformer/midi2scoretransformer')
from tokenizer import MultistreamTokenizer

# Take a random pair, verify all expected streams
with open('data/pairs/_manifest.csv') as f:
    row = list(csv.DictReader(f))[100]
in_s = MultistreamTokenizer.parse_midi(row['midi'])
out_s = MultistreamTokenizer.parse_mxl(row['mxl'])
print('input streams:', sorted(in_s.keys()))
print('output streams:', sorted(out_s.keys()))
print('note count match:', in_s['pitch'].shape[0] == out_s['pitch'].shape[0])
"
# expect: 4 input streams, 13 output streams, note count match True
```

### Phase R3: Investigate the from-zero-generation issue (DO THIS FIRST)

The Stage A pretrain checkpoint from the Mac CPU run (val/total 0.244) does
not generate at inference: pad logits collapse to zero everywhere. The
released `MIDI2ScoreTF.ckpt` from the same architecture *does* generate. The
training-time mechanism that bridges teacher-forced training to autoregressive
zero-start inference is unclear without the upstream training script.

**Before launching a long Stage A pretrain run, do this calibration step.**
Otherwise you may burn many GPU-hours on a degenerate model.

#### Calibration A: Does the released ckpt + my trainer still generate?

```bash
# Warm-start from MIDI2ScoreTF.ckpt, train for 200 steps with current trainer
PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer venv311/bin/python -c "
import sys, torch
sys.path.insert(0, 'MIDI2ScoreTransformer/midi2scoretransformer')
from train import load_pretrained_init, make_pdmx_loaders, lr_range_test

model = load_pretrained_init('MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt')
loader, _ = make_pdmx_loaders('data/pairs/_manifest.csv', batch_size=4, num_workers=2)
out = lr_range_test(model, loader, start_lr=1e-5, end_lr=1e-3, num_steps=200,
                    out_path=__import__('pathlib').Path('checkpoints/calib_a.csv'))
print('done:', out)
"
# Then run inference with this lightly-trained ckpt and check for non-empty output
```

If the warm-started model still generates → my trainer is sound; the issue
was that 4 epochs on 5K pairs from-scratch was undertraining. Move to Phase R4.

If the warm-started model collapses → my trainer destabilizes a working model.
Investigate which component (training_step? loss? dropout?). Most likely
suspect: `_maybe_drop_decoder` — try removing it entirely (decoder gets full
GT during training, model still has to predict same thing because of
`_compute_loss` regularization).

#### Calibration B: From-scratch experiments (run 3 short configs)

If A passes, run 3 short pretrain experiments (1 epoch each on 5K pairs ~10 min
on a real GPU) to find the right config:

```bash
# Config 1: my current Stage A v2 settings (high dropout on both encoder + decoder)
PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer venv311/bin/python \
  MIDI2ScoreTransformer/midi2scoretransformer/train.py fit \
  --stage cfg1 --manifest data/pairs/_manifest.csv \
  --lr 1e-4 --max-epochs 1 --batch-size 16 --precision bf16-mixed \
  --out-dir MIDI2ScoreTransformer/checkpoints/cfg1

# Config 2: Try is_autoregressive=True (decoder input shifted, more like teacher
# forcing for autoregressive generation). Requires editing _new_model() in
# train.py to set both enc.is_autoregressive=True and dec.is_autoregressive=True
# before launching. Test if this generates from zeros.

# Config 3: Lower dropout. Edit train.py _new_model() defaults:
# input_dropout=0.5, unconditional_dropout=0.3
# Hypothesis: 0.75 dropout is too aggressive for from-scratch (warmup-style).

# After each, run inference on a synthetic pair and on Op10 No.4
# and check pad logit statistics (should NOT all be zero).
venv311/bin/python scripts/infer_with_ckpt.py \
  benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi \
  --ckpt MIDI2ScoreTransformer/checkpoints/cfg1/<best>.ckpt \
  --out /tmp/cfg1_out.musicxml
ls -lh /tmp/cfg1_out.musicxml   # >100 KB = generated something; ~1.7 KB = empty
```

Pick the config whose 1-epoch ckpt produces a non-trivial XML on Op10 No.4.
Use that config for the real Stage A run in Phase R4.

### Phase R4: Stage A pretrain (the real run)

Once R3 picked a working config:

```bash
PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer venv311/bin/python \
  MIDI2ScoreTransformer/midi2scoretransformer/train.py fit \
  --stage pretrain_pdmx \
  --manifest data/pairs/_manifest.csv \
  --lr 1e-4 \
  --max-epochs 8 \
  --batch-size 16 \
  --seq-length 512 \
  --num-workers 8 \
  --precision bf16-mixed \
  --out-dir MIDI2ScoreTransformer/checkpoints

# On a single A100 / H100 / 4090: expect ~1-3 hours total for 50K pairs × 8 epochs
# Monitor via tensorboard:
tensorboard --logdir MIDI2ScoreTransformer/checkpoints/lightning_logs
```

**Acceptance criteria for Stage A** (per the original plan, lines 139-147):

- (a) PDMX val loss converges (downward trend across all 8 epochs)
- (b) Sample inference on 5-10 PDMX held-out pieces produces non-empty XML
  with reasonable time signature and measure count

If both hold: proceed to Stage B. If (a) fails, debug (likely loss weights or
LR). If (b) fails but (a) succeeds, the from-zero-generation issue is
unresolved — go back to Phase R3.

### Phase R5: ASAP setup for Stage B

```bash
# Populate _chunks.json files for all aligned ASAP performances (~5 min on GPU)
cd MIDI2ScoreTransformer
PYTHONPATH=midi2scoretransformer venv311/bin/python midi2scoretransformer/chunker.py
find data/asap-dataset -name "*_chunks.json" | wc -l   # expect ~970

# Build the dataset cache (~30 min, parses MIDI + MXL for each piece)
PYTHONPATH=midi2scoretransformer venv311/bin/python midi2scoretransformer/dataset.py

# Sanity check: dataset loader works and returns proper shapes
PYTHONPATH=midi2scoretransformer venv311/bin/python -c "
from dataset import ASAPDataset
ds = ASAPDataset('data/', 'train', seq_length=512, padding='per-beat')
print('train size:', len(ds))
x, y = ds[0]
print('input shapes:', {k: tuple(v.shape) for k, v in x.items()})
print('output shapes:', {k: tuple(v.shape) for k, v in y.items()})
"
cd ..
```

### Phase R6: Stage B fine-tune

```bash
# Find the best Stage A checkpoint
ls MIDI2ScoreTransformer/checkpoints/pretrain_pdmx*/

# Fine-tune on ASAP at 0.1× the Stage A LR (the plan calls this out)
# Note: train.py's PDMXDataset is what we wrote; for ASAP, you need to point
# at ASAPDataset instead. Quickest path: extend train.py's `fit()` with a
# --use-asap flag that swaps the dataset, OR write a thin wrapper script.
# Time budget: 30 min - 2 hours depending on GPU.

PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer venv311/bin/python \
  MIDI2ScoreTransformer/midi2scoretransformer/train.py fit \
  --stage finetune_asap \
  --manifest <ASAP-style manifest, see note below> \
  --lr 1e-5 \
  --max-epochs 5 \
  --batch-size 16 \
  --num-workers 8 \
  --precision bf16-mixed \
  --init-ckpt MIDI2ScoreTransformer/checkpoints/pretrain_pdmx-epoch=07-val/<best>.ckpt \
  --out-dir MIDI2ScoreTransformer/checkpoints
```

**Note for stage B**: the current `make_pdmx_loaders` in `train.py` only
loads from a flat manifest. For ASAP you'll need to either:

(a) Build a fake manifest by walking the ASAP file tree:
```python
# Pseudo-code; one row per aligned performance MIDI in ASAP
import os, csv, json
ann = json.load(open('MIDI2ScoreTransformer/data/asap-dataset/asap_annotations.json'))
rows = []
for k, v in ann.items():
    if not v['score_and_performance_aligned']: continue
    midi = f'MIDI2ScoreTransformer/data/asap-dataset/{k}'
    mxl = midi.rsplit('/', 1)[0] + '/xml_score.musicxml'
    chunks = midi.replace('.mid', '_chunks.json')
    rows.append({'id': k, 'midi': midi, 'mxl': mxl, 'chunks': chunks,
                 'cache': '', 'src_mxl': '', 'n_notes': 0,
                 'n_measures': 0, 'n_in_tokens': 0, 'n_out_tokens': 0})
# write to data/asap_manifest.csv
```

OR (b) extend `train.py` to accept a `--dataset-type asap` flag that constructs
`ASAPDataset` directly. Cleaner long-term.

### Phase R7: Final eval (Step 10)

```bash
# Run inference with each checkpoint on the 3 benchmark pieces
for piece in \
  "benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi" \
  "benchmark/chopin_op25/midi/Op25_No11_Aminor.midi" \
  "benchmark/liszt_transcendental/midi/Transcendental_No4_Mazeppa.midi"; do
    base=$(basename $piece .midi)
    for tag in baseline pretrain finetune; do
        case $tag in
            baseline) ckpt=MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt;;
            pretrain) ckpt=MIDI2ScoreTransformer/checkpoints/pretrain_pdmx-*/<best>.ckpt;;
            finetune) ckpt=MIDI2ScoreTransformer/checkpoints/finetune_asap-*/<best>.ckpt;;
        esac
        venv311/bin/python scripts/infer_with_ckpt.py "$piece" \
          --ckpt "$ckpt" --out "benchmark/post_eval/xml/${base}_${tag}.musicxml"
    done
done

# Compute structural metrics for each
venv311/bin/python benchmark/eval_baseline.py --tag baseline --xml-dir benchmark/post_eval/xml
venv311/bin/python benchmark/eval_baseline.py --tag pretrain --xml-dir benchmark/post_eval/xml
venv311/bin/python benchmark/eval_baseline.py --tag finetune --xml-dir benchmark/post_eval/xml

# Save the comparison table
python -c "
import json
results = {}
for tag in ['baseline', 'pretrain', 'finetune']:
    with open(f'benchmark/baseline_metrics_{tag}.json') as f:
        results[tag] = json.load(f)
with open('benchmark/post_finetune_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"
```

**Held-out pieces** (3 difficulty-matched non-ASAP / non-PDMX pieces) need
manual sourcing per [data/preflight/heldout_plan.md](data/preflight/heldout_plan.md).
Recommended candidates (verified absent from ASAP composer list):

| Piece | Source | Notes |
|---|---|---|
| Mendelssohn — Etude Op.104a No.3 | IMSLP MusicXML | Mendelssohn not in ASAP |
| Saint-Saëns — Etude Op.111 No.4 (Toccata) | IMSLP / MuseScore.com | Not in ASAP |
| Albeniz — Iberia, "Triana" | IMSLP | Not in ASAP composer list |

Before adding each held-out, grep against `data/pdmx_piano_subset.csv` to
confirm it's also not in PDMX (search by composer + piece title).

### Phase R8: Report results back

Append your results to this file (`SYNTHETIC_PRETRAIN_STATUS.md`) under a new
section:

```markdown
## GPU run results (date)

### Stage A pretrain (50K pairs, X epochs)
- Final val loss: ...
- LR used: ...
- Training time: ...
- Best ckpt: ...

### Stage B fine-tune (ASAP, X epochs)
- Final val loss: ...
- Training time: ...
- Best ckpt: ...

### Step 10 metrics table

[paste comparison table]

### Conclusions
...
```

## Files the person running the GPU build will need to know about

| Path | Purpose |
|---|---|
| `scripts/filter_pdmx.py` | Filter PDMX.csv to piano subset |
| `scripts/make_pairs.py` | Bulk synthetic pair generation (parallelized) |
| `scripts/expressive_render.py` | Score → perturbed MIDI with 1:1 alignment |
| `scripts/eyeball_check.py` | Sanity-check synthetic distributions |
| `scripts/infer_with_ckpt.py` | Single-ckpt inference helper |
| `MIDI2ScoreTransformer/midi2scoretransformer/train.py` | Lightning trainer + LR-range test |
| `MIDI2ScoreTransformer/midi2scoretransformer/pdmx_dataset.py` | Synthetic dataset wrapper |
| `benchmark/eval_baseline.py` | Structural metrics on benchmark pieces |
| `data/preflight/step1_overlap_findings.md` | What's in ASAP and what isn't |
| `data/preflight/heldout_plan.md` | Held-out picks and why |

## Hyperparameters from the released `MIDI2ScoreTF.ckpt`

For reference when reverse-engineering the training paradigm. These came from
inspecting `ckpt['hyper_parameters']`:

```
architecture: Roformer
batch_size: 32
learning_rate: 0.0003
betas: (0.9, 0.999)
weight_decay: 0.2
steps: 40000
warmup_steps: 4000
seq_length: 512
optimizer: AdamW
loss: ce
padding: per-beat
input_dropout: 0.75
context_dropout: 0.0
unconditional_dropout: 0.5
augmentations: {transpose: True, random_crop: True, tempo_jitter: [0.8, 1.2],
                onset_jitter: 0.05, random_shift: 8, velocity_jitter: 5}
gradient_clip_val: None
components: ['encoder', 'decoder']
domains: {in: 'midi', out: 'mxl'}
eval_interval: 100
full_eval_interval: 10000
dataset_weights: [0.5, 0.5]   # ← interesting; suggests 2 datasets weighted equally
```

Notes:
- `is_autoregressive: False` (in encoder + decoder configs) — yet the model
  generates fine. This is the mystery our trainer didn't solve.
- `dataset_weights: [0.5, 0.5]` — suggests they trained on 2 datasets jointly
  (probably ASAP-real + ASAP-synthetic-from-MIDI-score, since `bucket_midi`
  makes it easy to use the deadpan MIDI score as a synthetic input).
- The 40,000 total steps × batch 32 = 1.28M samples seen total. ASAP has
  ~800 train pieces; that's ~1600 samples/epoch with the per-beat chunk
  random_crop augmentation, so ~800 epochs equivalent. Lots of training.
- `input_dropout: 0.75` matches what we used. `context_dropout: 0.0`
  unused. `unconditional_dropout: 0.5` matches.

If the upstream paper or its supplementary materials describe their training
script anywhere, that's the highest-impact thing to find before doing more
experiments.

## Disk usage from this work

| Item | Size |
|---|---|
| `~/datasets/pdmx/PDMX.csv` | 215 MB |
| `~/datasets/pdmx/mxl/` (extracted) | ~5 GB |
| `~/datasets/pdmx/mxl.tar.gz` (compressed source) | 1.8 GB |
| `data/pairs/` (5K pairs + alignment + chunks + .mxl copies) | 489 MB |
| `data/cache_pdmx/` (tokenized .pkl) | 249 MB |
| `MIDI2ScoreTransformer/data/asap-dataset/` (full clone) | ~1 GB |
| `MIDI2ScoreTransformer/data/ACPAS-dataset/` | ~1 MB |
| `MIDI2ScoreTransformer/checkpoints/pretrain_pdmx*` | ~1.5 GB (4 epoch ckpts × 372 MB) |
| **Total in repo** | ~3.6 GB |
| **Total in `~/datasets/pdmx/`** | ~7 GB |
