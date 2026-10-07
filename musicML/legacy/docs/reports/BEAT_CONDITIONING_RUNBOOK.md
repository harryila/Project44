# Beat-Conditioning Experiment — GPU Runbook

The intervention aimed at the **proven** bottleneck (the model fails on tuplet/meter
because it must infer the metric grid and gets it wrong). We add a per-note
**phase-within-beat** input so the model is GIVEN the grid. Externally validated:
Wachter/Klangio (arXiv:2604.22290, 2026) did exactly this to a Beyer baseline and cut
onset error to MUSTER e_onset 12.30 (vs Beyer's 15.55) + gained unseen-meter
generalization.

## Status: IMPLEMENTED + VALIDATED ON MAC (2026-06-01)
- ✅ `beat_features.py` — per-note phase → 12 sub-beat ticks + no-beat bucket (the
  triplet-resolving grid). Validated: Mazeppa phase histogram peaks at downbeat AND
  triplet positions (0.33/0.67).
- ✅ `config.py` — `in_beat_vocab_size=13`, `use_beat_conditioning=False` (opt-in).
- ✅ `models/embedding.py` — opt-in `beat` Linear, summed into the input embedding.
- ✅ `tokenizer.py` — `bucket_midi` emits the `beat` stream (real phase when beats
  given, all-no-beat otherwise).
- ✅ `dataset.py` — threads ASAP ground-truth `performance_beats` per note.
- ✅ `train.py` — `load_pretrained_init(use_beat_conditioning=True)` builds + **zero-inits**
  the beat Linear (warm-start byte-identical to baseline: MUSTER 1.64 = 1.64, verified);
  `--use-beat-conditioning` CLI flag wired through `fit`/`make_asap_loaders`.
- ✅ Production eval unaffected (released ckpt loads clean; `use_beat_conditioning`
  defaults False → no beat Linear built).

## Launch (on a GPU box, from repo root)
```bash
# Prereqs: venv311 + CUDA torch; released ckpt at
#   MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt
# ASAP dataset + _chunks.json present (run chunker.py if not — see GPU_RUNBOOK.md).
# The ASAP parse-cache helps (build_asap_cache.py); beat features add ~ms/note.

export PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer
export CUDA_VISIBLE_DEVICES=0   # single GPU; ASAP loader is CPU-IO-bound

venv311/bin/python MIDI2ScoreTransformer/midi2scoretransformer/train.py fit \
  --stage beat_asap --dataset-type asap --data-dir ./MIDI2ScoreTransformer/data/ \
  --use-beat-conditioning \
  --init-ckpt MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt \
  --lr 3e-4 --max-epochs 12 --batch-size 16 --seq-length 512 \
  --num-workers 8 --precision bf16-mixed \
  --out-dir MIDI2ScoreTransformer/checkpoints/beat_asap
```
Notes: lr 3e-4 (the released recipe's peak — we're effectively continuing training
with a new input). Warm-start means it starts == baseline and learns the beat signal.
Train on ASAP-only first (cleanest test of the beat lever); a mixed/PDMX run can follow.

## Evaluate (the decisive comparison)
```bash
# Beat eval needs beats at inference. For the held-out test pieces, ASAP GT beats give
# the CEILING; a tracker (PM2S) gives the deployable number. First measure the ceiling:
venv311/bin/python benchmark/eval_tier1_asap.py \
  --ckpt MIDI2ScoreTransformer/checkpoints/beat_asap/<best>.ckpt \
  --device cuda --out benchmark/beat_asap_eval.json
# NOTE: eval_tier1_asap.py currently tokenizes WITHOUT beats (all-no-beat) — see
# "Inference beats" below; for the ceiling test, thread ASAP GT beats into the eval's
# tokenize_midi call (small edit: pass the piece's performance_beats).
```

## Success criteria
- **Warm-start safety:** non-tuplet pieces (Bach/Mozart/Chopin) must NOT regress
  (the zero-init guarantees step-0 identity; confirm post-train they hold ~baseline).
- **The win:** on tuplet/multi-meter pieces (Mazeppa-class), MUSTER e_onset and
  measure-count error drop materially vs the ~14 floor; predicted time-signatures
  become plausible (not the current 3/4-×14 chaos).
- **Honest ceiling:** the gold-beat eval is the upper bound; the deployable number is
  capped by the inference beat-tracker (PM2S ~80% beat F1, ~14-28% downbeat F1 on dense
  rubato). Report both.

## Inference beats (for deployment, after the ceiling test wins)
The model needs per-note beat phase at inference. Options, best first:
1. **PM2S** (cheriell/PM2S) — performance-MIDI beat/downbeat tracker, pretrained, CPU.
   `pip install`; run `RNNJointBeatProcessor().process(midi)` → beats → `phase_features`.
2. **partitura** (1.9, installable) — beat estimation from MIDI.
3. **pretty_midi.get_beats()** — crude fallback (over-segments ~30%); already in
   `beat_features.beats_from_pretty_midi`.
Thread the tracked beats into `tokenize_midi`/`parse_midi` via a `beats=` arg (small
add) so production inference fills the `beat_phase` stream.

## Expected payoff (honest)
~21% relative onset-error reduction is the literature precedent (Wachter). On our
benchmark: Mazeppa-class MeanER should drop meaningfully below ~14 with gold beats; a
partial-but-positive gain with a real tracker. This is the ONLY lever aimed at the
proven cause, implemented and warm-start-safe, ready to launch.
