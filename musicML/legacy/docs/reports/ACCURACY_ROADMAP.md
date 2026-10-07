# Accuracy Roadmap — Piano Audio → Sheet Music

A consolidated map of the whole project, an honest definition of "accuracy,"
the corrected diagnosis of the retrain blocker, and a prioritized plan to push
Block 2 (MIDI → score) toward the realistic ceiling.

> **Status:** analysis + plan only. No code has been changed. This document is
> the deliverable to review before any build work begins.

> **Checkpoint verification (2026-05-30).** Before committing to this plan, the
> facts it depends on were checked against the repo:
> - ✅ **The rigorous eval runs locally.** `muster`, `score_transformer`, and
>   music21 9.2.0b3 (Beyer's fork) all import in `venv311`; MUSTER's backend is
>   compiled **arm64** and a self-vs-self smoke test on an ASAP score returned
>   `PitchER 0.0, MissRate 0.0, OnsetER 0.0`. **No GPU or Linux box needed for
>   eval.** (Noise floor: even self-vs-self shows ~0.28 ExtraRate/VoiceER from the
>   score-cleaning step — don't claim sub-0.3 improvements as real.)
> - ✅ **235 ASAP ground-truth scores + ACPAS metadata are present locally.**
> - ⚠️ **2 of the 3 benchmark pieces *do* have GT scores** (Op.10 No.4, Op.25 No.11
>   are in ASAP); only **Mazeppa** lacks one (sourceable from PDMX). But those 2
>   are in ASAP's **train** split → in-distribution, not a clean generalization
>   test. (Corrects §5's earlier "no GT" claim.)
> - ✅ **Benchmark audio is present** (Op.10/Op.25/Mazeppa `.wav`) → end-to-end
>   audio→score eval is feasible, not just MIDI→score.

---

## 0. Bottom line up front

1. **The lineage is verifiable.** This repo's Block-2 model *is* the model of
   [arXiv:2410.00210](https://arxiv.org/abs/2410.00210) "End-to-end Piano
   Performance-MIDI to Score Conversion with Transformers" (Beyer & Dai, ISMIR
   2024), the same released model that the Songscription product is built on.

2. **Literal 100% accuracy is impossible for this task.** A single performance
   maps to *many* musically valid notations
   (enharmonic spelling, voice/staff split, trill vs. written-out, beaming, meter
   choice are all editorial). The paper's SOTA is MUSTER avg error **11.30**,
   which still means ~15–24% onset/offset disagreement. **The
   right target is "a clean draft a musician lightly edits" — i.e., beat the
   published SOTA — not 100%.**

3. **The retrain is not blocked by a mysterious bug, and you don't need to solve
   it.** The custom trainer learns an *identity copy*, not generation; at
   inference it collapses to `pad=0` everywhere → empty score. The **working
   released checkpoint is already in the repo**. Warm-start from it and fine-tune
   → the cold-start collapse disappears.

4. **You're steering with a broken speedometer.** The benchmark "F1" is a
   pitch-multiset that reads **0.97–0.98 even when the meter is wrong and the
   measure count is off by 93%**. The real field-standard eval (MUSTER +
   score-similarity over the ASAP held-out test set) **already exists in the repo,
   unused**. Wiring it up is step 1 and needs no GPU.

---

## 1. Lineage

This repo's Block-2 model is the transformer of [arXiv:2410.00210](https://arxiv.org/abs/2410.00210)
(ISMIR 2024), which is **MIDI → score only**; the `audio → hFT → MIDI2Score` pipeline here
puts an audio → MIDI front-end in front of it. The published bar to beat is **MUSTER avg 11.30** (§2).

---

## 2. What "accuracy" means here (and the honest ceiling)

The task is **one-to-many**: the same performance has many correct notations.
Note-level F1 hides this — which is *exactly why* the field uses **MUSTER**
(MUsic Score Transcription Error Rate; Nakamura et al., ICASSP 2018), an
edit-distance metric like speech WER, plus **score-similarity** for engraving.

**Two-tier metric stack (both already vendored in the repo):**

- **MUSTER** (transcription level): `PitchER`, `MissRate`, `ExtraRate`,
  `OnsetER`, `OffsetER`, `MeanER`, `VoiceER`, `HandER`.
- **score_similarity** (notation/engraving level): `TimeSignature`,
  `KeySignature`, `Clef`, `NoteSpelling`, `NoteDuration`, `StemDirection`,
  `Beams`, `Tie`, `StaffAssignment`, `Voice`, plus Grace/Staccato/Trill F1.

**The bar to beat (paper, lower = better):**

| Metric | Beyer transformer | Best classical baseline | MuseScore | Finale |
|---|---|---|---|---|
| Pitch (E_p) | 3.11 | ~2.0–2.4 | — | — |
| Missing | 7.56 | — | — | — |
| Extra | 6.44 | — | — | — |
| Onset | 15.55 | — | — | — |
| Offset | 23.84 | — | — | — |
| **Average** | **11.30** | 13.95 (HMM+heuristics) | 23.35 | 20.64 |

Notes: the transformer's *weakness* is pitch (3.11 vs ~2.0) because it rebuilds
the sequence from scratch instead of copying input pitches — many "missing note"
errors are actually it notating a trill where the reference wrote out the notes
(musically equivalent, penalized by the metric). Removing data augmentation
collapses E_avg from 11.30 → 35.30 ("augmentation is crucial").

**Measure accuracy at two levels — they are not the same number.** This is the
single most important framing correction:

- **Block-2-isolated** (MIDI → score): feed *performance MIDI* (or GT MIDI),
  score the predicted MusicXML vs GT MusicXML. This is what the paper's **11.30**
  measures, and the only number directly comparable to it. Use the **14-piece
  ASAP held-out test split** (`TEST_PIECE_IDS`) for the headline.
- **End-to-end** (audio → score): feed *audio*, score the final MusicXML. This is
  the real product quality and is **necessarily worse than 11.30** because Block 1
  adds transcription error on top. Report it separately; never claim the
  end-to-end pipeline "beats 11.30" — that's apples-to-oranges.

**Realistic targets, in two tiers:**

1. **Near-term (provable, no GPU):** *reproduce* the released checkpoint's real
   MUSTER on the held-out split (anchor it next to the paper's 11.30), and show
   **measurable before/after** from the cheap inference fixes. This alone is a
   credible, honest lab result.
2. **Research contribution (GPU):** *beat* 11.30 / close the out-of-distribution
   gap (Mazeppa-class) via data scaling. A 2025 follow-up using *learned*
   rendering already hit **E_avg 9.48** on the same ASAP task — proof the lever is
   data realism, not architecture. Treat this as the goal, not a guaranteed
   near-term win.

**Beware the leakage trap in the demo pieces:** Op.10 No.4 and Op.25 No.11 are in
ASAP's *training* split, so good scores on them show *memorization*, not
generalization. They are fine as a *narrative* ("the model handles repertoire it
has seen") but must **not** be the rigorous headline. Mazeppa (not in ASAP) is the
honest generalization test.

**Phase 1 is near-solved — don't touch it:** SOTA audio→MIDI is ~98% onset F1
(Yan "Scoring Intervals" 98.32%; hFT 97.44%; Kong/ByteDance 96.72%). Your hFT is
already excellent. All accuracy gains live in Block 2.

---

## 3. System map: where accuracy is won and lost

```
Audio ─►[Block 1: Audio→MIDI]─► perf-MIDI ─►[Block 2: MIDI→MusicXML]─► score ─►[Render]─► PDF
         hFT-Transformer                      MIDI2ScoreTransformer            MuseScore CLI
         ~97% onset F1                         ← THE BOTTLENECK →               (crashes on dense
         NEAR-SOLVED. Leave it.                ALL accuracy lives here.          scores; chunk-fallback
                                                                                 silently drops measures)
```

**Why Block 2 is data-bound (your own preflight proves it):**

| Piece | In ASAP? | Result |
|---|---|---|
| Chopin Op.10 No.4 | YES (22 perf) | 4/4 ✓, 6.8% measure error |
| Chopin Op.25 No.11 | YES (19 perf) | 4/4 ✓, 2.0% measure error |
| Liszt Mazeppa | **NO** | **3/4 ✗, 92.8% measure error** |

In-distribution works; out-of-distribution fails. Classic data-shortage
signature. The model was trained on ~200 pieces.

---

## 4. The retrain blocker — corrected diagnosis

Your docs call this a mysterious "from-zero generation" failure. It is a
**train/inference mismatch**, verified in code and against the paper.

**How inference works** ([generate()](MIDI2ScoreTransformer/midi2scoretransformer/models/model.py#L33)):
the decoder is bidirectional ([roformer.py:444-445](MIDI2ScoreTransformer/midi2scoretransformer/models/roformer.py#L444-L445)
zeroes the causal mask) but generation is **sequential** — it bootstraps from an
all-zero start token and feeds back its own predictions. Predicting the *first*
token from nothing is the hardest regime and needs the most training.

**The paper's bridge** (from ar5iv full text): heavy decoder token dropout
(`input_dropout=0.75` → expose only ~25% of preceding tokens, breaking the
trivial identity copy) **plus** `unconditional_dropout=0.5` used as
classifier-free conditioning dropout **on the 58,646 *unpaired* MuseScore scores**
(so the decoder learns a standalone score prior).

**What the custom trainer gets wrong:**

| # | Problem | Location | Effect |
|---|---|---|---|
| 1 | Decoder fed the **unshifted** target → learns identity, not next-token generation | [train.py:163-174](MIDI2ScoreTransformer/midi2scoretransformer/train.py#L163) + [model.py:17-30](MIDI2ScoreTransformer/midi2scoretransformer/models/model.py#L17) | Converged loss does not predict inference behavior |
| 2 | `_maybe_drop_decoder` never drops the **pad** stream and never practices the all-zero start | [train.py:87-101](MIDI2ScoreTransformer/midi2scoretransformer/train.py#L87) | Pad head collapses to 0 → empty score |
| 3 | Pad BCE has no `pos_weight`; per-beat padding makes `pad=0` the "safe" answer | [train.py:114-119](MIDI2ScoreTransformer/midi2scoretransformer/train.py#L114) | `pad` logits ≈ 0 everywhere |
| 4 | `unconditional_dropout` zeroes the **encoder MIDI** on *paired-only* data, no unpaired branch | [train.py:77-80](MIDI2ScoreTransformer/midi2scoretransformer/train.py#L77) | Model learns "ignore MIDI, output nothing" |
| 5 | Validation feeds the **full GT** decoder input (no dropout) → measures denoising, not generation | [train.py:178](MIDI2ScoreTransformer/midi2scoretransformer/train.py#L178) | `val/total=0.244` looks great while the model can't generate |
| 6 | Undertrained: ~20K samples seen vs the paper's 40k steps × batch 32 ≈ **1.28M** | — | From-zero regime never learned |

**The unblock (priority order):**

1. **Warm-start from `MIDI2ScoreTF.ckpt` and fine-tune. ✅ VALIDATED on Mac
   (2026-05-30, "Calibration A", [scripts/calibration_a.py](scripts/calibration_a.py)).**
   Warm-started from the released checkpoint, fine-tuned 250 steps on the synthetic
   pairs with our trainer — generation is **preserved** (non-pad 724→725/960,
   MeanER 5.78→5.52). The collapse was a *cold-start* problem, not a warm-start one.
   Both `unconditional_dropout=0.5` (as-is) and `0.0` keep generating; as-is held
   MeanER flat, behaving as a regularizer here, not a harm. **Track C is de-risked
   before any GPU spend.** Fastest path to a generating checkpoint *and* real gains.
2. If a from-scratch trainer is also wanted: add the teacher-forcing shift, drop
   the pad stream too, add `pos_weight` to the pad BCE, apply
   `unconditional_dropout` only to unpaired scores (or lower to ~0.1), add a
   **generate-based val hook** logging non-pad note count, and train to the
   paper's step budget.

---

## 5. The accuracy pipeline (build FIRST — no GPU, deps already installed)

This is the foundation. Until it exists, no training run is measurable. **Good
news from the checkpoint verification: the hard part is already done** — the eval
packages are installed and MUSTER runs on this Mac. This is wiring, not building.

**Build two eval tiers** (per §2 — they answer different questions):

- **Tier 1 — Block-2-isolated (MIDI → score), the headline.** Run the model on the
  **14-piece ASAP held-out test split**
  ([constants.py TEST_PIECE_IDS](MIDI2ScoreTransformer/midi2scoretransformer/constants.py))
  and score predicted MusicXML vs the GT `xml_score.musicxml` with **MUSTER +
  score_similarity**. [run_eval.py](MIDI2ScoreTransformer/midi2scoretransformer/evaluation/run_eval.py)
  already does almost exactly this — it is just never invoked by `benchmark/`.
  This is the number comparable to the paper's **11.30**.
- **Tier 2 — End-to-end (audio → score), the product number.** Run the full
  `audio → hFT → MIDI2Score → MusicXML` pipeline on pieces that have **both audio
  and a GT score**, and score the final MusicXML. Sources: the 3 benchmark pieces
  (audio present locally; GT for the 2 Chopin etudes is in ASAP, Mazeppa's is in
  PDMX), and/or **ACPAS** (MAESTRO audio aligned to scores). Without this tier, the
  Track-B audio-side improvements (hFT thresholds, tempo) are **unmeasurable**.

**Then:**

1. **Anchor to the released checkpoint. ✅ DONE (2026-05-30).** The released
   `MIDI2ScoreTF.ckpt` scores **MeanER 11.18** on the 59-performance held-out split
   — reproducing the paper's **11.30** to within noise (PitchER 3.19/3.11,
   StaffAssign 6.71/6.62, StemDir 25.04/25.03). The eval harness is validated and
   the anchor is set. Full results + the per-composer difficulty gradient:
   [benchmark/BASELINE_RESULTS.md](benchmark/BASELINE_RESULTS.md). (Eval noise
   floor ~0.28; sub-noise "gains" aren't real.)
2. **Kill the confounds:**
   - **Leakage (audited + DEDUPED ✅ 2026-05-30):** 12/17 eval pieces leaked into
     PDMX. The content-fingerprint dedup ([scripts/content_dedup.py](scripts/content_dedup.py))
     is built, validated (known leak 0.961 vs unrelated 0.001), and run: **23 leak
     rows removed** → clean training pool
     [data/pdmx_piano_subset.deduped.csv](data/pdmx_piano_subset.deduped.csv)
     (181,693 → 181,670). Caught alt-title (Gondoliera→"Venezia e Napoli") and
     embedded copies metadata missed. Full details: [benchmark/LEAKAGE_AUDIT.md](benchmark/LEAKAGE_AUDIT.md).
     Re-run on full PDMX if training expands beyond the piano subset.
   - **Fake F1:** retire the pitch-multiset F1
     ([eval_improvements.py:104-124](benchmark/eval_improvements.py#L104) ignores
     onsets despite the docstring) as a steering metric.
   - **Demo-piece honesty:** Op.10 No.4 and Op.25 No.11 are in ASAP's *train*
     split — report them as narrative, not as the rigorous headline (§2 leakage
     trap). Mazeppa is the clean generalization case; **source its GT score from
     PDMX and permanently hold it out of any training set.**
3. **Training-time structural validation.** Add a callback that decodes a val
   subset to MusicXML and logs `TimeSignature` / `NoteDuration` / `StaffAssignment`
   error rates — so checkpoints are selected on notation quality, not CE loss.

**Deliverable:** one `eval` command that prints MUSTER + score_similarity (both
tiers) for any checkpoint, with the released baseline and the paper's numbers
alongside. This is the "before/after" engine for reporting results.

---

## 6. Prioritized roadmap (the agreed sequence)

**Scoping note.** **Track A + Track B = a complete, honest, GPU-free deliverable**
(rigorous eval + real baseline + measurable before/after + the diagnosis narrative)
that can be done on a Mac. **Track C + D = the GPU research phase**, where we
actually try to beat 11.30 / fix the OOD gap. Don't let the GPU phase block the
measurable result.

### Track 0 — This document (done)
Review, then green-light the next track.

### Track A — Accuracy pipeline / eval (no GPU, ~days)
- Wire **Tier 1** (MIDI→score, MUSTER + score_similarity over the 14-piece ASAP
  held-out split) **and Tier 2** (end-to-end audio→score on benchmark/ACPAS).
- First action: record the **released checkpoint's real baseline** (the anchor).
- Hash-dedup PDMX vs ASAP + benchmark pieces; report overlap. Source + hold out
  Mazeppa's GT score.
- Replace the fake F1; add a structural val callback.
- **Output:** trustworthy two-tier numbers + a reproducible before/after harness.

### Track B — Cheap inference wins ✅ DONE (2026-05-30) — measured NULL
Full results: [benchmark/TRACKB_RESULTS.md](benchmark/TRACKB_RESULTS.md).
- **Implemented** (opt-in flags in [transcribe.py](transcribe.py), defaults
  unchanged): `--hft-onset-threshold` / `--hft-mpe-threshold` /
  `--hft-offset-threshold`, `--hft-offset-mode`, and `--no-render` (eval speed).
- **Skipped tempo recovery** — it inserts seconds-as-tempo scaffolding with no
  BPM-conversion downstream (risks garbage marks) and can't move MUSTER.
- **Deprioritized meter generalization** — Mazeppa's meter is fundamentally
  mispredicted (57% vote 3/4); only helps 6/8 repertoire absent from the benchmark.
- **Result:** the hFT-threshold sweep on Mazeppa moved MeanER 34.9→34.1 — **within
  the GT noise floor (~2.1) and smaller than run-to-run variance (~0.7)**; lowering
  thresholds made PitchER *worse*. The 38% MissRate is not recall-limited.
- **Verdict:** a third independent confirmation (with A1–A5 + the analysis) that the
  OOD failure is **data-bound, not knob-tunable**. The no-GPU levers are exhausted;
  accuracy now requires Track C/D. The flags remain for per-recording tuning.

### Track C — Retrain unblock + targeted fine-tune (GPU, ~days)
> **✅ RUNBOOK PREPPED (2026-05-31) — [GPU_RUNBOOK.md](GPU_RUNBOOK.md) +
> [scripts/gpu_finetune.sh](scripts/gpu_finetune.sh).** One-command, resumable:
> env-check → generate pairs from the deduped pool → warm-start fine-tune → eval
> every ckpt by real MUSTER → report (best vs baseline, hard-composer subset).
> Every phase validated end-to-end on the Mac (fit() completes + saves; the
> fine-tuned ckpt loads + generates). `MODE=smoke` de-risks in ~20 min on the GPU
> before the full run. Just provision a GPU and launch.

- Warm-start from `MIDI2ScoreTF.ckpt`; confirm it still generates with the trainer
  (the doc's "Calibration A"). Add the generate-based val hook.
- Fine-tune on real ASAP + a small curated set of **hard cases** (Chopin/Liszt/
  Rachmaninoff).
- **Honest expectation:** this is a *hypothesis to test*, not a guaranteed win.
  Mazeppa specifically may **not** be fixable by a small fine-tune: the preflight
  runs showed its meter predictions are fundamentally wrong (57% predict 3/4), and tiny
  fine-tunes risk overfitting/forgetting. The provable near-term win is Track A+B;
  closing the OOD gap is the *research contribution* and probably needs Track D.
- **Output:** measured (by Track A) movement on OOD pieces — report it honestly,
  whatever it is.

### Track D — Data scaling (GPU-weeks — the real accuracy lever)
**Match the paper's actual recipe: joint training on paired + unpaired data, not
a staged synthetic pretrain.** The released model trained ASAP-paired **and**
~58K *unpaired* MuseScore scores *jointly* (`dataset_weights=[0.5,0.5]`), using the
unconditional branch to learn a score-language prior. Your synthetic 1:1 PDMX
pairs partly reinvent this in a weaker form. Better design:
- **PDMX as the unpaired score-prior corpus** (its highest-value role — it *is*
  the paper's 58K-unpaired analog, at 181K). Feed it through the unconditional
  branch instead of (or alongside) manufacturing 1:1 pairs.
- **Upgrade synthetic pairs with learned rendering** (VirtuosoNet, pretrained
  `dasaem/virtuosonet`) and **break the 1:1 alignment** (real transcription has
  extra/missing/reordered notes — the difficulty the model exists to solve). The
  2025 "Disentangling Score Content and Performance Style" paper hit **E_avg 9.48
  vs 11.30** doing exactly this.
- **Add real aligned pairs** via **PianoCoRe (2026, 157K aligned pairs)** or
  **ACPAS** (497 scores / 2189 MAESTRO performances) using Parangonar/Nakamura
  alignment — far better ROI than hand-aligning MAESTRO.
- **Curate toward the target distribution, don't just scale.** "24K classical" is
  ~93% Scottish-fiddle + hymns (only ~1,600 canonical). Oversample canonical
  classical for the paired/fine-tune signal; use the broad set for the prior.
- Optional Stage E: self-training pseudo-labels from MAESTRO/Aria-MIDI, *after* a
  non-collapsing checkpoint exists.

---

## 7. Data / retraining plan

Designed to mirror the paper's **joint** recipe (paired + unpaired together),
warm-started from the released checkpoint:

| Role | Data | Size | Licensing | Effort |
|---|---|---|---|---|
| **Unpaired score-prior** (unconditional branch) | PDMX scores | up to 181K | PD/CC0 — use the **222,856 no-conflict** subset; safe for any use | GPU, runs jointly |
| **Paired (real)** | ASAP (967) + ACPAS / PianoCoRe aligned | 1K–10K+ | ASAP/PDMX safe; check PianoCoRe per-source | ~days |
| **Paired (synthetic)** | PDMX **learned-rendered**, non-1:1 | 50–150K | PD/CC0 | GPU render pass + train |
| Optional | Self-training pseudo-labels (MAESTRO/Aria-MIDI) | large | **Aria-MIDI is CC-BY-NC-SA → non-commercial; quarantine** | only after non-collapsing ckpt |

**Datasets worth knowing:**
- **PDMX** (UCSD, NeurIPS 2024) — ~254K public-domain MuseScore XML; the only
  large *cleanly licensed* score source. Your filter already yields 181,693 piano.
- **ASAP / ACPAS** — the real (perf-MIDI, score) pairs; small (~500 scores) but
  high-quality and domain-matched.
- **PianoCoRe (2026)** — 157,207 performances aligned to 1,591 scores via
  Parangonar DualDTW, ~92% recall; potential near drop-in for real pairs.
- **VirtuosoNet** — pretrained MusicXML → expressive-MIDI renderer (HuggingFace
  `dasaem/virtuosonet`); far more human-like than coded jitter.

**Rough budget to first credible, measurable gain over SOTA:** ~4–8 GPU-weeks,
dominated by the render pass + Stage A/B. A single RTX 4090 (~$0.69/hr) suffices.

---

## 8. Traps found in the current code/data (fix before claiming any gain)

| Trap | Where | Why it matters |
|---|---|---|
| **Test-set leakage (QUANTIFIED, HIGH)** | **12 of 17 eval pieces** (3/3 benchmark + 9/14 ASAP test) are content-verified copies in PDMX — see [benchmark/LEAKAGE_AUDIT.md](benchmark/LEAKAGE_AUDIT.md). Only Mozart K.332, Haydn XVI:31, Schumann Arabeske, Scriabin Op.8/11, Prokofiev Toccata are clean | Any PDMX-trained "gain" on these is confounded. Metadata blocklists are INSUFFICIENT (alt-titles, embedded movements, mislabeled rows); a **content-fingerprint** dedup (transposition-invariant interval 5-grams, drop >0.4 containment) is required before any PDMX training |
| **"24K classical" is a myth** | PDMX genre tag | Only ~1,500 are canonical Western classical; the rest are mostly gospel hymns + Scottish fiddle |
| **Degenerate `hand` label** | 95% of single-track PDMX → `hand=2` for all notes | Mis-trains the staff/hand split (the model's *strongest* feature) |
| **1:1 synthetic alignment** | `expressive_render.py` emits notes in canonical order | Removes the alignment difficulty the model exists to solve; no extra/missing/reordered notes |
| **Fake F1 metric** | `eval_improvements.py:104-124` ignores onsets | 0.97–0.98 regardless of structural correctness |
| **Curriculum mismatch** | sequential pretrain→finetune | Released model trained *jointly* (`dataset_weights=[0.5,0.5]`), not staged |
| **MuseScore chunk-fallback drops measures** | `run_musescore_chunked` | Output can silently miss whole spans (Mazeppa loses ~7 chunks) |
| **Tempo never recovered** | `detokenize_mxl(midi_sequence=None)` in prod | Output has no performance tempo; the code exists but is dead |

---

## 9. Open decisions

These are genuine forks (the rest now have a recommendation baked in above):

1. **Renderer (Track D):** keep hand-coded jitter, switch to learned (VirtuosoNet),
   or ensemble both? *Recommendation: ensemble — learned for realism, jitter for
   adversarial coverage.* Bigger accuracy lever but adds an integration.
2. **Real-pair source (Track D):** ingest PianoCoRe / ACPAS directly vs align
   MAESTRO ourselves with Parangonar. *Recommendation: ingest first (days), DIY
   align only if yield is short (weeks).*
3. **Commercial vs research:** if commercial, training data must stay PD/CC0 +
   permissive (no Aria-MIDI, no ATEPP/GiantMIDI transcriptions of copyrighted
   recordings), which shrinks usable real data and makes synthetic PDMX the
   backbone. *This is a goal-level call; defaulting to research-grade unless told.*

**Resolved at the checkpoint (no longer open):**

- ~~Dataset scope (181K vs 24K vs canonical)~~ → **Recommendation set:** broad
  PDMX as the *unpaired prior*; curated + oversampled canonical-classical for the
  paired/fine-tune signal. Not a single-axis "how much" question.
- ~~MUSTER on macOS~~ → **Verified working locally** (arm64 binaries, smoke test
  passed). No Linux box needed.

---

## 10. References

- Paper: [arXiv:2410.00210](https://arxiv.org/abs/2410.00210) ·
  [full text (ar5iv)](https://ar5iv.labs.arxiv.org/html/2410.00210) ·
  [upstream repo](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer)
- MUSTER: Nakamura et al., ICASSP 2018 (vendored as `muster` in the repo)
- Data: PDMX (UCSD, NeurIPS 2024) · ASAP/ACPAS (CPJKU) · PianoCoRe (2026) ·
  VirtuosoNet (`dasaem/virtuosonet`) · "Disentangling Score Content and
  Performance Style" (2025, E_avg 9.48)
- Phase-1 SOTA: Yan "Scoring Intervals" (98.32% onset F1) · hFT-Transformer
  (97.44%) · Kong/ByteDance high-resolution (96.72%)

---

*Compiled from a deep analysis of the repo (code internals, the upstream paper,
SOTA literature, and the eval methodology). Review this, then green-light
Track A (the accuracy pipeline).*
