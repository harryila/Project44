> Historical report (2026-06-09). Later results supersede some conclusions here; see the README status section.

# Piano Audio → Engraved Score: Complete Lab Report

**Author:** Harry I. (Bayen Lab) · **Date:** 2026-06-09
**Scope:** the full `musicML` repository — every working and non-working component, the design choices behind them, the optimizations we made, and the core research finding.

This document is meant to stand alone as a technical reference. It is written in plain English first, with technical detail where it matters.

---

## 0. TL;DR (read this if nothing else)

- **What this project is:** turn a piano *performance* (audio or MIDI) into a correct *engraved score* (MusicXML/sheet music), as accurately as possible. Metric = **MUSTER MeanER** (Musical Score Transcription Error Rate; lower is better).
- **The pipeline works and is fully reproducible.** It has two blocks: **(1) audio→MIDI** (transcription) and **(2) MIDI→score** (notation). On clean, in-distribution piano it produces excellent scores — **end-to-end MeanER ≈ 2.75 on clean Chopin**. The Block-2 score model we run is the *same open released checkpoint the Songscription product is built on*. (The apples-to-apples comparison is the **11.18 full-test reproduction** below; 2.75 is the easy end of the difficulty range, not a like-for-like comparison.)
- **We reproduced the published state-of-the-art** (MIDI2ScoreTransformer, Beyer & Dai, ISMIR 2024) to within noise — **MeanER 11.18 vs the paper's 11.30** — which validates our whole evaluation harness.
- **We found, and rigorously characterized, exactly where the accuracy ceiling is:** it is **not** the transcriber, and **not** pitch — it is **rhythmic notation on dense/Romantic music**, specifically **triplet/tuplet placement** (putting notes at the correct "off-the-grid" onsets).
- **The headline research result:** across a broad experimental matrix, **no lever we tried achieved correct-*rate* triplet onset placement** — not data reshaping, self-training, a new tuplet-aware representation, loss-weighting, nor warm-start/from-scratch training. (Reshaping the corpus *over*-shot to **19.9%** placement but regressed MUSTER; everything else under-shot or collapsed near 0%.) We matched the released model's **validation loss** and came **within ~1 MUSTER point**, but never reproduced its correct-rate placement. **The leading, best-supported explanation is the training *data*: our corpus is ~1.7% tuplets — far poorer in the exact signal that's missing — whereas the released model's corpus/schedule are unpublished. We did *not* prove this is impossible: the decisive experiment (a tuplet-aware representation trained *to convergence* on tuplet-rich paired data) was left undertrained — so this is "not recovered within our compute/data budget," not a proof of an irreproducible secret recipe.**
- **Practical implication:** the released model is the *placement-capable engine*; the right strategy is to **use it as a data engine** and to own the **music-notation representation** problem.

---

## 1. The Problem and Where It Sits

### 1.1 The concrete task
Given a piano performance, produce a human-readable **engraved score**. A performance is "what was played" (note onsets in real, expressive time, with rubato). A score is "what a musician reads" — quantized rhythm, beams, tuplets, time signatures, voices, hand assignment, stems, accidentals. Going from one to the other is a hard, *structured* prediction problem: the model must infer the underlying metrical grid and notational intent from messy human timing.

### 1.2 The lab's bigger goal
This pipeline (audio/performance → score) is one piece of a broader lab effort on music notation.

### 1.3 The metric: MUSTER / MeanER
**MUSTER** (Musical Score Transcription Error Rate) compares a predicted MusicXML to a ground-truth engraved score. It is a Python wrapper (`muster/muster.py`) around a C++ binary that parses both scores and computes six error rates (percent, lower=better):
- **PitchER** (wrong pitches), **MissRate** (notes missing), **ExtraRate** (extra notes), **OnsetER** (onset timing), **OffsetER** (duration/offset timing), and **MeanER** = the mean of these five (the headline number).
- It also reports engraving sub-metrics (note spelling, voice assignment, stem direction, staff assignment) and voice precision/recall/F1.
- **Caveat we hit:** the C++ binary **aborts on 3-staff scores**, so out-of-distribution evaluation is limited to 2-staff pieces.

---

## 2. System Architecture (the pipeline at a glance)

```
 AUDIO (.wav/.mp3/…)
     │   ── BLOCK 1: transcription (audio → MIDI) ──
     ▼
 Performance MIDI  ──(optional MIDI cleanup/filter)──►
     │   ── BLOCK 2: notation (MIDI → engraved score) ──
     ▼
 MusicXML score ──► (MuseScore/LilyPond render) ──► PDF / PNG
```

- **Block 1** is a *solved-enough* problem: off-the-shelf neural transcribers recover ~95–97% of notes even on dense music.
- **Block 2** is the *hard, interesting* problem and where the entire accuracy ceiling lives.
- **Entry point:** `transcribe.py` (CLI: choose `--transcriber` and `--backend`).

A key early result that shaped everything: **we decomposed the pipeline and proved Block 1 is not the bottleneck.** Feeding *ground-truth* MIDI directly into Block 2 gives essentially the same error as the full audio→score path (33.8 vs 34.2 MeanER on the hardest piece). So all research effort went into Block 2.

---

## 3. Block 1 — Audio → MIDI (Transcription)

**Status: ✅ fully working. Not the bottleneck. We deliberately stopped optimizing it.**

### 3.1 The four transcribers (all wired into `transcribe.py`)
| Transcriber | What it is | Status | Notes |
|---|---|---|---|
| **Basic Pitch** (Spotify CNN) | Frame-level CNN onset/frame detector | ✅ default | Zero setup (pip), good on clean audio. Chosen as default for lowest friction. |
| **hFT-Transformer** (Sony, ISMIR 2023) | Hierarchical transformer, sub-frame (~1–2 ms) precision | ✅ **best** | 96.72% onset-F1 on MAESTRO; **92% note recovery on dense Liszt Mazeppa**. Recommended for production. Checkpoint present (`hFT-Transformer/checkpoint/MAESTRO-V3/model_016_003.pkl`). |
| **MT3** (Google Magenta, ISMIR 2021) | T5 seq2seq, 10 ms grid | ✅ (separate venv) | Coarser timing; JAX deps require `venv_mt3`. |
| **Transkun** (event-based) | Continuous-time event model | ✅ | Fully continuous timestamps; less evaluated. |

### 3.2 Design choices & optimizations
- **Default = Basic Pitch** (no checkpoint needed); **recommended = hFT** (best accuracy + sub-frame precision). The `--transcriber` flag picks.
- **`--normalize-audio`** (`preprocess_audio()`): mono downmix + 16 kHz resample + peak-normalize to −3 dBFS. **hFT-only** (it's level-sensitive). Optimization with a caveat we documented: *don't enable it on already-hot recordings.*
- **MIDI cleanup** (`filter_midi()`): drops notes <50 ms or velocity <10, clips to the 88-key range, debounces overlapping same-pitch notes. Removes transcription noise before Block 2.
- **Threshold knobs** for hFT (`--hft-onset-threshold`, `--hft-mpe-threshold`): lowering to ~0.3 rescues soft/dense notes — the single biggest recall lever in Block 1.

### 3.3 Why we stopped here (a decision, not laziness)
We ran a threshold sweep (`TRACKB_RESULTS.md`) and a stage decomposition (`DECOMPOSED_FINDINGS.md`). Both showed transcription is ~95%+ correct and that the failures on hard pieces are **100% in Block 2**. We explicitly decided **not** to chase a better transcriber or ensemble — that would have been optimizing the wrong block.

---

## 4. Block 2 — MIDI → Engraved Score (the Beyer model)

**Status: ✅ working (released checkpoint), exhaustively studied. This is where the science is.**

### 4.1 What it is
The **MIDI2ScoreTransformer** (Beyer & Dai, ISMIR 2024) — an **encoder–decoder RoFormer** (rotary-position transformer). We verified the architecture is **~32.6M parameters**, 4 encoder + 4 decoder layers, hidden size 512, 8 heads, SwiGLU feed-forward, RoPE positions. The upstream repo ships **inference code and a checkpoint only — no training code** (we wrote all the training; see §6).

### 4.2 The compound-token representation (the crux of everything)
Instead of one token per note, each note is represented by **13 parallel attribute streams** plus a gating stream:

`offset` (within-measure position), `downbeat` (measure boundaries), `duration`, `pitch`, `accidental`, `keysignature`, `velocity`, `grace`, `trill`, `staccato`, `voice`, `stem`, `hand`, and **`pad`** (binary: is this slot a real note?).

- **Input (MIDI) side:** 5 streams — onset Δ, duration, pitch, velocity, (+ optional beat).
- **Rhythm grid:** time is quantized to a **1/24-of-a-quarter** grid. This is the single most important design fact: **dyadic** values (quarter=24, eighth=12, 16th=6) land on **multiples of 3**; **triplets** land *off* them (triplet-eighth=8, triplet-quarter=16, triplet-16th=4). So "is this a triplet?" becomes "is this onset on a non-multiple-of-3 bucket?"
- **Key tokenizer functions:** `parse_mxl` (MusicXML → raw per-note attributes), `bucket_mxl` (raw → one-hot buckets), `detokenize_mxl` (token streams → reconstructed MusicXML). The round-trip is tested to be faithful.

### 4.3 How it generates
The released model is **non-autoregressive (bidirectional "mask-predict")**: the decoder attends to all positions (no causal mask) and refines the whole sequence. The **`pad` stream** gates which slots become real notes (sigmoid → keep/drop). We later also built a **causal autoregressive** variant for from-scratch training (the original non-AR recipe is unpublished).

### 4.4 Why this representation is elegant — and where it bites
- **Elegant:** parallel streams allow per-attribute loss weights and *constraint satisfaction* at decode time (e.g., zero out impossible accidentals for a given pitch; force a downbeat when the offset resets).
- **The bite:** rhythm is split across `offset`/`duration`/`downbeat` on an **absolute** grid where triplet buckets are individually **rare**. This is the root of the tuplet problem (§7).

---

## 5. The Evaluation Harness

**Status: ✅ working and validated against the published paper.**

### 5.1 The test set
**14 pieces (one per composer) from the ASAP held-out test split**, 59 performances total — Bach, Beethoven, Brahms, Chopin, Debussy, Haydn, Liszt, Mozart, Prokofiev, Rachmaninoff, Ravel, Schubert, Schumann, Scriabin. This spans an honest difficulty gradient from clean classical (Bach ~5.8) to dense Romantic (Ravel ~21.5).

### 5.2 The eval scripts (what each does)
- **`eval_tier1_asap.py`** — the paper-comparable headline eval (MIDI→score on all 59 perfs). *Must run on CPU on macOS* (Apple-MPS produces degenerate pad logits — a real gotcha we found).
- **`eval_padsweep.py`** — our **optimization**: infer once per piece, then score *all* pad thresholds in a single pass (~8× cheaper). This became the fast "standing" measurement during the placement experiments.
- **`eval_decomposed.py`** — splits audio→score into Stage A (transcription, scored with onset-F1) and Stage B (notation, scored with MUSTER) to localize failures.
- **`legacy/benchmark/eval_baseline.py` / `eval_improvements.py`** (archived) — the early A1–A5 inference-tweak comparisons on 4 benchmark pieces.
- **`scripts/rerank_by_muster.sh`** — re-ranks every saved training epoch by **true MUSTER**, not validation loss (we proved repeatedly that **val loss ≠ MUSTER**, so picking checkpoints by val loss is wrong).

### 5.3 The reproduction that validates everything
| Metric | Ours | Paper (Beyer & Dai 2024) |
|---|---|---|
| **MeanER** | **11.18** | 11.30 |
| PitchER | 3.19 | 3.11 |
| MissRate | 7.83 | 7.56 |
| OnsetER | 14.50 | 15.55 |
| OffsetER | 23.73 | 23.84 |

All deltas <0.12 → **we reproduced SOTA to within noise.** This is the anchor: every later change is measured against a real, reproduced baseline, not a paper number.

> **Two measurement contexts (don't conflate them):** the rigorous paper-comparable number is **11.18 on 59 ASAP performances (CPU)**. During the fast iteration on the GPU box we used a **14-piece pad-sweep** subset where the released model scores **≈10.77–10.84**; our best *trained-from-scratch* model scores **11.79**, and our best *inference-tuned* result is **≈11.41**. Both are honest; they're just different slices.

### 5.4 The leakage audit (a piece of rigor worth presenting)
`LEAKAGE_AUDIT.md`: using transposition-invariant content fingerprints (not just titles), we found **12 of 17 evaluation pieces also exist in PDMX** (the large MuseScore corpus), some hidden under alternate titles. **This does not affect the current baseline** (the released model trained on ASAP only). But it is a hard constraint on *future* PDMX-based training — we built a content-dedup tool (`scripts/content_dedup.py`) and a blocklist so held-out numbers stay meaningful.

---

## 6. The Training Code (everything here is ours)

The upstream repo had no trainer. We built **`MIDI2ScoreTransformer/midi2scoretransformer/train.py`** — a full PyTorch-Lightning training stack:
- Weighted cross-entropy across all output streams (+ BCE on `pad`), AdamW + cosine schedule + linear warmup.
- **Data loaders for three regimes:** `pdmx`/`mixed` (paired synthetic), `ssl` (masked self-supervision), and ASAP (real pairs).
- **Warm-start with shape-filtering** (`load_pretrained_init`): loads the released body but reinitializes any changed/new heads — essential for the representation experiments.
- **Optimizations/levers we added (all flag-gated, default-off so the released behavior is byte-identical):**
  - `--tuplet-weight` — upweight the rare non-dyadic (tuplet) buckets in the rhythm-stream loss.
  - `--tuplet-gamma` — *reshape* the corpus to over-sample tuplet-rich scores.
  - `--beat-relative` (**B2**, see §7.4) — the within-quarter offset + `quarter_idx` representation.
  - `--decoder-full-drop` — force true generation-from-encoder (fixes a from-scratch all-pad collapse).
  - `--autoregressive` — causal-decoder from-scratch training (sidesteps the unpublished non-AR recipe).
  - Distillation / beat-conditioning hooks.

---

## 7. The Core Research Story: the Tuplet-Placement Ceiling

This is the heart of the lab report. It is a clean, well-evidenced **negative result** with a precise mechanistic explanation.

### 7.1 Where the error actually is
The per-composer breakdown shows error tracks rhythmic density. Decomposing MUSTER's sub-metrics shows it isn't pitch (PitchER is low everywhere) — it's **rhythm/onset on dense Romantic music**. Drilling in: the model **systematically under-produces tuplets** and **mis-places** the ones it does produce.

### 7.2 The decisive diagnostic
We wrote `diag_offset_phase.py`: of all notes, what fraction land at **triplet onset phases** (within-quarter buckets {4,8,16,20})? On Scriabin:
- **Ground truth: 4.78%** of notes are at triplet onsets.
- **Our best trained model: 0.0%.** It quantizes *every* onset to a binary (dyadic) position.

So the model emits tuplet **durations** but places them at **binary onsets** — a musically impossible combination (a triplet-eighth at a straight-eighth position). **The capability that's missing is triplet onset *placement*.**

### 7.3 What we tried, and what each taught us (the exhaustive matrix)
Every cell below was measured with full-14 MUSTER **and** the placement diagnostic:

| Axis | Variants tried | Triplet placement | MUSTER effect |
|---|---|---|---|
| **Inference cleanup** | pad-threshold (live), per-head decoding, **A2 metrical prior** | n/a (cleanup) | **The only robust gain: 11.79 → 11.41** |
| **Global logit shift (A1)** | Menon adjustment on duration | — | over-emits, regresses |
| **Corpus reshape** | tuplet-rate over-sampling γ=0.5→1.5 | 0% → **19.9%** (over-shoots) | regresses (drifts the model) |
| **Self-training (unpaired)** | pseudo-label MAESTRO, inject ×12 | 0.9% (under) | regresses |
| **Loss weighting** | tuplet-weight 1→5 | 0% | over-emits / regresses |
| **Beat-relative representation (B2)** | within-quarter offset + `quarter_idx` | 0% | ≈ baseline |
| **Paired distillation** | real MAESTRO timing → released score, targets carry 2.87% placement | 0% | ≈ baseline |
| **Rendered paired (high-signal)** | 23.7k rendered pairs; tuplet-rich subset carries **24.73%** placement | 0% | undertrained |
| **From scratch** | AR causal (left **undertrained**, val 1.3) *and* non-AR masked-SSL (matched released val 0.51; within ~1 MUSTER pt) | 0% | — |

### 7.4 The two cleverest attempts, and why they still failed
- **A2 metrical prior (the one durable win):** at decode time, boost durations consistent with the current onset phase (`+λ·log P(duration | offset-phase)`). It *cleans up* mis-placed tuplets — **A2 alone takes the best model 11.79 → 11.456**, and stacked with the pad-threshold + per-head decoding the inference-best is **11.41**, all with **zero retraining**. But it's a cleanup lever — it removes wrong tuplets; it can't *create* correct placement, and it actually *hurts* the already-correct released model.
- **B2 beat-relative tokenization (the literature's #1 structural fix):** we re-expressed onset as *within-quarter* position + an integer `quarter_idx`. This makes triplet onsets **common, shared buckets** {8,16} across all beats (so they should be sample-efficient to learn). We implemented it fully (lossless round-trip verified, 100% triplet concentration, non-B2 byte-identical, full model surgery + warm-start shape-filtering). **It still produced 0% placement.** The representation is *necessary but not sufficient* — it makes triplets *learnable* but doesn't make the model *learn* them.

### 7.5 The committed full-scale attempt (the final run)
We built the actual **data engine**: rendered **all 23,780 classical scores** into *paired* (performance-timing MIDI → score) data with valid per-beat alignment; the tuplet-rich subset carries **24.73% triplet placement** (5× ground truth). We then ran the **released-style recipe from scratch** (no warm-start, so it couldn't inherit a collapsed base). **Result: still 0.0% placement** (and the from-scratch AR model was also undertrained at val 1.3 vs the 0.51 baseline).

### 7.6 The conclusion (stated carefully)
We reproduced the released model's **architecture** and its **validation loss**, and came **within ~1 MUSTER point** — but never reproduced its correct-*rate* triplet placement under any lever we tried (corpus reshaping *over*-shot to **19.9%** with a MUSTER regression; everything else under-shot or collapsed near 0%). Importantly, the most promising configuration — a tuplet-aware representation (B2) trained **to convergence** on tuplet-rich paired data — was **not** completed: that from-scratch run landed **undertrained** (val 1.3 vs the 0.51 baseline).

**The leading, best-supported explanation is the training *data*.** Our corpus is ~1.7% tuplets — poor in exactly the signal that's missing — while the released model's corpus, masking/beat-alignment schedule, and music21 fork are unpublished. We did **not** prove placement is unrecoverable; we showed it is **not recovered within our compute/data budget**, and that the one experiment that would settle it (a tuplet-aware representation trained *to convergence* on tuplet-rich paired data) remains **open**. This is an honest, well-bounded characterization of a real failure mode — and it pins where the remaining accuracy most likely lives: **in the data.**

---

## 8. Honest Status Table (working / not-working)

| Component | Status | Notes |
|---|---|---|
| Audio→MIDI (Basic Pitch / hFT / MT3 / Transkun) | ✅ working | hFT best; not the bottleneck |
| `--normalize-audio`, MIDI filtering | ✅ working | hFT-only normalization |
| Block 2 released checkpoint (inference) | ✅ working | reproduces paper (11.18) |
| MUSTER eval harness + 14-piece set | ✅ working | validated; CPU-only on macOS; aborts on 3-staff |
| `eval_padsweep` single-pass sweep | ✅ working | ~8× faster standing eval |
| Training stack (we wrote it) | ✅ working | SSL / mixed / paired; warm-start shape-filter |
| Synthetic-jitter fine-tune (Track C) | ❌ negative | catastrophic forgetting (+6.9 MeanER); documented |
| Inference tweaks A1/A4/A5 | ❌ negligible/unusable | A4 breaks the 512-token limit |
| **Inference levers** (A2 + pad-thr + per-head) | ✅ **win** | 11.79 → 11.41 stacked (A2 alone 11.456), zero retraining |
| pad-threshold (we fixed a real no-op bug) | ✅ working | continuous keep-prob + raw streams |
| **B2 beat-relative** representation | ✅ implemented, ❌ didn't fix placement | lossless, validated; representation ≠ sufficient |
| Self-training / paired distillation | ✅ pipeline works, ❌ didn't fix placement | targets carry signal; model won't learn it |
| Full-scale from-scratch reproduction | ✅ ran (undertrained), ❌ placement still 0% | the open cell — not a proof of impossibility |
| **Triplet onset placement** | ❌ not reproduced within our budget | core finding; leading constraint = tuplet-poor data |

---

## 10. Appendix — Key Files

- **Pipeline entry:** `transcribe.py`
- **Block 1:** `hFT-Transformer/model/amt.py`, `mt3_inference.py`
- **Block 2 model:** `MIDI2ScoreTransformer/midi2scoretransformer/models/{roformer,model,embedding}.py`, `tokenizer.py`, `config.py`
- **Training (ours):** `.../train.py`, `.../pdmx_dataset.py`, `.../dataset.py`
- **Data engine (ours):** `scripts/{make_pairs,filter_pdmx,expressive_render,build_maestro_segments,build_maestro_paired,pseudo_label_maestro,compute_tuplet_rates,content_dedup}.py`
- **Inference levers (ours):** `scripts/{compute_duration_priors,diag_offset_phase}.py`, the `--pad-threshold`/per-head/A2 hooks in `models/model.py` + `utils.py`
- **B2 representation (ours):** `tokenizer.py` (`BEAT_RELATIVE`, `PARAMS_BR`), `scripts/{test_beat_relative_roundtrip,test_b2_model_smoke}.py`
- **Eval:** `benchmark/eval_{tier1_asap,padsweep,decomposed,baseline,improvements}.py`, `scripts/rerank_by_muster.sh`
- **Prior result docs:** `benchmark/{BASELINE,IMPROVEMENT,GPU_FINETUNE,DECOMPOSED,TRACKB}_RESULTS.md`, `LEAKAGE_AUDIT.md`
