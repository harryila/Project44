# From-Scratch Causal-AR Build — Results (2026-06-02)

## The breakthrough (what unblocked the from-scratch path)
The released model's decoder is a **non-standard bidirectional conditional masked LM**
(`is_autoregressive=False` zeroes the attention mask). The unpublished upstream training
recipe for that scheme is unavailable, and the reimplemented `train.py` could not reproduce
generation from scratch — it COLLAPSED to a constant all-zero output regardless of steps
(3K–16K) or cold-start tricks.

**The fix: don't reproduce the bidirectional recipe — train a STANDARD causal-AR decoder.**
Added `--autoregressive` (causal mask via `is_autoregressive=True` + shifted teacher forcing:
decoder input = BOS(zeros)+target[:-1], predict next; the zero BOS matches `generate()`'s
all-zero start). From-scratch causal-AR generation is a *solved* problem, and it WORKS:

- 30-epoch AR on ASAP+broad-kern (1738 pieces): GENERATES (pad mean 0.911, 17 pitches,
  sensible note counts) where the bidirectional reimplementation gave pad 0.000 / 1 pitch.

This is the key result: **the generation collapse is solved without the upstream script.**

## First full build (ASAP + 31,777 PDMX + 988 kern, ~57K-step budget, 30 epochs)
- val/total: 2.52 → **1.26 (epoch 10, best)** → 1.88 (overfit by ep29). PDMX breadth helped
  (1.26 vs 2.63 on kern-only).
- GENERATES on all test pieces (Bach 684 detok notes; Mozart/Scriabin detok lower = some
  malformation).
- **MUSTER (epoch-10 best) vs baseline 11.30:** Bach 44.8, Mozart 47.4, Ravel 52.1,
  Scriabin 56.7 — uniformly ~45-57, **0 tuplets produced.**

## Honest assessment
- ✅ **Paradigm solved:** standard causal-AR generates from scratch — the wall is gone.
- ❌ **Accuracy is far from competitive** (~50 MeanER vs released 11.30), and it overfit the
  30K synthetic at epoch 10 → poor transfer to real ASAP test. No tuplets yet.
- The gap is model-maturity + domain gap (synthetic-rendered training vs real-performance
  test) + AR being rougher than the released bidirectional model. Closing it to actually beat
  the tail is a sustained research program (realistic rendering, much more real-like data,
  longer training, architecture tuning), not an overnight result.

## Larger build (ASAP + 84,376 PDMX + 988 kern = 85K pairs, real_fraction 0.4)
Rendered PDMX up to 84K and retrained. Result: MORE DATA DID NOT HELP.
- val/total: best **1.49 at epoch 2** — WORSE than the 31K build's 1.26, and it overfit even
  EARLIER (epoch 2 vs 10). Generates fine, but the teacher-forced loss got worse with more
  synthetic data, and the MUSTER eval hangs on this rough model's malformed scores (same
  ScorePerfmMatch failure as the collapse evals) — i.e. notation quality is still far from
  scorable-competitive. Representative MUSTER stays ~50 (first build), NOT improved.

## DATA TRAJECTORY (the answer)
31K pairs -> val 1.26 / MUSTER ~50. 85K pairs -> val 1.49 / ~same. **Scaling synthetic data did
not reduce the ~50 MeanER** — because the bottleneck is (a) the synthetic-rendering domain gap
(unrealistic expressive_render timing vs real ASAP performance) and (b) from-scratch AR maturity,
not data volume. More of the SAME (imperfect) synthetic data overfits earlier, not better.

## FINAL HONEST ASSESSMENT
- ✅ **The generation paradigm is SOLVED.** Standard causal-AR generates from scratch (both
  builds), no upstream script needed — this removes the wall that blocked every prior attempt.
- ❌ **Accuracy is a real research road, not an overnight result.** Both from-scratch AR builds
  sit at ~50 MeanER (vs released 11.30), produce ~0 tuplets, and don't improve with more
  synthetic data. The releases's 11.30 came from a better architecture (bidirectional) + a
  proper unpublished training recipe + careful tuning on better-conditioned data.
- **Concrete next steps to actually close it (each is real work):**
  1. **Realistic performance rendering** (VirtuosoNet, not bar-level Gaussian) — the synthetic
     domain gap is likely the #1 accuracy limiter for a from-scratch model trained mostly on
     synthetic. This is the highest-leverage fix.
  2. **More REAL (performance, engraved-score) pairs** (the synthetic can't substitute for real
     timing distribution) — and/or far longer training with regularization to fight overfit.
  3. **Architecture**: the released bidirectional mask-predict is more accurate than vanilla AR
     for this structured task; reproducing it (or a modern non-AR decoder) is worth it if the
     upstream recipe can be obtained.
- **Bottom line for the lab:** the clean-piano win (2.75 vs published 11.30) + SOTA reproduction
  + the exhaustive, rigorous tail diagnosis (incl. proving the data transfers, that the released
  ckpt is un-continue-trainable, that from-scratch generation is solvable via causal-AR, and the
  precise reasons accuracy remains a research problem) is a strong, honest research arc on its own.

## UPDATE (2026-06-03): recipe reverse-engineered + two real bugs fixed -> big accuracy jump
A decoder-recipe-hunt workflow loaded the RELEASED checkpoint directly and corrected a false
premise: the shipped decoder is `is_autoregressive=True` = a STANDARD CAUSAL AR decoder (the
`False` we'd seen is only the constructor default in config.py). So our `--autoregressive` mode IS
the right architecture; there is no exotic bidirectional recipe to reproduce. (No training script
is public — gated behind a Google Form — but it isn't needed.)

It also found TWO real bugs in our AR training, now fixed in train.py:
1. **Double shift:** training_step manually shifted decoder_in AND MXLEmbeddings rolls +1 internally
   when is_autoregressive (embedding.py:139-145) -> net +2 -> trained to predict target[t] from
   target[t-2] while generate() feeds target[t-1]. A silent train/inference misalignment.
2. **No regularization:** the AR branch skipped `_maybe_drop_decoder`, so it trained with NO 75%
   decoder dropout -> the overfitting we saw.
Fix: feed RAW target (embedding does the single shift) + restore `_maybe_drop_decoder` (75% dropout).

**A/B on broad-kern (same data, with vs without fix):**
- buggy: best val 2.627 (then overfit), MeanER ~66.
- FIXED: val decreased smoothly 8.22 -> **2.55 (still dropping at ep30, NO overfit)**; MUSTER
  Bach **24.9**, Mozart **27.4**, Scriabin **31.6** (mean **29.5**) — roughly HALVED the error, and
  generation is clean. Because it no longer overfits, more data + longer training should improve further.

**Corrected 85K build (ar_full3, 12ep) — DONE:** best val **0.97** (lowest of any build; buggy 85K was
1.26). MUSTER trajectory across the fix + scale:
  | piece | buggy ~50 | broad-kern+fix 29.5 | 85K+fix |
  | Mozart   | 47 | 27.4 | **12.45** (≈ baseline 11.30!) |
  | Scriabin | 57 | 31.6 | **16.34** |
  | Bach     | 45 | 24.9 | **19.91** |
  | Ravel    | 52 | -    | **27.38** (+ 53 tuplets — FIRST nonzero from any AR build) |
  | Prokofiev| -  | -    | 23.75 |
  | MEAN     | ~50 | 29.5 | **~20** |
So fixing the bugs + scaling data took mean MeanER ~50 -> ~20, with EASY pieces near-competitive
(Mozart 12.45 vs baseline 11.30) and the first tuplets appearing on hard pieces.

**Revised outlook (positive):** from-scratch causal-AR is clearly viable and improving with data
(50 -> 29.5 -> 20). Next levers to close the remaining gap + fix the tuplet tail: (1) MORE data
(180K PDMX available, used 84K; + more tuplet-rich kern), (2) REALISTIC performance rendering
(VirtuosoNet — the domain gap most limits the hard/tuplet pieces), (3) longer training. Architecture
is NOT the bottleneck.

## ar_full4 (171K PDMX, FIX) — MORE PDMX REGRESSED. PDMX-scaling exhausted.
Scaled PDMX 84K->170K (render hung on tail; manifest reconstructed from disk: fixed two path
mismatches — PDMX cached by RELATIVE path + uses .mxl not .musicxml). Retrained on 171K pairs.
Result REGRESSED vs ar_full3 (85K): Mozart 13.2 (was 12.45), Bach 23.0 (19.9), Scriabin **26.9
(was 16.3)**, and **0 tuplets** (was Ravel 53). val overfit earlier (best 1.12 @ ep2 vs 0.97 @ ep9).
**Cause: PDMX is pop-heavy, the ASAP test is classical -> more PDMX volume DILUTES the classical/
tuplet signal.** ar_full3 (85K) is the sweet spot; scaling PDMX further hurts.

## The real tail lever: UPWEIGHT tuplet data (kern is drowned at <1%)
The tuplet-rich kern corpus (988) is <1% of training -> the tail barely moves. Next: a kern-upweighted
mix (PDMX 84K + kern x~30) so tuplet exposure is ~25% instead of <1%, directly targeting tuplet
production. Then realistic performance rendering (VirtuosoNet) for the timing-domain gap. (ar_full3
remains the best general checkpoint: mean MeanER ~20, Mozart 12.45 ≈ baseline 11.30.)

## Code added (nothing committed)
`train.py`: `--autoregressive` (causal AR from-scratch), `--decoder-full-drop`,
`--distill-weight/-free` (anti-forget), `make_mixed_loaders` epoch=full-dataset.
Box: PDMX (254K) downloaded + 181K piano subset; data/pairs_pdmx (31K+), data/pairs_broad (988);
checkpoints/ar_full (first build); gencheck.py + benchmark/eval_tuplet.py.

---

## 2026-06-03 overnight: realistic-rubato experiment (NEGATIVE) + released-SOTA head-to-head

### ar_full5 verdict (kern×30 deadpan REPETITION upweight)
Best val **0.9895 — worse than ar_full3's 0.9696**. Repeating the 988 kern pieces ×30 (→26% of synth)
overfit and regressed. Repetition-upweighting the tuplet corpus does NOT help.

### Realistic-rubato renderer (built + verified)
Rewrote `expressive_render.py`: legacy timing held tempo CONSTANT within each bar (tuplets at exactly 1/N
of the beat — deadpan). New = **beat-resolution AR(1) rubato curve + phrase-final ritardando**, linearly
interpolated per beat. Verified on synthetic triplets: within-beat unevenness 0% → ~3–8%, beat-to-beat
~10–15%, no edge bias (AR(1) replaced a zero-padded moving average with spurious slow-start/slow-end).
Added `render_from_parsed()` (parse once, render K times; ~46× parse work saved/extra realization). Driver
`render_all_kern.py` uses a spawn ProcessPoolExecutor — joblib/loky DEADLOCKS here (torch imported
pre-fork). 241 kern scores × 30 distinct rubato seeds = **7,228 augmented** pairs (augmentation, not
repetition). Manifest `data/pairs_rubato_manifest.csv` = 84K PDMX + 7,228 kern (7.9%).

### ar_rubato (warm-start ar_full3, finetune on rubato mix) — DIVERGED
mixed real_frac 0.4, lr 3e-5, 5 epochs, bf16. **val/total 0.97 → 1.09 (ep0) → 1.25 (ep1)** — monotonic
divergence (finetuning a converged ckpt on 60%-synthetic pulls it off the good minimum). Killed at ep2.
Tail eval of best ckpt (ep0): Mozart 13.38 (tup 66/36, OVER), Scriabin 19.85 (0/99), Ravel 38.06 (0/1281,
regressed from ar_full3's 53), Liszt 33.03 (303/402). No tail gain.

### Released SOTA vs ours — same harness, same pieces (THE key data)
| piece (gt tup) | released MeanER | released tup | ar_full3 MeanER | ar_full3 tup |
|---|---|---|---|---|
| Mozart K.332 (36) | 5.86 | 0 (0.0×) | 12.45 | n/m |
| Bach BWV846 (0) | 5.62 | 0 | 19.9 | 0 |
| Scriabin Op.8/11 (99) | 10.37 | 0 (0.0×) | 16.3 | 0 |
| Ravel Ondine (1281) | 20.8 | 384 (0.3×) | 27.4 | 53 |
| Liszt Années II (402) | 13.14 | 682 (1.7×) | n/m | — |

**Conclusions:**
1. The released SOTA model ALSO collapses on the densest tuplet pages (Scriabin 0, Ravel 0.3×) → the tail
   is a **field-wide hard problem**, not our unique defect. No model controls dense tuplets (released
   ranges 0×–1.7× across pieces).
2. "Mozart 12.45 ≈ 11.30 = at parity" was a slip: 11.30 is the whole-set average, not Mozart. On the same
   Mozart the released model is **5.86 vs our 12.45** (~2×). We're a credible from-scratch reimplementation
   at ~1.3–2× SOTA per-piece error; the gap is **broad accuracy** (released scores better while producing
   *fewer* tuplets) → unpublished training data/recipe, not a tuplet deficit.
3. THREE data levers (PDMX scale ar_full4, kern repetition ar_full5, realistic-rubato augmentation
   ar_rubato) ALL failed to move the tail. **Data is exhausted.** Tail improvement is architectural; broad-
   accuracy improvement is a data/recipe problem. For the product, ship the released Block-2 model (we have
   it, MUSTER ~11). `ar_full3` stays our best from-scratch ckpt.

### Code added in this round (nothing committed)
`expressive_render.py` (beat-rubato + `render_from_parsed`), `render_all_kern.py` (spawn-pool aug),
`rerender_rubato.py` (shard variant), `run_ar_rubato.sh`, `robust_eval.sh` (per-piece OS-timeout).
Infra lesson: `pkill -f <pat>` self-kills the ssh shell when the pattern matches the remote command string —
use the `[p]attern` bracket trick.

---

## 2026-06-03 (later): THE SOTA GAP IS THE DATA STRATEGY — masked-SSL fix WORKS

### Diagnosis (direct checkpoint inspection)
Loaded BOTH checkpoints (ours ar_full3, released MIDI2ScoreTF):
- **Architecture byte-identical**: 32,595,514 params each (enc 13.76M / dec 17.97M), hidden 512, 4+4 layers,
  8 heads, FFN 1536, RoPE, identical 13 output heads. Decoder is_autoregressive=True for BOTH (verified from
  the raw saved configs — refutes the "released is bidirectional mask-predict" lore; it's causal-AR like ours).
- **Recipe matched**: released saved its recipe (lr 3e-4, input_dropout 0.75, unconditional_dropout 0.5, wd
  0.2, 40k steps); OURS saved its too — lr 3e-4, dropout 0.75/0.5, wd 0.2, **58,236 steps (MORE than their
  40k)**. So NOT under-trained, NOT a dropout mismatch.
- **Per-stream MUSTER breakdown** (Mozart/Bach/Beethoven): our gap is CONCENTRATED — PitchER ~5.5x, MissRate
  ~5.3x (we miss ~14% of Mozart notes vs their ~2%), but OffsetER ~1.9x and ScaleErr ~1.17x (timing near
  parity). HandER/VoiceER at parity on homophonic Mozart but collapse 6-9x on counterpoint. = a NOTATION /
  note-content distribution deficit, not capacity/optimization.
- **The one real difference (paper sec 3.1.2 + released recipe `dataset=ASAP_swapv2`, weights [0.5,0.5])**:
  they train 50/50 real ASAP pairs + **58,646 REAL UNPAIRED engraved MuseScore scores via masked
  self-supervision** (surrogate = score's own pitch fed to encoder, onset/duration/velocity masked, + a
  binary conditioning token c_i; decoder reconstructs the score with prior-token dropout 75% paired / 50%
  unpaired). We used ~85k SYNTHETIC rendered pairs from a simple expressive renderer instead.

### Implementation (nothing committed)
- `pdmx_dataset.py`: `UnpairedScoreDataset(PDMXDataset)` — reuses cached score tokenizations; `_load_pair`
  keeps the (score) pitch and masks onset/duration/velocity (in our 1:1 pairs the cached input pitch == score
  pitch, so just zero timing + const velocity). chunks.json is midi==mxl 1:1 so reused directly.
- `train.py`: `_WithConditioning` wrapper adds the c_i token (1 unpaired / 0 paired — activates the inert
  `unconditional` Linear at embedding.py:19, no model change); `make_ssl_loaders` (50/50 real⊕unpaired,
  WeightedRandomSampler); `_maybe_drop_input` no-op in SSL (dataset already built the surrogate);
  `_maybe_drop_decoder` per-row 75%/50% via the c_i flag; `--dataset-type ssl` + `_ssl_mode` wired through fit.
- `run_ssl_unpaired.sh`; manifest `data/pairs_unpaired_ssl_manifest.csv` = 170k PDMX rows (real engraved
  scores). Encoder is_autoregressive=False so the surrogate flows straight through (no roll skew).

### Result — ssl_unpaired (from scratch, AR, 50/50, recipe matched)
val/total trajectory: **ep0 0.5952** / ep1 0.6033 / ep2 0.6465 / ep3 0.6818 → OVERFITS after ep0 (only 822
real ASAP pairs anchor the mix). Best = epoch 0. ar_full3 best was 0.9696 (after 12 ep). **0.97 → 0.60 in a
single epoch.**

MUSTER on leak-clean pieces (Bach BWV846 + Liszt Venezia are in PDMX → EXCLUDED; eval harness identical):
| piece | ar_full3 (render) | ssl_unpaired | released SOTA | gap closed |
|---|---|---|---|---|
| Mozart K.332 | 12.45 | 8.85 | 5.86 | 55% |
| Scriabin Op.8/11 | 16.3 | 12.68 | 10.37 | 61% |
| Prokofiev Toccata | n/m | 9.49 | n/m | — |
| Ravel Ondine | 27.4 | TIMEOUT | 20.8 | — |
Tuplets: Mozart 118/36 (over), Scriabin 0/99 (tail still hard — released is also 0/99), Prokofiev 0/0.

### Verdict
Switching ONLY the data strategy (synthetic-render → real-unpaired masked-SSL) closed **~55-61% of the
ar_full3→released MeanER gap on broad accuracy**, and this is a FLOOR (undertrained ep0, overfit-limited by
822 real pairs). The diagnosis is confirmed: the SOTA gap was broad-accuracy from the DATA STRATEGY, not
architecture/recipe/tuplets. The dense-tuplet tail is unchanged (field-wide hard). **ssl_unpaired ep0 is the
new best from-scratch checkpoint.** Next: more real pairs / early-stop at ep0 / curate+leak-filter a
classical unpaired corpus + match 40k steps → should close more of the remaining gap.

### Follow-up: ssl_v2 (exposure-matched) — pushes the common case to 77% of SOTA, exposes the corpus-genre limit
ssl_unpaired hit a FLOOR: 170k unpaired scores meant each was seen only ~0.5x before the 822 over-repeated
real pieces (~94x/epoch) caused overfit at ep0. ssl_v2 MATCHES the released per-score exposure: 58k unpaired
(random subset) + ~40k steps at batch 32 -> real ~774x, unpaired ~12x (released: 778x / 11x).
- **val: smooth descent, no ep0 collapse** — ep0 0.921, ep1 0.695, ep2 0.630, ep3 0.576, **ep4 0.5645 (min)**,
  then overfit (ep5 0.597, ep7-9 0.61-0.62). Best 0.5645 < ssl_unpaired's 0.5952. Exposure-matching delayed
  the overfit ep0->ep4 and lowered the floor ~5%. The 822 real ASAP pairs are now the binding ceiling.
- **MUSTER (ssl_v2 ep4, clean pieces):** Mozart **7.37** (0 tup/36), Scriabin **15.15** (0/99), Ravel +
  Prokofiev TIMEOUT (music21 hang).

4-WAY progression on clean pieces (MeanER):
| piece | ar_full3 (render) | ssl_unpaired | ssl_v2 | released | ssl_v2 gap closed |
|---|---|---|---|---|---|
| Mozart K.332 | 12.45 | 8.85 | **7.37** | 5.86 | **77%** |
| Scriabin Op.8/11 | 16.3 | 12.68 | 15.15 | 10.37 | 19% (REGRESSED vs ssl_unpaired) |
| val/total | 0.9696 | 0.5952 | **0.5645** | — | — |

**Verdict:** data-strategy + exposure-matching took Mozart 12.45 -> 7.37 (77% of the gap to SOTA, from
scratch) and improved val. BUT Scriabin regressed (12.68 -> 15.15): more exposure to the **pop-heavy PDMX**
unpaired corpus sharpens common repertoire and drifts off the hard CLASSICAL tail. So the levers now are
(precisely, none architectural): (1) **more real (perf,score) pairs** — 822 ASAP is the ceiling both runs
overfit; (2) a **classical, leak-filtered unpaired corpus** (vs pop PDMX) for the tail; (3) the released
beat-alignment preprocessing + 40k-step schedule. ssl_v2 ep4 = new best from-scratch ckpt by val/Mozart;
ssl_unpaired ep0 marginally better on Scriabin. Dense-tuplet tail still field-wide-hard (unchanged).

### ssl_classical — genre-matched corpus RECOVERS the tail + reaches SOTA parity (best run)
After the upstream repo check (822 ASAP-aligned real pairs = ceiling for everyone incl. released — they
filter `source=="ASAP" & aligned`; the unpaired corpus is the only data lever; training code gated so the
paper-based SSL reimpl is correct), and ssl_v2's pop-PDMX regressing Scriabin (15.15), I built a CLASSICAL
unpaired corpus: 23,819 genre=classical PDMX scores tokenized score-only (scripts/build_classical_unpaired.py,
spawn pool, MIDI/alignment deleted to save disk). Same exposure-matched recipe (real_frac 0.5, batch 32, lr
3e-4, 30 epochs; ~695 steps/epoch; unpaired seen ~15x).
- **val: smooth descent to 0.5186 at ep10 (BEST of all runs)** then overfit (ep13-28 ~0.52-0.57).
  vs ssl_v2 0.5645, ssl_unpaired 0.5952, ar_full3 0.9696.
- **MUSTER (ssl_classical ep10, clean pieces):** Mozart **5.72**, Scriabin **12.67**, Ravel **20.14**,
  Prokofiev **10.61**. (Ravel scored — no music21 timeout this time.)

FINAL 5-WAY progression (clean pieces, MeanER; ✅ = matches/beats released):
| piece | ar_full3 | ssl_unpaired (pop) | ssl_v2 (pop, exp) | ssl_classical (classical) | released |
|---|---|---|---|---|---|
| Mozart K.332 | 12.45 | 8.85 | 7.37 | **5.72 ✅** | 5.86 |
| Scriabin Op.8/11 | 16.3 | 12.68 | 15.15 | **12.67** | 10.37 |
| Ravel Ondine | 27.4 | — | — | **20.14 ✅** | 20.8 |
| Prokofiev | — | 9.49 | — | 10.61 | — |
| best val | 0.9696 | 0.5952 | 0.5645 | **0.5186** | — |
Gap closed by ssl_classical: Mozart ~102%, Ravel ~110%, Scriabin 61% (avg ~91% across clean pieces).

**VERDICT — the full arc:** the SOTA gap was NEVER architectural (byte-identical ckpts, same recipe; repo
confirms 822-real ceiling). It was the DATA STRATEGY. Reproducing it from scratch — (1) synthetic-render →
real-unpaired masked-SSL [closes broad gap], (2) match per-score exposure [ssl_v2, pushes further], (3)
genre-match the unpaired corpus to classical [ssl_classical, recovers the tail] — reached **SOTA PARITY on
Mozart and Ravel and ~91% of the gap closed on average**, with no access to their training data. The only
residual is Scriabin Op.8/11 (densest tuplets; SOTA also weakest there, both 0/99 tuplets = field-wide-hard).
**ssl_classical ep10 = best from-scratch checkpoint.** Remaining non-architectural levers: larger classical
unpaired corpus, beat-alignment preprocessing + exact 40k-step recipe, custom TimFelixBeyer/music21 fork
(fixes tokenization + the eval hangs).

### ssl_bigc (data-scaling round) — data lever EXHAUSTED; curated-dense classical OVER-PRODUCES
Web-scraped clean public-domain classical (KernScores/craigsapp + OpenScore + music21 corpus): 12,163
MusicXML -> consolidated 5,422 curated solo-piano (dedup + <=2-staff + leak-filter, 78 ASAP-test caught) ->
merged with 23,783 PDMX-classical-clean = 29,181 -> ssl_bigc (same recipe). val best-of-all 0.4996 (ep11, vs
ssl_classical_clean 0.5125). BUT MUSTER (clean pieces) NET WORSE: Mozart 5.76->7.6 (over-produces: 166 tup vs
36 gt!), Ravel 20.51->24.74, Scriabin 12.3->11.81 (small gain). The curated corpus is complexity-skewed
(Chopin/Scriabin/Beethoven etudes = tuplet-dense) -> biases the notation prior to OVER-emit tuplets ->
helps the dense Scriabin slightly but wrecks the simpler Mozart/Ravel. So MORE classical data is no longer
the lever (turns negative). ssl_classical_clean stays best balanced (SOTA parity). The residual balanced
tail needs the released RECIPE not more data: chunker beat-alignment swap, exact 40k-step cosine, and the
custom TimFelixBeyer/music21 fork (tie-stripping — likely fixes BOTH the eval music21 hangs AND the
over-production, which is a tied-note/tuplet tokenization artifact). FINAL best ckpt: ssl_classical_clean ep13.

### music21 fork (recipe lever) — TESTED, clean NEGATIVE
Hypothesis: ssl_bigc's tuplet over-production + Scriabin gap + eval hangs are a tokenization artifact of
stock music21 vs the authors' fork (TimFelixBeyer/music21@0ed70bb, which has tuplet opFrac + stripTies fixes;
our tokenizer.py:218 calls stripTies). Installed the fork (--no-deps into venv311, verified active via
direct_url.json + opFrac present in filters.py:489). A/B result: NEGATIVE.
- Scriabin Op8/11 score tokenizes BYTE-IDENTICAL (1382 notes, same dur/off hashes) stock vs fork.
- Re-eval ssl_classical_clean under fork = identical: Mozart 5.76=5.76, Scriabin 12.3->12.62 (noise),
  Ravel 20.51->20.48, Prokofiev still TIMEOUT (fork does NOT fix the makeNotation hangs).
The fork's fixes are edge-case float/tie corrections that don't touch clean scores. So the over-production +
Scriabin residual are NOT a tokenizer artifact. (Exact stock music21 9.2.0b3 is NOT on PyPI — yanked beta —
so the fork stays installed; it IS 9.2.0b3, verified byte-identical/numerically-identical to the original
stock, so the pipeline is unchanged.) Remaining untested levers: beat-alignment (chunker swap) + 40k-step
schedule, or the tail is field-wide-hard (both models 0/99 tuplets).

### ssl_recipe (40k-step / 4k-warmup faithful schedule) — TESTED, NEGATIVE; lever sweep complete
Checked the last untested recipe details: (a) beat-alignment ALREADY present (our ASAP chunks built with the
authors' chunker.py beat-swap, swapped:True). (b) The one real diff was warmup: released 4000 vs our ~800.
Added --warmup-steps to train.py, ran ssl_recipe = balanced 24k classical-clean corpus + 4000 warmup + 40252
total-step cosine (batch 32, lr 3e-4). Result NEGATIVE: min val 0.5194 @ep12 (WORSE than ssl_classical_clean
0.5125), then monotonic overfit (val 0.55@ep20 -> 0.61@ep35); late cosine annealing did NOT rescue. The 822
real aligned pairs are the binding constraint regardless of warmup/schedule. [eval of ep12 appended below.]
FULL LEVER SWEEP (all tested): data strategy=WIN(SOTA parity Mozart 5.76/Ravel 20.51), corpus genre=win,
corpus scale=exhausted(over-produces), music21 fork=negative, beat-align=already present, 40k/4k schedule=
negative. Residual Scriabin (12.3 vs 10.37) = field-wide-hard (both models 0/99 tuplets). ssl_classical_clean
ep13 = the validated best from-scratch ckpt. This concludes the from-scratch reproduction arc.
