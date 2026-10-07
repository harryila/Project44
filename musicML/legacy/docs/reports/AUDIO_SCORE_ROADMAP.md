# Audio → Score Roadmap — closing the gap to "Songscription level" on real piano

The pivot after concluding Block-2 (MIDI→score) is at its MUSTER ceiling. This doc
scopes the *real* remaining work: end-to-end **audio → score** robustness on real
recordings — which is where the gap to a Songscription-grade product actually lives.

> **Status:** scope + plan, grounded in empirical decomposition. No production code
> changed yet. See `benchmark/GPU_FINETUNE_RESULTS.md` for why Block-2 is done.

---

## 0. The empirical decomposition (CORRECTED 2026-06-01 after testing)

> **Important correction.** An earlier version of this doc claimed Mazeppa's failure
> was the hFT transcriber dropping 18% of notes. **That was wrong** — built on a stale
> `_hft.mid` artifact (5,858 notes) from old settings. Re-running the current pipeline
> and testing the decomposition directly overturned it. The corrected findings below.

I split the end-to-end pipeline into its two stages on the 3 benchmark pieces
(real MAESTRO Disklavier audio + GT MIDI + GT score):

| Piece | hFT note recovery | Block-2 on CLEAN GT MIDI | **End-to-end MUSTER** |
|---|---|---|---|
| Chopin Op.10/4 (clean) | near-perfect | excellent | **2.75** ✅ |
| Chopin Op.25/11 (clean) | near-perfect | excellent | **2.76** ✅ |
| Liszt Mazeppa (dense fff) | **92%** (6,562/7,158) | **MeanER 33.8** ❌ | **34.2** ❌ |

**The decisive test (Tier-1 on Mazeppa):** feeding the *perfect* GT Disklavier MIDI
(7,158 notes) into the score model gives **MeanER 33.8 — essentially identical to the
end-to-end 34.2.** So the transcriber is NOT the bottleneck; with perfect input MIDI
the score model still collapses on Mazeppa.

**Corrected facts:**
1. **On clean/in-distribution audio the pipeline already beats Songscription's
   published number** (MeanER 2.75 end-to-end vs. their paper's 11.30). True.
2. **Mazeppa's catastrophe is the BLOCK-2 score model failing on out-of-distribution
   dense Liszt** — the same data-bound ceiling established in `GPU_FINETUNE_RESULTS.md`.
   hFT recovers 92% of notes (fine); the GT note counts match (7,158 vs 7,112 score);
   GT self-noise is 2.1 (so 34 is genuine 16× error). The model produces a
   structurally-scrambled score (wrong meter/measures/voicing → 37% mutual miss+extra
   that MUSTER can't align), exactly the OOD failure the original Tier-1 work found
   (3/4 predicted instead of 4/4, 322 measures vs 167).
3. **`transcript_stride` (overlapping windows) is a null** — no note recovery, 2× slower.

**So there is NO separate "audio→score transcription gap" to chase.** The end-to-end
behavior collapses to what we already knew: excellent on seen repertoire, OOD-limited
on hard repertoire — a Block-2 generalization limit (at its ceiling), not Block-1.

---

## 1. What the corrected diagnosis means

The "swap in a better transcriber" plan is **moot** — the transcriber was never the
bottleneck (Tier-1 on perfect MIDI = 33.8 ≈ end-to-end 34.2). hFT is fine (92%
recovery, near-SOTA; Marták et al. 2025 rate it among the most density-robust models).
Do **not** replace it, do not chase transkun, do not ensemble. That was built on a
stale-artifact premise that testing overturned.

The end-to-end picture collapses to two regimes, both already understood:

| Regime | End-to-end MUSTER | Bottleneck | Status |
|---|---|---|---|
| Clean / in-distribution piano | **~2.75** (beats published SOTA 11.30) | none — both blocks excellent | **solved** |
| OOD dense virtuoso (Liszt/Rach) | **~34** | Block-2 score model on OOD repertoire | at the data-bound ceiling (`GPU_FINETUNE_RESULTS.md`) |

There is **no third "transcription gap" lever.** The hard-virtuoso tail is the *same*
Block-2 generalization limit we exhaustively proved is data-bound and at its ceiling
with the public data we can access (synthetic hurt it; mixing neutral; more
same-source data leaks/drifts).

## 2. So what's actually worth doing

Three honest options, none of which is "chase MUSTER lower":

1. **Bank it.** The result is complete and strong: reproduced SOTA, proved Block-2 is
   at its ceiling, proved (via decomposition) the transcriber is fine, and showed the
   pipeline *beats* the published number on normal piano. The hard-virtuoso tail is a
   known limit the underlying model (and likely Songscription itself — reviews note its
   rhythm struggles on hard material) shares.

2. **Build the decomposed eval (still genuinely valuable, ~3–4 days, no GPU).** The
   n=3 benchmark is too thin to characterize the product. Add a `mir_eval` audio→MIDI
   note-F1 stage (decouples transcription from engraving) and 10× the benchmark using
   MAESTRO audio + ASAP scores (data partly on disk; ~120GB MAESTRO download). This
   gives a *defensible product-quality number* across difficulty strata with CIs —
   useful for a lab deck or product claims even though it won't *raise* accuracy.
   License: MAESTRO/ASAP/ACPAS are CC-BY-NC-SA → eval-only, quarantined from training.

3. **Product-framing work (if the goal is a usable tool, not a benchmark number).**
   The honest product answer for the hard tail is **confidence-flagging**: detect
   low-agreement / high-density regions and surface them as "needs human review,"
   matching how every real transcription product (incl. Songscription) actually ships.
   This is engineering, not modeling.

## 3. Recommended next action
**Option 1 or 2.** We've reached the honest end of the *accuracy* road on available
data. If you want a stronger deliverable, **build the decomposed 10× eval (Option 2)** —
it converts "we beat SOTA on 2 pieces" into a statistically defensible characterization
of where the product works and where it doesn't. If you'd rather stop, **bank it** —
the conclusion is complete and correct.

## 4. Avoid
- Don't swap/ensemble transcribers — the transcriber is not the bottleneck (verified).
- Don't tune hFT thresholds or `transcript_stride` — both proven null.
- Don't enable `--normalize-audio` on already-hot recordings.
- Don't expect *any* available-data intervention to fix the OOD dense-Liszt tail —
  it's the Block-2 ceiling, already exhaustively established.
- Don't let MAESTRO/ASAP/ACPAS (non-commercial) touch any training pool.
