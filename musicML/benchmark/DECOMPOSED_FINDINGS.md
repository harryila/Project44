# Decomposed Audio→Score Findings (2026-06-01)

These two diagnostic tests **overturn the project's central assumption** about
why Mazeppa-class pieces fail, and split the failure into two separately-fixable parts.

## Test 1 — Stage decomposition (transcription vs engraving)
`benchmark/eval_decomposed.py`: Stage A = hFT audio→MIDI scored vs GT MIDI with
mir_eval note-F1; Stage B = score model on GT MIDI scored vs GT score with MUSTER.

| Piece | Stage A onset-F1 | Stage B MUSTER | End-to-end |
|---|---|---|---|
| Chopin Op.10/4 | 0.974 | 1.64 | 2.75 |
| Chopin Op.25/11 | 0.983 | 1.40 | 2.76 |
| **Liszt Mazeppa** | **0.952** | **32.97** | 34 |

**Conclusion: Mazeppa's failure is ~100% Stage B (the score model), ~0% Stage A.**
Even on dense fff Mazeppa, hFT transcription is **95% onset-F1** — the transcriber is
fine. The earlier "hFT drops 18% of notes / transcription is the bottleneck" framing
was **wrong** (it came from a stale `_hft.mid` artifact). The transcriber is not the
problem on any piece. **Do not replace/ensemble the transcriber.**

## Test 2 — Mazeppa is IN ASAP; the benchmark over-stated its failure
**Mazeppa (Transcendental Étude No.4) is in ASAP training data** — 11 performances +
engraved score (`data/asap-dataset/Liszt/Transcendental_Etudes/4/`). So the
"Mazeppa fails because it's out-of-distribution repertoire" diagnosis — which the
whole project narrative rested on — is **false**.

Running the model on ASAP Mazeppa performances, scored vs the **ASAP** score (the
edition it trained on) instead of our PDMX-sourced benchmark GT:

| Eval setup | MeanER |
|---|---|
| ASAP perf (Cai03) vs ASAP score | **16.9** |
| ASAP perf (ChenC02) vs ASAP score | **13.9** |
| MAESTRO perf vs **PDMX** score (our benchmark) | 33.8 |

**This splits the 33.8 into two distinct, quantified problems:**

1. **~half is a SCORE-EDITION MISMATCH (eval artifact).** Our benchmark scored against
   a PDMX-sourced Mazeppa GT that is a *different engraving edition* (different
   voicing/spelling/measure layout) than what the model learned. Matched-edition
   scoring drops MeanER ~34 → ~14-17. **The benchmark over-stated the failure ~2×.**
2. **~14-17 is STILL bad** (vs Chopin ~1.4-1.6, test-set avg 11.1) — and this is on a
   piece the model **trained on**. So there is *also* a genuine **dense-polyphony
   capacity limit**: even having seen Mazeppa, the model only reaches ~14 because
   14-notes/sec polyphony is hard for this architecture/tokenization (Miss 17-22%,
   Extra 13-15% — notes dropped/scrambled in dense passages).

## What this means for "Songscription parity"
The real target is now concrete, not vague:
- **NOT a transcription problem** (hFT is 95-98% F1 everywhere).
- **NOT primarily an OOD-repertoire problem** (Mazeppa was in training).
- **It is (a) eval methodology** — score against matched editions; our PDMX GTs inflate
  error — **and (b) score-model capacity on dense polyphony** — the concrete thing a
  better-resourced team (Songscription) would improve: bigger model, better
  dense-polyphony tokenization, and far more dense training pairs.
- The clean-piano pipeline **matches commercial quality** on easy repertoire (2.75) —
  not a like-for-like win, since the Block-2 model is the same released checkpoint the
  product runs (apples-to-apples = the **11.18** reproduction). The gap to "product
  level" is specifically the dense-virtuoso tail, and it lives in the score model's
  capacity, not the data coverage or the transcriber.

## Immediate corrections to make
1. **Re-anchor the benchmark to matched editions.** The Mazeppa "34" is partly an
   artifact; the honest number is ~14-17. Re-score the benchmark using ASAP scores
   where the piece is in ASAP.
2. The Tier-1 11.10 test-set number is unaffected (it already uses ASAP's own scores) —
   that remains the valid SOTA-reproduction anchor.
