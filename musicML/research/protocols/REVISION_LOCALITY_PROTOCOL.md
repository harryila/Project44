# Revision-locality secondary analysis

Recorded: 2026-08-24, before inspecting locality results.

## Scope

This is a descriptive secondary analysis of the six frozen N=32 pools with at least 20 exact ground-truth tuplet notes. It does not alter the decoder, selector, candidate pools, candidate order, oracle rule, MeanER guard, or any paper headline result.

For each piece, compare the greedy baseline with the candidate already selected by the frozen N=32, baseline-inclusive, +0.50 MeanER-guarded oracle in `benchmark/oracle_prefix_v10_1.json`. Ground truth was used previously to choose those oracle candidates. It is not used to define or measure the stream differences below.

## Position and stream definitions

A position is one performed-note input index in the tokenized test sequence. Baseline and sampled outputs have one aligned output position per input index, so this comparison does not align or diff serialized MusicXML.

At each position, extract the decoded keep decision and the argmax offset, duration, and downbeat classes. A rhythm position is changed when either:

1. the keep decision changes; or
2. both outputs keep the position and at least one of offset, duration, or downbeat changes.

Changes to offset, duration, or downbeat at positions dropped by either output are excluded. Head-specific counts can overlap and therefore must not be summed.

## Reported quantities

For every piece, report:

- changed positions and their fraction of all input positions;
- keep flips, plus offset, duration, and downbeat changes among positions kept by both outputs;
- the number and lengths of maximal runs of consecutive changed input positions;
- the span from the first through last changed position; and
- the already frozen TPF and MeanER deltas for context.

Report micro-aggregated changed-position fractions and per-piece summaries across all six pieces. No locality threshold, favorable subset, or categorical pass rule will be selected after seeing the results.

## Interpretation boundary

This analysis can show whether an available improved complete sequence is close to or far from the greedy sequence in aligned decoded rhythm streams. It cannot identify the first causally beneficial edit, establish a valid iterative revision path, or show that a ground-truth-blind method can locate the changed positions.
