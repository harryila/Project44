# Step 1: ASAP overlap check

Source: `https://github.com/TimFelixBeyer/asap-dataset/tree/8cba199e15931975542010a7ea2ff94a6fc9cbee` (commit pinned by upstream MIDI2ScoreTransformer/README.md), `metadata.csv` fetched directly.

## The 3 benchmark pieces

| Piece | Path in ASAP | Performances | Notes |
|---|---|---|---|
| Chopin Op.10 No.4 | `Chopin/Etudes_op_10/4/` | 22 (rows 559-580) | Has both score (`xml_score.musicxml`) and many performance MIDIs incl. multiple from MAESTRO |
| Chopin Op.25 No.11 | `Chopin/Etudes_op_25/11/` | 19 (rows 640-658) | Same — present with score and many performances |
| Liszt Transcendental No.4 (Mazeppa) | **NOT IN ASAP** | 0 | Liszt's other works are present (Annees de pelerinage, Ballade 2, Concert Etude S145, Gran Etudes de Paganini), but no Transcendental Etudes |

## Block 2 result on each piece (from `benchmark/IMPROVEMENT_RESULTS.md`)

| Piece | In ASAP | Time Sig | Measure error |
|---|---|---|---|
| Chopin Op.10 No.4 | YES | 4/4 (correct) | 6.8% |
| Chopin Op.25 No.11 | YES | 4/4 (correct) | 2.0% |
| Liszt Mazeppa | NO | 3/4 (wrong) | 92.8% |

## Hard-stop check

The plan's hard-stop trigger was: "if any of the 3 are in ASAP's training set AND Block 2 still failed on them at inference, the bottleneck is architectural."

What we actually see: pieces in ASAP succeed; the piece NOT in ASAP fails catastrophically. This is the data-shortage signature, not the architectural-limits signature. The pretrain plan is justified.

**Decision: PROCEED to Step 2.**

## Side note for Step 10 held-out picks

Whatever held-out pieces we pick must avoid the ASAP `Chopin/Etudes_op_*` and `Liszt/*` directories listed above. Specifically:
- Chopin Op.10 Nos.: 1, 4, 5, 6, 9, 12 (and possibly more) appear in ASAP. Need to grep again.
- Chopin Op.25 Nos.: 5, 6, 11, 12 (and possibly more) appear. Need to grep.
- Liszt: Annees, Ballade 2, Concert Etudes, Gran Etudes de Paganini, Transcendental Etudes (other than No.4) — many of these absent.
