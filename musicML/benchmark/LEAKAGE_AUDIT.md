# PDMX ↔ Evaluation Leakage Audit

**Date:** 2026-05-30 · **Method:** metadata match + adversarial content-fingerprint
verification (transposition-invariant interval 5-grams, validated with positive
controls). **Verdict: HIGH leakage — content-confirmed.**

## Why this matters

If we ever pretrain/finetune on PDMX (Track C/D), any evaluation piece that also
exists in PDMX is **train/test contamination** — the held-out number for that
piece becomes meaningless. This audit establishes exactly which eval pieces leak,
so we can exclude them *before* any PDMX training.

**It does NOT affect the current Tier-1 baseline.** The released `MIDI2ScoreTF.ckpt`
was trained on ASAP only (never PDMX), so the ASAP held-out test split is
legitimately held-out for it. The leakage is a constraint on *our future* training,
not the baseline anchor.

## Result: 12 of 17 eval pieces leak; 5 are content-proven clean

| Piece | Set | In PDMX? | Evidence |
|---|---|---|---|
| Chopin Op.10 No.4 "Torrent" | benchmark | **LEAK** | content-verified copy (c# minor, 83 meas), in piano subset |
| Chopin Op.25 No.11 "Winter Wind" | benchmark | **LEAK** | 3 copies incl. full Op.25, in piano subset |
| Liszt Mazeppa | benchmark | **LEAK** | exact S.139/4 copy (= the GT we sourced), in piano subset |
| Bach Fugue BWV 846 | asap-test | **LEAK** | ~21 fugue copies (C major, 27 meas verified) |
| Beethoven Sonata Op.10 No.1 | asap-test | **LEAK** | full sonata, mvt1 embedded (520 meas verified) |
| Brahms Op.118 No.2 | asap-test | **LEAK** | 3 copies |
| Chopin Ballade No.1 Op.23 | asap-test | **LEAK** | ~10 copies (89% content containment) |
| Debussy "Reflets dans l'eau" | asap-test | **LEAK** | content-verified (Db major, 95 meas); in PDMX not subset |
| Liszt "Gondoliera" | asap-test | **LEAK** | hidden under alt title "Venezia e Napoli" (92% containment) |
| Rachmaninoff Prelude Op.23 No.4 | asap-test | **LEAK** | content-verified (D major, 77 meas) |
| Ravel "Ondine" | asap-test | **LEAK** | embedded in full "Gaspard de la Nuit" (78% containment) |
| Schubert Impromptu D.899 No.1 | asap-test | **LEAK** | content-verified (85% containment) |
| **Mozart Sonata K.332 mvt1** | asap-test | **clean** | content-proven (max 7.8% vs all PDMX Mozart) |
| **Haydn Sonata Hob.XVI:31 mvt1** | asap-test | **clean** | content-proven (9.7%); PDMX has XVI:34 not :31 |
| **Schumann Arabeske Op.18** | asap-test | **clean** | content-proven (2.3%); absent from PDMX |
| **Scriabin Étude Op.8 No.11** | asap-test | **clean** | content-proven (2.2%); PDMX has Op.8 No.12 only |
| **Prokofiev Toccata Op.11** | asap-test | **clean** | content-proven (absent from 20 PDMX Prokofiev rows) |

Positive-control thresholds: true leaks score **0.78–0.92** interval-5gram
containment; unrelated same-composer pieces score **<0.10**. Dedup threshold ~0.4.

## Required dedup before any PDMX training (do NOT skip)

Metadata blocklists are **insufficient** — they miss three real leak vectors found
here:
1. **Alternate-title copies** — Liszt "Gondoliera" leaks as "Venezia e Napoli".
2. **Embedded movements** — Ravel "Ondine" inside full "Gaspard"; Op.25 No.11
   inside "Douze Études op.25 (full)". A `No.11`/`No.1` keyword block misses these.
3. **Mislabeled metadata** — Debussy "Reflets" tagged "Images pour orchestre";
   Mazeppa S.137 mislabeled as "Paganini S.141".

**Build a content-based dedup pass** (validated recipe):
1. For each eval GT score, extract a **time-sorted** pitch sequence (`music21`
   flatten, sort by offset — **NOT** XML document order; doc-order gave false 3–9%
   even for known leaks).
2. Build **transposition-invariant interval 5-grams**.
3. Drop any PDMX row whose 5-gram containment with any eval piece **> 0.4**.
4. Run over composer-restricted PDMX piano rows (ideally all of PDMX).
5. Belt-and-suspenders: also drop the explicit `mxl` paths listed in the audit
   output, and **block whole parent works** for the embedded-movement cases
   (full Op.25, full Gaspard, the entire BWV 846 family).
6. Apply to **both** `PDMX.csv` and `pdmx_piano_subset.csv`, keyed by `mxl` path.

Until this runs, treat held-out numbers for the 12 leaking pieces as confounded.

## Content-dedup EXECUTED (2026-05-30)

The recipe above is implemented in [scripts/content_dedup.py](../scripts/content_dedup.py)
and has been run. Validation (positive/negative controls): a known leak scored
containment **0.961**, an unrelated piece **0.001** — the 0.4 threshold sits in a
clean gap.

**Result on the piano subset** (1,134 eval-composer candidates fingerprinted, 1
parse-fail, ~7 min): **23 PDMX rows flagged as leaks** (containment > 0.4), only **1
borderline** (0.2–0.4). Blocklist: [data/pdmx_eval_leak_blocklist.csv](../data/pdmx_eval_leak_blocklist.csv).
Deduped training pool: [data/pdmx_piano_subset.deduped.csv](../data/pdmx_piano_subset.deduped.csv)
(181,693 → **181,670**).

Leaks per eval piece: Bach BWV846 fugue ×6, Chopin Ballade No.1 ×7, Brahms Op.118
No.2 ×2, Liszt "Gondoliera" ×1 (caught under its alt-title "Venezia e Napoli",
0.924), Rachmaninoff Op.23/4 ×1, Schubert Impromptu ×1, + benchmark Chopin Op.10
No.4 ×3, Op.25 No.11 ×1, Mazeppa ×1.

**Notes:**
- The content method confirmed the metadata audit *and* the alt-title/embedded
  cases, and correctly did **not** flag Debussy "Reflets" or Ravel "Ondine" — those
  leak only into *full* PDMX, not the piano subset (the intended training pool).
- One discrepancy with the metadata audit: it flagged a Beethoven Op.10 No.1 copy in
  the subset, but content containment was < 0.4 (likely a different edition/movement
  or metadata false-positive). The content result is authoritative.
- If training ever expands to **full PDMX** (not the subset), re-run
  `content_dedup.py --subset-csv <full PDMX csv>` to catch the extra leaks.

## Side effect: Mazeppa GT sourced

`benchmark/liszt_transcendental/gt_score.musicxml` was extracted from PDMX
(idx 132858, the genuine Transcendental Étude S.139/4): D minor, 7208 note members
(vs the GT MIDI's 7333, ~2%), pitch-class histogram Pearson **0.9986** with the
benchmark GT MIDI, opening cadenza matches note-for-note. This is now the held-out
GT for the Tier-2 end-to-end eval. **Note:** this exact file is one of the leaked
PDMX copies, so it must be excluded from any PDMX training (idx 132858 + 86971).
