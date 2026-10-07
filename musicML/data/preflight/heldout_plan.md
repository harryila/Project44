# Held-out pieces for Step 10

## ASAP composers to AVOID
Bach, Balakirev, Beethoven, Brahms, Chopin, Debussy, Glinka, Haydn, Liszt,
Mozart, Prokofiev, Rachmaninoff, Ravel, Schubert, Schumann, Scriabin.

## ASAP-included Liszt Transcendental Etudes
At least Nos. 1, 10 are in ASAP. Need to verify which others. Mazeppa (No.4)
is NOT in ASAP (which is why we used it as our hard benchmark).

## Candidate held-out pieces (not in ASAP)

Difficulty-matched picks (high virtuosity, dense polyphony) from composers
NOT represented in ASAP:

1. **Mendelssohn — Etude Op.104a No.3 (in F major)** or another
   Etude/Songs-without-Words. Mendelssohn is NOT in ASAP.
2. **Saint-Saëns — Etude Op.111 No.4 (Etudes Op.111 — Toccata)** or another
   Saint-Saëns etude. Saint-Saëns is NOT in ASAP.
3. **Albeniz — Iberia (Triana, Lavapies)** — heavy texture; Albeniz is NOT
   in ASAP.

Alternatives if above are unavailable:
- Granados — Goyescas
- Medtner — Forgotten Melodies / Skazki
- Czerny — School of Velocity Op.299

## Verification before use

For each held-out piece, before computing metrics:
1. `grep` against `data/preflight/asap_metadata.csv` to confirm not in ASAP.
2. `grep` against `data/pdmx_piano_subset.csv` (composer/title) to confirm
   not in PDMX subset.
3. Source the MusicXML score (public domain via IMSLP / MuseScore.com).
4. Source the GT MIDI (deadpan-render the score, or use MAESTRO if available).

## Pragmatic note

Curating 3 difficulty-matched held-out pieces with publicly available scores
and MIDI is several hours of manual work. The primary post-pretrain success
measure is on the 3 MAESTRO benchmark pieces (Op.10 No.4, Op.25 No.11,
Mazeppa). Held-out is the "didn't break" sanity check; we will return to it
after Stage A and Stage B numbers are in.
