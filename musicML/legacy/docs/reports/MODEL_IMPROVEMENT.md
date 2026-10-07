# Model Improvement: Where Training Data Fits In

This doc walks through what's actually being trained, where the data comes from, and what the bottleneck is for making the pipeline better.

---

## The Two Blocks

Our pipeline has two machine-learning blocks. They learn different things and need different training data.

```
Audio (.wav)  ──►  [Block 1]  ──►  MIDI  ──►  [Block 2]  ──►  MusicXML  ──►  PDF
                  Audio→MIDI                  MIDI→Score              (rendered by
                  transcriber                 model                    MuseScore)

Block 1 examples: Basic Pitch, MT3, Transkun, hFT-Transformer
Block 2 example:  MIDI2ScoreTransformer
```

Important: **MusicXML and the printed sheet score are the same thing**. MusicXML is just the machine-readable format; the PDF is the visual rendering of that XML. So "MIDI → MusicXML" and "MIDI → sheet music" are the same task.

Same for **audio and MIDI** — MIDI is essentially a structured representation of "what notes were played, when, how loud." It's not identical to audio (audio has timbre, room, dynamics nuance), but for our purposes the audio→MIDI step is recovering the performance.

So conceptually:

| Concept             | Format pair                  |
|---------------------|------------------------------|
| Performance         | audio ≈ MIDI                 |
| Notation            | MusicXML ≈ sheet score (PDF) |

Block 1 maps performance → performance representation.
Block 2 maps performance representation → notation.

---

## What Each Block Is Trained On

**Block 1 (Audio → MIDI)** is trained on `(audio, MIDI)` pairs.
- Input: a `.wav` of someone playing piano.
- Target: the MIDI file describing what notes were played.
- Dataset: **MAESTRO** (~200 hours of piano performance with aligned MIDI).

**Block 2 (MIDI → Score)** is trained on `(MIDI, MusicXML)` pairs.
- Input: a performance MIDI.
- Target: the MusicXML score the performer was reading.
- Dataset: **ASAP** (~200 piano pieces with both performance MIDI and the engraved score).

So **no block is trained directly on `(audio, XML)` pairs**. Block 1 never sees MusicXML. Block 2 never sees audio. They're chained at inference time, but they were trained independently on different supervision.

That's an important point: when you ask "is the model trained on audio paired with XML?" — the answer is no. It's two separate models, each with its own pairing.

---

## MAESTRO vs ASAP

| Feature                  | MAESTRO                                            | ASAP                                                                  |
|--------------------------|----------------------------------------------------|-----------------------------------------------------------------------|
| Size                     | 1276 performances, ~200 hours                       | 222 pieces, ~1067 performances                                        |
| Audio                    | Yes (high-quality WAV)                              | Audio for some, not for all                                            |
| Performance MIDI         | Yes (recorded from Disklavier piano)                | Yes (mostly sourced from MAESTRO + others)                            |
| **MusicXML score**       | **No**                                              | **Yes** (this is the key difference)                                   |
| Alignment to score       | Not provided                                        | Yes — MIDI notes are aligned to score positions, with downbeat labels |
| Use case                 | Train Block 1 (audio → MIDI)                        | Train Block 2 (MIDI → score)                                          |
| What it took to build    | Recording sessions on a Disklavier (automated)      | Manual alignment work — humans matched performance MIDI to engraved scores |

**Why MAESTRO is used for Block 1:** it has the `(audio, MIDI)` pairs the audio transcriber needs. No MusicXML required for that task.

**Why ASAP is used for Block 2:** it has the `(MIDI, MusicXML)` pairs the score model needs. ASAP exists *because* MAESTRO doesn't have scores.

**Why the datasets aren't just "combined":**

- They aren't combined because they answer different questions and supply different supervision. You can't train Block 1 on ASAP because most of ASAP doesn't have audio. You can't train Block 2 on MAESTRO because MAESTRO has no MusicXML.
- ASAP actually *already pulls some MIDI from MAESTRO* — the overlap exists. The reason ASAP is small isn't that they ignored MAESTRO; it's that adding the MusicXML side requires finding the published score and aligning it manually. That step is the bottleneck.

In short: **MAESTRO has the easy half (audio↔MIDI). ASAP has the hard half (MIDI↔score). The hard half is small because alignment is human labor.**

---

## Where the Bottleneck Is

Our benchmarking on the 3 MAESTRO pieces (Chopin Op.10 No.4, Chopin Op.25 No.11, Liszt Mazeppa) showed:

- Block 1 with hFT-Transformer is solid. Note counts and timing line up reasonably with ground truth.
- Block 2 (MIDI2ScoreTransformer) is the weak link. Even when fed *ground-truth* MIDI from MAESTRO (i.e., perfect Block 1 input), it predicts the wrong time signature and drifts on measure counts for the hard pieces.

This isn't surprising. Block 2 was trained on ~200 ASAP pieces. That's not enough to generalize to the density and irregularity of late-Romantic etudes and Liszt transcendentals.

So the model that needs improving is **Block 2 (MIDI2ScoreTransformer)**, and the way to improve it is: **more `(MIDI, MusicXML)` pairs**.

---

## Ways to Improve Block 2

### 1. Augment the training set (most impactful, most labor)

Take MAESTRO's 1276 performance MIDIs and pair each one with a published MusicXML score. The scores exist on IMSLP, MuseScore.com, etc. The challenge is *alignment*: matching each performance note to a score note, accounting for repeats, ornaments, performer-added notes, missed notes.

If you got even 500 well-aligned pairs added on top of ASAP's 200, that's a ~3.5x increase in training data. For a model this size, that's substantial.

This is essentially what the ASAP team did, but at a larger scale.

### 2. Self-training / bootstrapping (medium impact, medium labor)

Run our current MIDI2ScoreTransformer on 500 MAESTRO performance MIDIs. Have a human go through the resulting MusicXML and fix the obvious errors (wrong time signature, wrong key, misplaced barlines, hand-splitting mistakes). Add the corrected `(MIDI, fixed MusicXML)` pairs to the training set. Retrain.

This shifts the work from "align scores from scratch" to "correct model output", which is faster.

Risk: errors can self-reinforce if you don't correct carefully — the model will learn its own mistakes.

### 3. Auxiliary supervision from MAESTRO (low impact, low labor)

MAESTRO MIDIs include some metadata (composer, piece title, recording year) and the MIDI itself encodes pedal info. None of this is a full score, but you could:

- Use known piece titles to look up the canonical time signature and key, and add that as a weak supervision signal.
- Use pedal events as proxy boundaries for phrasing.

This wouldn't fix the time signature bug directly, but it would give the model more context than just "raw note sequence."

### 4. Targeted fine-tuning on hard cases (most efficient short-term)

Don't retrain from scratch. Take the existing MIDI2ScoreTransformer and fine-tune it on a small set of hand-curated `(MIDI, MusicXML)` pairs that specifically target the failure modes — Chopin etudes, Liszt, anything with dense polyphony or unusual meter. Even 20-50 well-chosen pairs could move the needle on time signature accuracy without needing 1000+ new examples.

This is probably the highest ROI option.

---

## Summary

- The pipeline has two trained blocks: audio→MIDI and MIDI→score.
- They're trained on different datasets because each task needs different pairings: MAESTRO for `(audio, MIDI)`, ASAP for `(MIDI, MusicXML)`.
- You can't trivially combine them because MAESTRO has no MusicXML and ASAP has limited audio. Combining them really means *creating new MusicXML for MAESTRO performances*, which is the same manual alignment work that limited ASAP's size.
- The bottleneck for our pipeline is Block 2, because ASAP's 200 pairs aren't enough to handle complex repertoire.
- The fix is more `(MIDI, MusicXML)` data — either by full augmentation, bootstrapping, or targeted fine-tuning. Targeted fine-tuning is the most realistic short-term path.
