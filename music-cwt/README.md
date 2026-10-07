# Music CWT

Compound-word transformer experiments for symbolic piano generation from
Humdrum `**kern` / MusicXML scores.

This repository contains the Music CWT model code, tokenizers, generation
scripts, and project documentation. It intentionally does not include training
datasets, generated corpora, logs, model checkpoints, or unrelated conversion
experiments.

## Core design principle: lossless round-trip

The project's guiding rule is **100 % notation fidelity**. The full pipeline
`MusicXML → **kern → compound tokens → **kern → MusicXML` must be lossless for
every piece of musical information (up to equivalent spellings, e.g.
enharmonics). **No silent heuristic** is allowed to "fix" an ambiguity: if
information is lost, the exact cause is located and an exact fix is implemented
rather than a probabilistic shortcut. This is why the tokenizers and converters
are so detailed — the multi-head token representation is designed so that
`**kern → tokens → **kern` is character-exact and idempotent.

## Version status

- **v1** — frozen baseline, trained (pre-training + Humdrum fine-tune). 12 heads.
- **v2** — trained (pre-training + fine-tuning). 14 heads, ~16.2 M params. Main
  reference implementation.
- **v3** — **coded and round-trip validated, not yet trained.** 18 heads,
  ~16.6 M params. Adds atomic chords (H15/H16), intra-stave multi-voice (H17),
  an explicit `Sustain` anchor token (replaces the `SEP` delimiter, −23 % tokens),
  and a character/expression head (H18).

## Repository Layout

```text
music_cwt/
  v1/                 earlier baseline (12 heads, trained)
  v2/                 main 14-head CWT implementation and harmonization scripts (trained)
  v3/                 18-head work: atomic chords, multi-voice, Sustain, character (not yet trained)
  prompt/             small prompt examples used for generation tests
  projet_a_pt_etre_reprendre/
                      research notes and prototype analysis scripts
datasets.md           dataset inventory and provenance notes
todo.md               project roadmap / open research notes
```

## What Is Not Included

The following are excluded on purpose:

- model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `checkpoints/`)
- scraped datasets and corpora (`kern_data/`, `melody_data/`, large PDMX/KernScores folders)
- generated samples (`generated/`)
- training logs (`logs/`, `*.log`)
- local environments, caches, and temporary files
- auxiliary conversion experiments outside `music_cwt/`
- the scraping / dataset tooling under `pdmx_scrapper/`

> **Note on MusicXML → kern:** the project uses a single exhaustive converter,
> `pdmx_scrapper/mxl_to_kern.py`, which lives in the excluded scraping tooling and
> is **not shipped here**. This repository therefore only includes the reverse
> direction, `music_cwt/*/kern_to_mxl.py` (kern → MusicXML). To feed a MusicXML
> prompt you need that converter (or an equivalent `**kern` file) yourself.

## Main Code Paths

- `music_cwt/v2/tokenizers/multihead_tokenizer.py` converts `**kern` files into
  compound tokens.
- `music_cwt/v2/cwt_model.py` defines the transformer architecture.
- `music_cwt/v2/train_cwt.py` trains the base model.
- `music_cwt/v2/finetune_humdrum.py` fine-tunes on curated Humdrum files.
- `music_cwt/v2/finetune_harmonize.py` fine-tunes the melody-to-piano
  harmonization objective.
- `music_cwt/v2/generate.py` generates `**kern` output from a checkpoint.
- `music_cwt/v2/gen_and_convert.py` generates and converts output to MusicXML.
- `music_cwt/v2/gen_harmonize_and_convert.py` generates a harmonization from a
  melody prompt and converts it to MusicXML.
- `music_cwt/v2/postprocess.py` converts decoded compound tokens back to
  Humdrum `**kern`.
- `music_cwt/v2/kern_to_mxl.py` provides an alternate `**kern` to MusicXML
  conversion pipeline.

The v3 folder mirrors the v2 structure and extends it to 18 heads (atomic
chords, multi-voice, `Sustain`, character). See `music_cwt/v3/README.md`.

## Documentation

Start with:

- `music_cwt/v2/README.md`
- `music_cwt/v2/docs/tokenization_rules_reviewed.md`
- `music_cwt/v2/docs/archi_model.md`
- `music_cwt/v2/docs/training_strategy.md`
- `music_cwt/v3/docs/tokenization_rules_reviewed.md`
- `datasets.md`

## Minimal Setup

The exact training environment evolved during the project, but the core Python
dependencies are:

```bash
pip install torch music21 numpy tqdm
```

Some conversion helpers additionally require external tools such as MuseScore,
Audiveris, or HOMR depending on the workflow.

## Example Commands

Generate from an existing checkpoint:

```bash
cd music_cwt/v2
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt.pt \
  --output generated/sample.krn \
  --prompt-krn ../prompt/example.musicxml \
  --prompt-measures 6 \
  --max-tokens 1200 \
  --temperature 0.8 \
  --top-k 30
```

Generate a harmonization from a melody prompt:

```bash
cd music_cwt/v2
python gen_harmonize_and_convert.py \
  --checkpoint checkpoints_harmonize/ckpt_harm.pt \
  --prompt-krn ../prompt/melody.musicxml \
  --output generated/harmonized.krn \
  --prompt-measures 8 \
  --max-tokens 1200
```

You will need to provide your own checkpoints and corpora.
