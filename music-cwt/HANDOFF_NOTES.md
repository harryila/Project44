# Handoff notes for music-cwt (added at import, not written by the author)

**Source.** Private GitHub repository `Toitoine1/music-cwt`, commit `44db7f6` (2026-07-02), 77 files,
4.2 MB, copied as files into this folder. The author (GitHub `Toitoine1`) gave explicit permission for
this copy into the private joint repository. Upstream has no LICENSE file: treat the code as the
author's, for use inside this project, until terms are stated. The author's own documentation starts at
[`README.md`](README.md).

**What this is.** The writing half of the system: a compound-word transformer (about 16.6M parameters,
d_model 320, 10 layers, 8 attention heads, rotary positions, 3-stage output) that generates piano
scores as Humdrum `**kern` text, which converts to MusicXML for MuseScore. `musicML/` next to this
folder is the reading half (performance to score).

## What is current

- **`music_cwt/v3/`** is the current code: 18 heads per token (type, duration, pitch, ornament,
  articulation, tie, slur, phrase, voicing, beam, staff 1 to 4, dynamic, pedal, tempo, chord quality,
  chord voicing, voice 1 to 3, character), atomic chords, multi-staff, a Sustain anchor. Verified from
  `v3/tokenizers/multihead_tokenizer.py` and `v3/cwt_model.py`.
- **v3 has never been trained.** No v3 checkpoint exists anywhere (`v3/README.md` line 11). The v2
  checkpoints cannot be loaded into v3 (the new heads do not exist in v2; a compatibility layer is
  listed as "to write" in `v3/README.md`).
- **`music_cwt/v2/`** is the last trained version (14 heads): pre-training to epoch 50 and a
  fine-tuning run to epoch 15 are documented in `v2/docs/training_strategy.md`. Its checkpoints are not
  in this repository.
- `music_cwt/v1/` is the frozen first baseline (12 heads).

## What is not here, and who has it

Everything below lives on the author's machines; ask the author.

| Item | Why it matters |
|---|---|
| v2 checkpoints `ckpt_ep050_step356402.pt` (pre-train), `ckpt_ft_ep015_step359042.pt` (fine-tune, best val), harmonization checkpoints; v1 `ckpt_ft_ep050_step349566.pt` (about 60 MB each) | `generate.py` and `finetune_harmonize.py` need one |
| `pdmx_scrapper/mxl_to_kern.py` and its siblings (`download_pdmx.py`, `rename_pdmx_kern.py`, `diagnose_kern.py`) | the only MusicXML to kern converter; `v3/generate.py` imports it for MusicXML prompts, and the corpus cannot be rebuilt without it (`README.md` says it is not shipped) |
| The training corpus `kern_all/` (219,300 kern files: PDMX 213,369, KernScores 5,219, Chopin first editions 512, ASAP 200), `PDMX.csv`, `rename_mapping.tsv` | training and the held-out split |
| `music_cwt/data_analysis/` scripts and their reports | cited by `v3/README.md` and `todo.md`, not shipped |
| Training logs (`train_v2.log`, `finetune_v2.log`), generated samples (`v2/generated/`), round-trip outputs | the only written evaluation is the qualitative note in `v2/docs/training_strategy.md` |
| `CLAUDE.md` (project conventions the docs refer to), `article/idees.txt` | referenced, not shipped |
| The license of the two musescore.com prompt scores (`prompt/liszt-...mxl`, `prompt/national-anthem-...mxl`) | no license tag inside the files; the compositions are public domain, the engravings are third-party uploads |

The raw PDMX archive and its CSV are available from the `musicML` assets (`musicML/scripts/fetch_assets.sh data`);
they are the same Zenodo record the author used.

## What runs today, without the author's files

```bash
python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt   # torch + music21
cd music_cwt/v3
python postprocess.py ../prompt/009_1et2-1a-W-002.krn /tmp/recon.krn    # kern -> tokens -> kern round trip
python cwt_model.py                                                        # builds the 18-head model, forward pass
python kern_to_mxl.py                                                      # kern -> MusicXML (v2/kern_to_mxl.py also works)
```

Verified at import with Python 3.11, torch 2.11, music21 9.2: the round trip is idempotent (a second
pass is byte-identical), the model self-test runs. `generate.py`, training and harmonization need the
author's checkpoints or corpus. The `musicML/venv311` environment (`musicML/requirements.lock`)
already contains torch and music21 and can be reused.

## Edits made at import (everything else is byte-identical to commit 44db7f6)

1. `music_cwt/v2/auto_train.sh` and `v3/auto_train.sh` line 11: the author's cluster home path in a
   cron comment replaced by `$HOME/...`.
2. `music_cwt/v2/_find_val_files.py`: three corpus roots that were the author's Windows paths now come
   from the `CWT_KERN_SCRAPPING` environment variable (default `kern_scrapping`).
3. `music_cwt/v1/README.md`, `v2/README.md`, `v3/README.md`: the author's Windows Python path and
   upload paths replaced by placeholders (`<your Python 3.12 with torch + music21>`, `<local copy of taff>`).
4. `music_cwt/v3/kern_to_mxl.py` line 262: added the `sys.path` insert for `tokenizers/`, without which
   the module failed to import (bug).
5. Removed `music_cwt/prompt/nocturne_op9n1.krn` and `v1/prompt/nocturne_op9n1.krn`: 36-byte files
   holding a Python object repr, not scores.
6. Added this file and `requirements.txt`.

The READMEs still describe the author's training setup (an institutional GPU cluster reached through
an ssh gateway, `~/taff/...` layout, cron and tmux conventions). Those sections are his record of how
the runs were made; students cannot execute them and do not need to.

## Reading order

1. `README.md` (author's overview; its "start with" list and code paths are v2-centric)
2. `music_cwt/v3/README.md` and `music_cwt/v3/docs/tokenization_rules_reviewed.md` (the current
   tokenization; note the stale head counts below)
3. `music_cwt/v3/docs/plan_multivoix_sustain.md` and `atomisation_accords.md` (what v3 added and why)
4. `datasets.md` (corpus construction, sources, licenses, the 2-staff squeeze and its limits)
5. `todo.md` (the author's open list; partly stale)
6. `music_cwt/v2/docs/training_strategy.md` (the only training journal; the v3 copy is older)

Most of these are in French. `README.md`, `TOKENIZER_DOC.md` (12-head era) and `v1/README.md` are in English.

## Statements in the author's docs that the code contradicts

- `v3/docs/tokenization_rules_reviewed.md` says 16 or 17 heads; `v3/docs/archi_model.md` says 16 heads
  and 14 in several places; the code has 18 (`multihead_tokenizer.py` lines 171 to 172).
- `todo.md` still lists v2 pre-training as in progress; `v2/docs/training_strategy.md` records it
  reaching epoch 50, and `finetune_harmonize.py` names an epoch-15 fine-tune checkpoint.
- `v3/README.md` line 14 lists de-squeezing the converters as remaining; `todo.md` lines 46 to 47
  record it as done and tested on 3-staff Liszt.
- `v3/README.md` line 81 documents an open mismatch between the tokenizer (stamps H12 to H14 and H18
  on every token, including the Sustain anchor) and the loss-mask table (line 75). Resolve before
  training v3.
- `gen_and_convert.py` swaps staves only for exactly 2 parts; v3 can emit up to 4 staves, so the swap is
  skipped for 3 or 4 staff output.
- The root `README.md` example commands use placeholder file names (`checkpoints/ckpt.pt`,
  `../prompt/example.musicxml`) that do not exist.
- The two `auto_train.sh` and `watchdog.sh` files under v3 are byte-identical to the v2 ones and still
  drive the v2 training directory.
- The fine-tuning validation files were also seen in pre-training, so the reported validation loss is
  optimistic; a true held-out set is item one on `todo.md`.
