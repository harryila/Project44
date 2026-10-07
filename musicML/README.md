# musicML: piano performance to engraved score

Turn a piano **performance** (audio or MIDI) into a correct **engraved score** (MusicXML, PDF), as
accurately as possible. This is the transcription half of a two-part system whose other half generates
new score in a composer's style (the `music-cwt` project next to this one).

```
AUDIO --[ Block 1: audio to MIDI ]--> performance MIDI --[ Block 2: MIDI to score ]--> MusicXML --> PDF
          hFT-Transformer (best), MT3, Transkun, Basic Pitch      MIDI2ScoreTransformer (RoFormer)     MuseScore
```

Block 1 is solved enough (about 96 to 97 percent of notes) and was deliberately left alone. Block 2 is
where the research lives: the sound does not contain the notation, so beat grid, bar lines, tuplets,
voices and hands must be inferred from expressive human timing.

**Start here:** this file, then [AGENTS.md](AGENTS.md) (rules for people and coding agents),
[RUNS.md](RUNS.md) (every training run and where its evidence is), [DATA.md](DATA.md) (every dataset,
its provenance and its gaps), [MODELS.md](MODELS.md) (every checkpoint), and
[docs/reports/LAB_REPORT.md](docs/reports/LAB_REPORT.md) (the most recent full technical report).

## Status as of October 2026

One table, each number tied to the file that backs it. MeanER is the mean error rate of the MUSTER
score-comparison metric; lower is better.

| What | Number | Backing file |
|---|---|---|
| Reproduction of the published model on the 59-performance ASAP test set | **11.18** vs the paper's 11.30 | `benchmark/tier1_baseline.json` |
| Released model, 14-piece corpus mean (threshold 0.50) | **10.77** | `benchmark/padsweep_released.json` |
| Best own model `ssl_tuplet20` (from scratch, masked SSL on classical scores, tuplet loss weight 2.0) | **11.87** (a later re-evaluation of the same checkpoint gives 11.79) | `benchmark/ssl_tuplet20_last.json`, `benchmark/rerank/ssl_tuplet20__ssl_tuplet20_last.json` |
| Own model `ssl_classical_clean` epoch 13 (before the tuplet weight) | 12.69 | `benchmark/padsweep_sslcc.json` |
| Clean, in-distribution Chopin, full audio to score | about 2.75 | `benchmark/chopin_op10`, `docs/reports/SONGSCRIPTION_PARITY.md` |
| Rank diagnostic at 1,184 teacher-forced tuplet positions: correct bucket is the top guess | **36.66 percent** | `benchmark/topk_offset_diag.json` |
| Same positions: correct bucket within the top 15 | **98.48 percent** | same file |
| Best-of-N: in all six sampled tuplet-bearing pools a better candidate exists (guarded oracle) | 9.54 percent lower tuplet error, but every ground-truth-blind selector failed, including the frozen confirmation (CONFIRMED=false) | `research/protocols/`, `benchmark/confirmatory_v10_1_results.json`, `benchmark/oracle_prefix_v10_1.json` |

What the numbers say:

1. The evaluation harness is trustworthy (11.18 vs 11.30).
2. The remaining error is rhythm, specifically tuplets, not pitch and not the audio transcriber.
   Feeding perfect MIDI into Block 2 fails the same way as real audio.
3. A split of the metric into tuplet and non-tuplet onsets shows tuplet error is about 2 to 10 times
   worse on most pieces, hidden by averaging (`scripts/muster_tuplet_decompose.py`,
   `benchmark/muster_tuplet_decomposed.json`).
4. Data strategy, not architecture, closed most of the gap to the released model: both models are the
   same 32.6M-parameter RoFormer, and masked self-supervised pretraining on genre-matched classical
   scores took a from-scratch model from about 2x worse to within 1.1 MeanER of the released one.
   A bigger corpus made it worse (tuplet over-production).
5. The model usually ranks the right rhythm highly but greedy decoding does not pick it, and sampling
   more candidates does produce better ones, but no selector that cannot see the answer has been able
   to pick them. That support-selection gap is the open research problem.
6. Things that did not work: fine-tuning the released checkpoint (always drifted worse), beat
   conditioning, uniform triplet boosting, pitch snapping, synthetic rendered pairs, a bigger classical
   corpus, and ten selector designs. Each is documented with numbers in `docs/reports/` and `RUNS.md`.

Known gaps you should know on day one (details in `DATA.md` and `RUNS.md`): the classical_clean
training renders and their cache never left the GPU machine (the manifest is here; rebuilding needs
PDMX and is a new dataset version); ten runs have evaluation results but no surviving checkpoint;
MAESTRO-derived corpora are gone except the three benchmark pieces.

## Quickstart

```bash
git clone <this repository>            # the project is the musicML/ folder of the joint repository
cd musicML
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -r requirements.lock       # exact Mac/CPU environment; works on CUDA Linux too
brew install zstd gh musescore lilypond   # macOS; on Linux use your package manager (MuseScore needs QT_QPA_PLATFORM=offscreen)
gh auth login                          # an account with access to the asset release
scripts/fetch_assets.sh quickstart     # 1.6 GB: checkpoints, ASAP, caches, benchmark pieces
python musicml_paths.py                # prints every resolved location and whether the checkpoints are present
```

Transcribe a recording (your own, or the MAESTRO Chopin etude the quickstart installs):

```bash
python transcribe.py "samples/Park Blvd 2.m4a" -t hft -b transformer
python transcribe.py benchmark/chopin_op10/audio/Op10_No4_CsharpMinor.wav -t hft -b transformer
# output: outputs/transformer/<name>.pdf and .musicxml; use --midi-input file.mid to start from MIDI
```

Evaluate (CPU on a Mac; both scripts `cd` into `MIDI2ScoreTransformer/`, so `--ckpt` is relative to it):

```bash
python benchmark/eval_tier1_asap.py --out outputs/tier1_smoke.json --limit 3          # paper-comparable tier-1 MUSTER
python benchmark/eval_padsweep.py --ckpt checkpoints/MIDI2ScoreTF.ckpt --device cpu \
       --limit-per 1 --out outputs/padsweep_smoke.json                                 # 14-piece corpus sweep
```

Use `-t hft`: it is the best transcriber and the one the lock file guarantees everywhere (the default
`-t basic-pitch` resolves to a CoreML model under this lock and runs only on macOS).
Do not install `hFT-Transformer/requirements.txt` (torch 1.10) or `mt3/setup.py` into this
environment; MT3 is optional and has its own lock (`requirements-mt3.lock`, separate venv).
Never run Block 2 on Apple's MPS device: it silently corrupts the pad logits. `musicml_paths.device()`
picks cuda or cpu.

## Repository map

| Path | Status | What it is |
|---|---|---|
| `transcribe.py` | CURRENT | the one-command pipeline (audio or MIDI to PDF) |
| `musicml_paths.py`, `.env.example` | CURRENT | every machine-specific location, overridable by environment variables |
| `MIDI2ScoreTransformer/` | CURRENT | Block 2 model, our training stack (`train.py`, `pdmx_dataset.py`, `pathmap.py`); upstream shipped inference only |
| `hFT-Transformer/`, `mt3/` | CURRENT (vendored) | Block 1 transcribers, upstream code and licenses |
| `benchmark/` | CURRENT | evaluation harness, test pieces, every result JSON; see `benchmark/README.md` |
| `scripts/` | mixed | data builders, the two portable training launchers, diagnostics, the frozen research chain; see `scripts/README.md` |
| `research/` | CURRENT | best-of-N protocols and audits, the provenance ledger with hash pins, figure builders |
| `data/` | CURRENT | manifests, priors, leakage blocklist; the packs arrive with `fetch_assets.sh data` |
| `docs/reports/` | CURRENT | topic reports: `LAB_REPORT.md` (newest), AR build, data scaling, beat conditioning, pad threshold, hard-tail diagnostics |
| `docs/related_work/` | reference | related-work scan |
| `LAB_REPORT_COMPLETE.md` | HISTORICAL (June 2026) | the long-form report of the first phase; some conclusions superseded, see the top note |
| `legacy/` | HISTORICAL | archived experiments, April 2026 pretraining, old reports and runbooks, GPU box logs; see `legacy/README.md` |
| `checkpoints_backup/` | records only | training configs and curves of runs; the checkpoints themselves are assets |
| `prez/figs/` | reference | result figures |
| `assets/` | CURRENT | asset list, checksums, manifest, release pointer |
| `outputs/`, `logs/` | generated | gitignored |

## Data and assets

Everything larger than a few MB is a release asset, in four tiers (`assets/MANIFEST.md`). Git holds
code, docs, manifests, result JSONs, training curves and configs, so a clone is about 130 MB.
`DATA.md` describes each dataset's source, license, producing script, consumers and gaps.

The manifests record absolute paths from the machines the corpora were built on, and those strings
are the cache keys. Do not rewrite them: `MIDI2ScoreTransformer/midi2scoretransformer/pathmap.py`
maps them to this machine when files are opened and hashes the recorded string for the cache.
`python -c "import sys; sys.path.insert(0,'MIDI2ScoreTransformer/midi2scoretransformer'); import pathmap; print(pathmap.PATH_MAP)"`
shows the active mapping.

## Research directions

Tagged tested, partly tested, or open. The decoding and data directions attack the measured problem.

- **A selector that works** (partly tested): good candidates exist in sampled pools; hand-built
  selectors fail. Open: learned selectors, stronger rhythm verifiers, exact per-bar search.
- **Decoders that can revise** (open): the current decoder writes left to right with no refine step;
  a draft-and-refine decoder needs new decoding code and probably retraining.
- **Music theory as constraints or rewards** (barely tested): bar-tiling rules at decode time; scoring
  on countable musical features; theory-aware losses (the tuplet weight was the first one and gave the
  best model).
- **Data** (tested lesson, open work): genre-matched classical data was the lever, more of it was
  not. Open: which pieces, what tuplet balance, composer-specific collections, optical music
  recognition of scanned scores.
- **Reinforcement learning with checkable rewards over a text score format** (open), hierarchical
  generation (open), graph representations (open but representation has not been the bottleneck),
  joint audio-and-notation models (open), modelling the performer's expressive residual (open).
- **Robust decoding for noisy amateur recordings** (open): the product-facing lever.

## Rules

- Stage files by path. Never `git add -A`. Never commit checkpoints, data packs, caches or outputs.
- Frozen research scripts (listed in `research/supplement/PROVENANCE.md`) are not edited; write a
  wrapper or a new script.
- Run evaluations on CPU on a Mac. Pick checkpoints by MUSTER, not by validation loss (they disagree).
- The MAESTRO and ASAP material is CC BY-NC-SA: research use inside this private repository only.

## Key references

- Beyer and Dai, *End-to-end Piano Performance-MIDI to Score Conversion with Transformers*, ISMIR 2024, arXiv:2410.00210, code github.com/TimFelixBeyer/MIDI2ScoreTransformer
- hFT-Transformer (Sony, ISMIR 2023); MT3 (Google Magenta, ISMIR 2021); Basic Pitch (Spotify); Transkun
- ASAP dataset (Foscarin et al.), PDMX (Zenodo 15571083), MAESTRO v3
