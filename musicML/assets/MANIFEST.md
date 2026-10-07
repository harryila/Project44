# Release assets: what is not in git and how to get it

Large or binary files live as release assets, not in git. `scripts/fetch_assets.sh` downloads a tier,
verifies every file against `assets/SHA256SUMS`, and extracts it into this project's root so the layout
is exactly the one the code expects. `assets/ASSETS.csv` is the machine-readable list (name, tier, size,
sha256, where it extracts, what it contains). `assets/RELEASE` names the GitHub repository and release
tag that host the files; `gh auth login` with an account that can read that repository is required.

```
scripts/fetch_assets.sh quickstart     # 1.6 GB: run transcribe.py and the evaluations
scripts/fetch_assets.sh data           # 4.1 GB: training packs, PDMX, gpu-branch corpora
scripts/fetch_assets.sh archive        # 22 GB: candidate pools, outputs, all 56 non-live checkpoints
scripts/fetch_assets.sh all
```

Downloads are kept in `assets/_downloads/` (gitignored) so a re-run does not fetch again. Existing
files are never overwritten unless `--force` is given. Use `--from-dir <dir>` to install from files
you already have.

## Tiers

| Tier | Assets | Size | What it gives you |
|---|---:|---:|---|
| quickstart | 8 | 1.6 GB | the three live checkpoints, hFT and MT3 weights, the pinned ASAP dataset with its generated chunk files and tokenization cache, the three MAESTRO benchmark pieces |
| data | 11 | 4.1 GB | pairs_deduped_full and its cache, the smoke packs, the April 2026 pack, the PDMX subset CSVs, the raw PDMX metadata and MusicXML archive, the six gpu-branch corpora |
| archive | 2 | 0.2 GB | the best-of-N candidate pools and selector work directories, development outputs |
| archive-checkpoints | 56 | 22.1 GB | every other unique checkpoint, superseded epochs and ruled-out runs alike, full size with optimizer state |

## Where things land

| Asset | Extracts to |
|---|---|
| `MIDI2ScoreTF.ckpt` | `MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt` (released upstream model) |
| `ssl_tuplet20_last.ckpt` | `MIDI2ScoreTransformer/checkpoints/ssl_tuplet20/last.ckpt` (best own model, corpus MeanER 11.87) |
| `ssl_classical_clean_epoch13.ckpt` | `MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt` |
| `hft-checkpoint.tar.zst` | `hFT-Transformer/checkpoint/` |
| `mt3-checkpoints.tar.zst` | `mt3/checkpoints/` |
| `asap-dataset-8cba199e-with-chunks.tar.zst` | `MIDI2ScoreTransformer/data/asap-dataset/` (TimFelixBeyer/asap-dataset at commit 8cba199e, nested `.git` included, plus 967 `*_chunks.json`) |
| `asap-cache.tar.zst` | `MIDI2ScoreTransformer/data/cache/` (997 pickles) |
| `benchmark-pieces.tar.zst` | `benchmark/<piece>/audio/*.wav`, `benchmark/<piece>/midi/*.midi` |
| `data-*.tar.zst`, `legacy-data-*.tar.zst` | `data/...`, `legacy/data/...` |
| `pdmx-subset-csvs.tar.zst` | `data/pdmx_piano_subset.csv`, `data/pdmx_piano_subset.deduped.csv` |
| `PDMX.csv.zst`, `PDMX_mxl.tar.gz` | `${MUSICML_DATASETS:-~/datasets}/pdmx/` (outside the project, like the original setup) |
| `gpu-corpora.tar.zst` | `data/pairs_pdmx`, `pairs_classical`, `pairs_kern_rubato`, `pairs_scraped`, `pairs_broad`, `pairs_kern`, plus 12 manifests and `data/preflight` |
| `benchmark-work-dirs.tar.zst` | `benchmark/*_work*/` |
| `outputs.tar.zst` | `outputs/` |
| `ckpt-archive/*.ckpt` | the original path of each checkpoint (see `MODELS.md`) |

## Provenance and licenses

- The three MAESTRO performances in `benchmark-pieces` and the ASAP dataset are CC BY-NC-SA 4.0
  (research use, attribution required, no commercial use). Keep them inside the private repository.
- PDMX (Zenodo record 15571083) is distributed under its own terms; the piano subset used here kept
  only scores whose license column is `publicdomain` or `cc-zero`. The raw archive is included for
  convenience and is identical to the upstream download (sha256 in `SHA256SUMS`).
- `pairs_kern` and `pairs_scraped` inside the gpu-corpora asset were built from online kern and
  MusicXML sources whose terms were not recorded; treat them as internal research data.
- The checkpoints are this project's own except `MIDI2ScoreTF.ckpt` (Beyer and Dai, ISMIR 2024,
  released on GitHub), hFT (Sony, ISMIR 2023 release) and MT3 (Google Magenta, ISMIR 2021).

## Checking a download by hand

```
shasum -a 256 -c assets/SHA256SUMS --ignore-missing      # from the directory holding the files
```

Archive sizes over 2 GB cannot be hosted as a single GitHub release asset; none of the current assets
exceeds it (largest: `PDMX_mxl.tar.gz`, 1.89 GB). If a future asset does, split it with `split -b 1900m`
and list the parts in `ASSETS.csv`.
