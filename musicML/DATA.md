# DATA.md: every dataset and derived pack in this export

This file describes each dataset the project uses and each pack derived from it: where it came from,
what license it carries, which script produced it, which scripts or recipes consume it, which release
asset ships it and where that asset extracts, and how the recorded paths inside its manifest resolve
on a new machine. It is written for people (and coding agents) who have never seen the project.

Conventions used below:

- "project root" is the directory that contains this file and `musicml_paths.py`.
- `<datasets root>` is `${MUSICML_DATASETS:-~/datasets}`; PDMX lives at `<datasets root>/pdmx`
  (override with `MUSICML_PDMX_ROOT`). See `.env.example`.
- "asset" means a file in the GitHub release named in `assets/RELEASE`; `assets/ASSETS.csv` lists
  every asset with its tier, size, sha256, extraction path and contents, and `scripts/fetch_assets.sh
  <tier>` downloads, verifies and extracts a tier (`quickstart`, `data`, `archive`, `all`).
- "manifest" means a CSV with one row per training pair. The standard columns are
  `id, src_mxl, midi, mxl, chunks, cache, n_notes, n_measures, n_in_tokens, n_out_tokens`.
  Training reads only `midi`, `chunks`, `n_in_tokens` (and `tuplet_rate` when present); `src_mxl`,
  `mxl` and `cache` are provenance.
- The two build machines were a rented GPU box (project at `/root/Music-ML-BayenLab`, datasets at
  `/root/datasets`) and a macOS laptop. Manifests record paths from whichever machine built them.

Counts below were checked against the files themselves (the export, the release assets, or the
original repository where the export holds only a manifest). Where a fact could not be established
it is marked as not recorded.

## 0. Quick index

| Pack | What it holds | Shipped as (tier) | Extracts to | State |
|---|---|---|---|---|
| ASAP clone + chunks | 235 scores, 1,067 performance MIDIs, 967 `_chunks.json` | `asap-dataset-8cba199e-with-chunks.tar.zst` (quickstart) | `MIDI2ScoreTransformer/data/asap-dataset/` | complete |
| ASAP tokenizer cache | 997 pickles | `asap-cache.tar.zst` (quickstart) | `MIDI2ScoreTransformer/data/cache/` | complete |
| ACPAS metadata | `metadata_R.csv`, `metadata_S.csv` | in git | `MIDI2ScoreTransformer/data/ACPAS-dataset/` | complete |
| PDMX raw | `PDMX.csv` (254,077 rows) and 254,035 `.mxl` | `PDMX.csv.zst`, `PDMX_mxl.tar.gz` (data) | `<datasets root>/pdmx/` | complete |
| PDMX piano subset CSVs | 181,693 rows and 181,670 rows | `pdmx-subset-csvs.tar.zst` (data) | `data/pdmx_piano_subset.csv`, `data/pdmx_piano_subset.deduped.csv` | complete |
| Leakage blocklist | 24 rows (23 leaks) | in git | `data/pdmx_eval_leak_blocklist.csv` | complete |
| pairs_deduped_full + cache | 52,918 rendered pairs, 52,902 manifest rows, 52,902 pickles | `data-pairs_deduped_full.tar.zst`, `data-cache_pdmx_full.tar.zst` (data) | `data/pairs_deduped_full/`, `data/cache_pdmx_full/` | complete |
| Smoke packs | 2,133 pairs and 2,133 pickles | `data-pairs_deduped_smoke.tar.zst`, `data-cache_pdmx_smoke.tar.zst` (data) | `data/pairs_deduped_smoke/`, `data/cache_pdmx_smoke/` | complete |
| April 2026 pack | 5,248 pairs, 5,248 pickles, render test | `legacy-data-pairs.tar.zst`, `legacy-data-cache_pdmx.tar.zst`, `legacy-data-render_test.tar.zst` (data) | `legacy/data/pairs/`, `legacy/data/cache_pdmx/`, `legacy/data/render_test/` | complete; manifest in git at `data/pairs_april_manifest.csv` |
| gentle5k manifest | first 5,000 rows of the full pack | in git | `data/pairs_gentle5k_manifest.csv` | complete (files come with the full pack) |
| Unpaired 58k manifest | 58,000 rows into `pairs_pdmx` | in git | `data/pairs_unpaired_ssl_58k.csv` | files in gpu-corpora; cache missing (rebuilt on demand) |
| classical_clean manifest | 23,783 rows | in git | `data/pairs_classical_clean_manifest.csv` | renders and cache missing; only chunks shipped |
| tuplrate manifest | the same 23,783 rows plus `tuplet_rate` | in git | `data/pairs_classical_clean_tuplrate.csv` | same as above |
| gpu-corpora | six corpora, 737,883 files, 9.4 GB raw | `gpu-corpora.tar.zst` (data) | `data/pairs_pdmx`, `pairs_classical`, `pairs_kern_rubato`, `pairs_scraped`, `pairs_broad`, `pairs_kern`, 13 CSVs, `data/preflight` | partial (see section 12) |
| Benchmark test pieces | 3 MAESTRO performances (wav + midi) | `benchmark-pieces.tar.zst` (quickstart) | `benchmark/<piece>/audio/`, `benchmark/<piece>/midi/` | complete |
| Held-out plan | ASAP metadata copy + two notes | in git | `data/preflight/` | complete |
| Known-missing corpora | tuplrich, maestro_seg, maestro_pseudo, maestro_paired, classical renders, MAESTRO v3 | manifests in git only | see section 15 | data gone |
| Inference priors | two small `.pt` files | in git | `data/duration_priors.pt`, `data/offset_phase_prior.pt` | complete |

## 1. ASAP: pinned clone, generated chunk files, tokenizer cache

**What it is.** The real (performance MIDI, engraved MusicXML) pairs that every model here is trained
and evaluated on. The upstream MIDI2ScoreTransformer code reads it together with the ACPAS metadata.

**Source and version.** `TimFelixBeyer/asap-dataset` at commit
`8cba199e15931975542010a7ea2ff94a6fc9cbee` (the commit the upstream README pins; the clone's nested
`.git` is included in the asset so `git -C MIDI2ScoreTransformer/data/asap-dataset log -1` shows it).
Contents: 16 composer folders, 235 piece folders each with `xml_score.musicxml` and `midi_score.mid`,
1,067 performance MIDIs, `metadata.csv` (1,067 rows), `asap_annotations.json`, `LICENSE.md`.
ACPAS: `metadata_R.csv` (578 rows) and `metadata_S.csv` (1,611 rows) from the ACPAS dataset page,
tracked in git under `MIDI2ScoreTransformer/data/ACPAS-dataset/`.

**License.** CC BY-NC-SA 4.0 (the clone's `LICENSE.md`). Research use, attribution, no commercial use.
Keep it inside the private repository.

**How the dataset is filtered (dataset.py, `ASAPDataset._load_metadata`).** Concatenate the two ACPAS
CSVs, keep `source == ASAP` and `aligned`, drop `constants.SKIP`, drop every performance whose
`asap_annotations.json` entry has `score_and_performance_aligned == false`, drop duplicate
`performance_MIDI_external`, drop `constants.TO_IGNORE_INDICES`. That leaves 967 performances
(split `all`). Split `test` = the 14 `TEST_PIECE_IDS` (14 pieces, one per composer that has test data,
59 performances); `validation` = `piece_id % 10 == 0` among the rest (86); `train` = everything else
(822).

**The 967 chunk files.** `MIDI2ScoreTransformer/midi2scoretransformer/chunker.py` writes
`<performance>_chunks.json` next to each of the 967 performance MIDIs, using the per-beat
`midi_score_beats` and `performance_beats` annotations and a greedy swap of notes across beat
boundaries. A fresh clone does not contain these files; the asset does (the run is in
`legacy/benchmark/box_logs/chunker.log`, 967 files in 8.8 minutes).

**How the cache key works.** `ASAPDataset.__getitem__` builds
`sample_path = performance_MIDI_external.replace("{ASAP}", data_dir + "asap-dataset")` and loads
`<data_dir>/cache/<sha256(sample_path + id)>.pkl`, where `id` is the constructor argument. The code
default is `id="diffusion_2024_04_18"` (dataset.py line 39); the docstring on line 70 still says
`diffusion_2023_10_13`, which is stale. The shipped cache was built with `data_dir="./data/"`, so the
hashed string is, for example, `./data/asap-dataset/Bach/Fugue/bwv_846/Shi05M.mid` followed by
`diffusion_2024_04_18`. Consequences:

- Run anything that instantiates `ASAPDataset` from inside `MIDI2ScoreTransformer/` with
  `--data-dir ./data/` (`benchmark/eval_tier1_asap.py` and `scripts/diag_topk_offset.py` `os.chdir`
  there and pass `"./data/"`; the portable launchers `cd` there before calling `train.py`). Passing an
  absolute data dir changes the key and silently rebuilds the whole cache (music21 parsing, slow).
- 967 of the 997 shipped pickles are keyed this way, one per performance. The other 30 are keyed on
  an absolute macOS data dir from a run on 2026-06-01; they are harmless dead weight on any other
  machine.
- The pickle holds the raw parsed `(input_stream, output_stream)` before bucketing and before beat
  features, so one cache serves baseline and `--use-beat-conditioning` runs alike.
- As read in the code, the train split's length-weighted sampler (dataset.py line 138) looks for a
  `<sha>_.pkl` variant that nothing writes, catches the `FileNotFoundError`, and falls back to uniform
  sampling. Treat the train sampler as uniform.

**Produced by.** `git clone` at the pinned commit; `chunker.py` for the chunk files;
`scripts/build_asap_cache.py --data-dir ./data/` (or `MIDI2ScoreTransformer/build_asap_cache.py`)
to warm the cache (`legacy/benchmark/box_logs/asap_cache_warm.log`: 967 ok in 152 s with 64 workers).

**Consumed by.** `benchmark/eval_tier1_asap.py` (test split, 59 performances), `benchmark/eval_padsweep.py`
and the rerank scripts (the 14-piece corpus), `train.py` (`--dataset-type asap`, `mixed`, `ssl`: the real
half of every SSL run is the 822-performance train split), `scripts/content_dedup.py` (the 14 test
scores are its needles), `scripts/compute_duration_priors.py` (train + validation scores only).

**Shipped as.** `asap-dataset-8cba199e-with-chunks.tar.zst` (quickstart, 118.9 MB) extracts to
`MIDI2ScoreTransformer/data/asap-dataset/`; `asap-cache.tar.zst` (quickstart, 33.5 MB) extracts to
`MIDI2ScoreTransformer/data/cache/`. Both directories are gitignored.

**Path map.** Not involved. ASAP paths are rebuilt from the `{ASAP}` placeholder and the `data_dir`
string; `pathmap.py` only handles the synthetic-pair manifests under `data/`.

## 2. PDMX raw

**What it is.** The public MuseScore-derived corpus used for every synthetic and unpaired pack. The
project uses two upstream files: the metadata CSV and the MusicXML archive.

**Source and version.** Zenodo record 15571083. `PDMX.csv`: 254,077 rows, 62 columns (`path`, `mxl`,
`mid`, `license`, `genres`, `tracks`, `n_notes`, the `subset:*` flags, and so on). The MusicXML
archive: 254,035 `.mxl` files (equal to the number of rows with `subset:all_valid`). The assets were
made from the files downloaded on 2026-04-29: `PDMX_mxl.tar.gz` is that download unchanged (local name
`mxl.tar.gz`), and `PDMX.csv.zst` is a zstd compression of the downloaded `PDMX.csv` made for the
release. Their sha256 values are in `assets/SHA256SUMS`; byte-identity with the files Zenodo serves
today was not checked.

**License.** PDMX is distributed under its own terms (see the Zenodo record). Every row's `license`
column is either `publicdomain` (210,364 rows) or `cc-zero` (43,713 rows); 222,856 rows carry the
authors' `subset:no_license_conflict` flag, which is the first filter the piano subset applies.

**Where it lands.** `scripts/fetch_assets.sh data` writes `PDMX.csv.zst` to `<datasets root>/pdmx/PDMX.csv`
and untars `PDMX_mxl.tar.gz` into `<datasets root>/pdmx/`, giving `<datasets root>/pdmx/mxl/<a>/<b>/<Qm...>.mxl`
(two levels of numeric shard directories, then the `Qm...` hash file name). This is outside the project
root on purpose: every manifest's `mxl` column is relative to that root (`./mxl/8/6/Qm....mxl`) and every
`src_mxl` column is the absolute path on the build machine (`/root/datasets/pdmx/mxl/...`).

**Produced by.** Upstream. Nothing in this project modifies it.

**Consumed by.** `scripts/filter_pdmx.py` (reads `PDMX.csv`), `scripts/make_pairs.py --mxl-root`,
`scripts/content_dedup.py`, `scripts/build_classical_unpaired.py`, `scripts/build_classical_clean_manifest.py`
(all read `.mxl` files under the root resolved by `musicml_paths.PDMX_ROOT`).

**Shipped as.** `PDMX.csv.zst` (data, 44.3 MB) and `PDMX_mxl.tar.gz` (data, 1.89 GB).

**Path map.** `pathmap.py` rewrites the prefix `/root/datasets` to `<datasets root>` (and the macOS
equivalent), so a recorded `src_mxl` resolves to `<datasets root>/pdmx/mxl/...`. The training loader
never opens `src_mxl`; only the corpus builders do, and they take the root as an argument.

## 3. The PDMX piano subset: `pdmx_piano_subset.csv` and `.deduped.csv`

**What it is.** The pool of PDMX scores the project is allowed to render or use as unpaired scores.

**Produced by.** `scripts/filter_pdmx.py` (run 2026-04-29) keeps, in this order: `subset:no_license_conflict`,
`subset:all_valid`, every track is MIDI program 0 (the `tracks` string is made only of `0` tokens:
`0`, `0-0`, `0-0-0-0`, so solo piano in one or more staves), and `32 <= n_notes <= 10000`. It keeps 22
columns: `path, mxl, mid, title, song_name, composer_name, artist_name, license, genres, tracks, n_tracks,
n_notes, song_length.bars, song_length.beats, complexity, rating, n_ratings` and five `subset:*`
flags (`rated`, `deduplicated`, `rated_deduplicated`, `no_license_conflict`, `all_valid`; the raw
`subset:all` flag is not kept). Result: 181,693 rows. The `license` column of the subset holds only `publicdomain` (171,412)
and `cc-zero` (10,281); the filter never tests that column directly, the values are inherited from
PDMX, where those are the only two values. 24,195 subset rows have `classical` in `genres`; they are
the pool behind the classical SSL corpus (section 10).

`data/pdmx_piano_subset.deduped.csv` (2026-05-30) is the subset minus the 23 rows the content
fingerprint audit flagged as copies of evaluation pieces: 181,670 rows (`publicdomain` 171,410,
`cc-zero` 10,260), same column order. The script in this export, `scripts/content_dedup.py`, writes only
the blocklist (section 4); the deduped CSV was then produced by dropping the subset rows whose `mxl`
value appears in the blocklist with `is_leak == True` (verified: the 23 removed rows are exactly the
23 `is_leak` rows). To regenerate it, filter the subset on that column with pandas.

**License.** Inherited from PDMX (`publicdomain` or `cc-zero` per row).

**Consumed by.** `scripts/make_pairs.py --subset-csv` (the full and smoke packs used the deduped CSV via
`scripts/gpu_finetune.sh`; the April pack used the pre-dedup CSV), `scripts/content_dedup.py --subset-csv`
(default: the non-deduped subset, which is what it audits), `scripts/build_classical_clean_manifest.py`
(reads the subset to find leaked `mxl` paths by composer and title), `scripts/gpu_finetune.sh` (refuses
to start without the deduped CSV), `data/preflight/heldout_plan.md` (grep target).

**Shipped as.** `pdmx-subset-csvs.tar.zst` (data, 25.4 MB) extracts both files into `data/`. Both are
gitignored. The deduped CSV also arrives a second time, byte-identical, inside `gpu-corpora.tar.zst`.

**Path map.** The `mxl` column is relative to the PDMX root (`./mxl/...`); scripts join it onto
`musicml_paths.PDMX_ROOT` after stripping `./`.

## 4. The evaluation leakage blocklist: `data/pdmx_eval_leak_blocklist.csv`

**What it is.** The list of PDMX subset rows that are copies (whole or embedded) of evaluation pieces.
Columns: `mxl, composer, title, n_notes, containment, matched_eval, is_leak`. 24 rows: 23 with
`is_leak == True` (containment 0.78 to 1.00) and one borderline row at 0.31 kept for inspection.
`matched_eval` names the evaluation reference: `ASAP:<composer>:<asap folder>` for the 14 ASAP test
pieces or `BENCH:<piece>` for the three benchmark pieces.

**Produced by.** `scripts/content_dedup.py` (run 2026-05-30, about 7 minutes). Method: for each of the
17 evaluation scores (14 ASAP test scores plus the three benchmark ground-truth scores) extract the
time-sorted pitch sequence with music21, build transposition-invariant interval 5-grams, and for every
subset row whose `composer_name` contains an evaluation composer (1,134 candidates, 1 parse failure)
compute containment = |needle grams in candidate| / |needle grams|. Rows at or above 0.2 are written;
`is_leak` is containment >= 0.4 (positive controls scored 0.78 to 0.96, unrelated same-composer pieces
below 0.10). The script needs the ASAP clone, the ACPAS metadata and `benchmark/liszt_transcendental/gt_score.3staff.raw.musicxml`.

**What the audit found.** `benchmark/LEAKAGE_AUDIT.md`: 12 of the 17 evaluation pieces exist somewhere in
PDMX (metadata plus content evidence), 5 are content-proven clean (Mozart K.332, Haydn XVI:31, Schumann
Arabeske, Scriabin Op.8/11, Prokofiev Toccata). Two of the 12 (Debussy "Reflets", Ravel "Ondine") leak
only into full PDMX, not into the piano subset, so they do not appear in the blocklist. The leaks the
blocklist catches: Bach BWV 846 fugue (6 copies), Chopin Ballade 1 (7), Brahms Op.118/2 (2), Liszt
"Gondoliera" under the alternate title "Venezia e Napoli" (1), Rachmaninoff Op.23/4 (1), Schubert
D.899/1 (1), plus the benchmark pieces Chopin Op.10/4 (3), Op.25/11 (1), Liszt Mazeppa (1).

**Consumed by.** The deduped subset (section 3) is defined by it. `scripts/build_classical_clean_manifest.py`
does not read it; that script uses its own composer-and-title regexes over the subset.

**Shipped as.** In git. A byte-identical copy also sits inside `gpu-corpora.tar.zst`.

**Caveat.** The released model (`MIDI2ScoreTF.ckpt`) was trained on ASAP only, so the blocklist is a
constraint on this project's own PDMX training, not on the baseline reproduction. If training ever uses
full PDMX rather than the piano subset, rerun `content_dedup.py --subset-csv <full PDMX csv>`.

## 5. `pairs_deduped_full` and `cache_pdmx_full`: the main synthetic pack

**What it is.** The main synthetic (perturbed performance MIDI, engraved MusicXML) corpus rendered from
the deduped piano subset on the GPU box on 2026-05-31. This is what the `pdmx` and `mixed` loaders train
on; `gentle5k`, `scripts/gpu_finetune.sh` and `scripts/run_arm1_mixed.sh` all index it.

**Produced by.** `scripts/gpu_finetune.sh` with `MODE=full` calls

    scripts/make_pairs.py --subset-csv data/pdmx_piano_subset.deduped.csv --mxl-root <pdmx root>
        --out-dir data/pairs_deduped_full --cache-dir data/cache_pdmx_full
        --manifest data/pairs_deduped_full/_manifest.csv --errors data/pairs_deduped_full/_errors.log
        --n 50000 --n-jobs -1 --prefer-multi-track

`make_pairs.py` with `--prefer-multi-track` sorts the subset by (number of tracks descending, `n_notes`
ascending), takes the first `int(n * 1.1) + 50` = 55,050 rows, and processes each one: copy the source
`.mxl` next to the output, render it to a perturbed MIDI plus an alignment JSON with
`scripts/expressive_render.py` (seed = 42 + row index; per-chord onset jitter, tempo drift, velocity
wiggle, duration scaling), build `_chunks.json` from the alignment (per-beat index lists, `midi[b] ==
mxl[b]` by construction, `swapped: false`), tokenize MIDI and MusicXML with `MultistreamTokenizer`, check
the two token counts agree, and save the pickle. Pairs are numbered `000000` upward in candidate order,
so a numeric gap means a failed row.

**What is on disk.** 52,918 ids, each with four files: `<id>.mid`, `<id>.mxl` (the source score, copied),
`<id>_chunks.json`, `<id>.alignment.json`; plus `_manifest.csv` (52,902 rows, absolute `/root/...`
paths) and `_errors.log` (2,148 lines: 2,124 `render:parse`, 16 `cache:ValueError`, 8
`render:no_notes_in_score`). 211,674 files, 1.63 GB raw. The 16 ids on disk but not in the manifest are
the `cache:ValueError` rows: the renderer and chunker succeeded, then the tokenizer produced different
token counts for MIDI and MusicXML (for example `017606`: midi=78, mxl=80), so no pickle was written and
no manifest row. They are inert; loaders only iterate the manifest.

Byte budget: `.alignment.json` is 1.24 GB (76 percent of the pack), `.mxl` 205 MB, `_chunks.json`
105 MB, `.mid` 60 MB, `_manifest.csv` 21 MB. Only the render scripts read `.alignment.json`
(`make_pairs.py` builds chunks from it; `rerender_rubato.py`, `render_all_kern.py`,
`build_classical_unpaired.py`, `build_kern_pairs.py` do the same); no training or evaluation code opens
it, so it can be dropped locally if disk is tight.

**The cache.** `data/cache_pdmx_full/` holds 52,902 pickles, one per manifest row. Key:
`sha256(<the manifest's midi string>)`, for example
`sha256("/root/Music-ML-BayenLab/data/pairs_deduped_full/000001.mid") = 33f0f5ef...` (verified against
the manifest's `cache` column). The hashed string is the raw recorded path, never the resolved one.
Each pickle is `torch.save((input_stream, output_stream))`, the raw parsed streams before bucketing.

**License.** Derived from PDMX rows (`publicdomain` or `cc-zero`). The rendered MIDIs are this
project's own synthetic output.

**Consumed by.** `train.py fit --dataset-type pdmx|mixed --manifest data/pairs_deduped_full/_manifest.csv`
(`scripts/run_arm1_mixed.sh`, `scripts/gpu_finetune.sh`), `PDMXDataset` in
`MIDI2ScoreTransformer/midi2scoretransformer/pdmx_dataset.py`, `scripts/gpu_launchers/launch_gentle.sh`
(builds gentle5k from it).

**Shipped as.** `data-pairs_deduped_full.tar.zst` (data, 367.9 MB) extracts to `data/pairs_deduped_full/`;
`data-cache_pdmx_full.tar.zst` (data, 130.4 MB) extracts to `data/cache_pdmx_full/`.

**Path map.** `pathmap.resolve_path` rewrites the prefix `/root/Music-ML-BayenLab` to the project root,
so `midi` and `chunks` open at `data/pairs_deduped_full/<id>.mid` and `..._chunks.json`.
`pathmap.cache_dir_for` maps the resolved pairs directory `data/pairs_deduped_full` to
`data/cache_pdmx_full` (an explicit entry in `_DEFAULT_CACHE_MAP`, because the original rule
`<pairs parent>/cache_pdmx` would point at the wrong directory) and the loader opens
`data/cache_pdmx_full/<sha256(recorded midi string)>.pkl`. If a pickle is missing the loader parses
`.mid` plus the `.mxl` or `.musicxml` beside it and writes the pickle.

## 6. The smoke packs: `pairs_deduped_smoke` and `cache_pdmx_smoke`

**What it is.** The `MODE=smoke` run of the same script, meant to de-risk the pipeline before the full
render: `--n 2000`, so 2,250 candidates (ids `000000` to `002249`) in the same deterministic order as
the full pack. 2,133 pairs succeeded (the lowest surviving id is `000001`, the highest `002249`; 117
error lines), 2,133 pickles.

**Relationship to the full pack.** Same sort, same seeds, same renderer: every smoke id is also in the
full pack with the same source file, and the MIDIs are byte-identical (spot-checked on `000001`,
`000003`, `001000`). The cache pickles differ in name only, because the key hashes the recorded path
(`.../pairs_deduped_smoke/...` versus `.../pairs_deduped_full/...`).

**Consumed by.** `scripts/gpu_finetune.sh MODE=smoke`; useful for any quick loader or training test.

**Shipped as.** `data-pairs_deduped_smoke.tar.zst` (data, 65.3 MB) to `data/pairs_deduped_smoke/`;
`data-cache_pdmx_smoke.tar.zst` (data, 17.1 MB) to `data/cache_pdmx_smoke/`.

**Path map.** As for the full pack; `_DEFAULT_CACHE_MAP` has an entry mapping `data/pairs_deduped_smoke`
to `data/cache_pdmx_smoke`.

## 7. The April 2026 pack: `legacy/data/pairs`, `legacy/data/cache_pdmx`, `legacy/data/render_test`

**What it is.** The first synthetic pretraining set, rendered on the macOS laptop on 2026-04-29 from the
pre-dedup piano subset (the content audit came a month later). Superseded by `pairs_deduped_full`; kept
because the April-era checkpoints under `legacy/MIDI2ScoreTransformer/checkpoints/` were trained on it.

**Produced by.** `scripts/make_pairs.py --subset-csv data/pdmx_piano_subset.csv --mxl-root ~/datasets/pdmx
--out-dir data/pairs --cache-dir data/cache_pdmx --manifest data/pairs/_manifest.csv --errors
data/pairs/_errors.log --n 5000 --n-jobs -1 --prefer-multi-track`. The exact command line is not
logged; `legacy/docs/reports/SYNTHETIC_PRETRAIN_STATUS.md` records this invocation with `--n 50000` as
the planned GPU run and notes that "the Mac CPU run only did 5K", and the pack's 5,550 candidate ids
(`000000` to `005549`) match `--n 5000` (`int(5000 * 1.1) + 50`). 5,248 pairs, 302 errors
(300 `render:parse`, 2 `no_notes_in_score`).

**What is on disk.** `legacy/data/pairs/`: 5,248 each of `.mid`, `.mxl`, `_chunks.json`, `.alignment.json`,
plus `_manifest.csv` and `_errors.log`. `legacy/data/cache_pdmx/`: 5,248 pickles.
`legacy/data/render_test/`: the one-off render sanity check (`Twinkle_perturbed.mid`,
`Twinkle_perturbed.alignment.json`, an `eyeball/` folder).

**The manifest.** Byte-identical copies of the pack's `_manifest.csv` and `_errors.log` sit at
`data/pairs_april_manifest.csv` and `data/pairs_april_errors.log` (both under `data/`; when and by
what they were copied is not recorded, the copies keep the pack's 2026-04-29 timestamps). The manifest records
relative paths: `midi = data/pairs/000001.mid`, `chunks = data/pairs/000001_chunks.json`,
`cache = data/cache_pdmx/<sha>.pkl`; `src_mxl` is an absolute path under the laptop's home directory.
The cache key is `sha256("data/pairs/000001.mid") = b6220996...` (verified).

Note: `.gitignore` in this export ignores `*.log` (with an exception only for the box logs), so
`data/pairs_april_errors.log` will not be committed unless an exception is added.

**Relationship to the full pack.** The same deterministic ordering means that below id `001992` every
id present in both manifests (1,892 of them) names the same source score, and for those the renders
are byte-identical (checked on `000001` and `001000`); from `001992` on, the first row the dedup
removed shifts the numbering, so an April id `k` mostly names the full pack's source `k - 1`
(3,177 of the 3,356 April rows at or above `001992`; the rest differ because of failed rows).

**Consumed by.** Nothing active. `legacy/MIDI2ScoreTransformer/checkpoints/` (the `pretrain_pdmx` epochs,
archive-checkpoints tier) were trained on it; `legacy/scripts/eyeball_check.py` reads `render_test`.

**Shipped as.** `legacy-data-pairs.tar.zst` (data, 101.1 MB) to `legacy/data/pairs/`;
`legacy-data-cache_pdmx.tar.zst` (data, 29.4 MB) to `legacy/data/cache_pdmx/`;
`legacy-data-render_test.tar.zst` (data, 59 KB) to `legacy/data/render_test/`.

**Path map.** `pathmap.py` has a dedicated rule: the prefix `data/pairs/` rewrites to
`legacy/data/pairs/`, and `_DEFAULT_CACHE_MAP` sends `legacy/data/pairs` to `legacy/data/cache_pdmx`.
The hash is still computed on the relative string, so the shipped pickles are found. (If a `data/pairs/`
directory exists in the working directory, `resolve_path` returns the relative path unchanged because
it exists; do not create one.)

## 8. `data/pairs_gentle5k_manifest.csv`

**What it is.** The first 5,000 rows of `data/pairs_deduped_full/_manifest.csv`
(`head -5001`, taken in `scripts/gpu_launchers/launch_gentle.sh`, 2026-06-01). It indexes files of the
full pack; it has no data of its own. Byte-identical copies exist in git and inside `gpu-corpora.tar.zst`.

**Consumed by.** The two "gentle" low-learning-rate warm-start fine-tunes of the released checkpoint
(`ft_gentle_2e6`, `ft_gentle_1e6`; `train.py fit --stage pretrain_pdmx --manifest
data/pairs_gentle5k_manifest.csv --init-ckpt MIDI2ScoreTF.ckpt`, 2 epochs). Those were diagnostic runs
(`benchmark/GPU_FINETUNE_RESULTS.md`) and both aborted (`benchmark/diag/train_gentle_*.log` end with a
DataLoader worker killed by signal); no checkpoint or result from them exists (see `RUNS.md`).

**Path map.** Identical to the full pack (absolute `/root/...` strings, cache in `data/cache_pdmx_full`).
Requires the two `data-pairs_deduped_full` and `data-cache_pdmx_full` assets.

## 9. `data/pairs_unpaired_ssl_58k.csv` and its parent corpus `pairs_pdmx`

**What it is.** The unpaired-score manifest for the masked self-supervision recipe, sized to match the
released model's exposure (Beyer and Dai 2024 trained on about 58k unpaired MuseScore scores). It is a
random 58,000-row subset of `data/pairs_unpaired_ssl_manifest.csv` (170,420 rows), which itself is the
reconstructed manifest of `data/pairs_pdmx` (the GPU box's big PDMX render).

**How the rows look.** `id = 12815`, `src_mxl` empty, `midi = data/pairs_pdmx/012815.mid`,
`mxl = data/pairs_pdmx/012815.mxl`, `chunks = data/pairs_pdmx/012815_chunks.json`,
`cache = data/cache_pdmx/<sha256("data/pairs_pdmx/012815.mid")>.pkl`, `n_notes = 0`, `n_measures = 0`,
`n_in_tokens = 256`, `n_out_tokens = 256`. The zeros and 256s are placeholders: the original render hung
on its tail without writing a manifest, and `legacy/scripts/rebuild_and_combine.py` rebuilt it from the
files on disk (keeping only ids that had a `.mid`, a `.mxl` or `.musicxml`, a `_chunks.json` and a cache
pickle). Because `n_in_tokens` is constant, the length-weighted sampler is effectively uniform.

**What the unpaired loader does with it.** `UnpairedScoreDataset` (pdmx_dataset.py) loads the cached
pair, keeps the input pitch stream (equal to the score pitch in these 1:1 renders), zeroes onset and
duration, sets velocity to 64, and the `_WithConditioning` wrapper in train.py adds the conditioning
token (1 = unpaired). So the perturbed MIDI timing in `pairs_pdmx` is never seen by SSL runs; only the
score tokens and chunk boundaries matter.

**Produced by.** `make_pairs.py` on the GPU box into `data/pairs_pdmx` (170,487 ids on disk, see section
12) from the piano subset (the deduped CSV is the one committed on the gpu branch; the render command
itself is not recorded); `rebuild_and_combine.py` for the 170,420-row manifest; a random subset for
the 58k file (the selection command is not in the export).

**Consumed by.** `scripts/run_ssl_v2.sh` (`train.py fit --dataset-type ssl --manifest
data/pairs_unpaired_ssl_58k.csv`, the ssl_v2 run, val 0.5645). The 170k parent manifest fed
`scripts/run_ssl_unpaired.sh` (ssl_unpaired).

**Shipped as.** The manifest is in git (and again, byte-identical, inside `gpu-corpora.tar.zst`). The
files it points to come with `gpu-corpora.tar.zst` (`data/pairs_pdmx/`: `.mid`, `.mxl`, `_chunks.json`,
`.alignment.json` per id). The cache directory `data/cache_pdmx/` is not shipped (it was never in git on
any branch), so the first pass over this manifest re-tokenizes each pair from `.mid` plus `.mxl` and
writes `data/cache_pdmx/<sha>.pkl`. Expect a long first epoch (music21 parse per score) and about 125
bytes of cache per note (the full pack's cache is 1.13 GB raw for 9.1 million notes, about 21 KB per
score), so roughly 1.2 GB for the 58k rows and 3.6 GB for all of `pairs_pdmx`.

**Path map.** The recorded paths are relative (`data/pairs_pdmx/...`). `resolve_path` returns a string
unchanged when `os.path.exists` is true, and otherwise tries the prefix rules, none of which match
`data/pairs_pdmx/`. So either run from the project root (as the original launchers did) or set
`MUSICML_PATH_MAP="data/pairs_pdmx/=<project root>/data/pairs_pdmx/"`. The portable launchers
(`scripts/run_ssl_classical_clean.sh`, `run_ssl_tuplet.sh`) `cd` into `MIDI2ScoreTransformer/` for the
ASAP cache key, which breaks relative manifest paths unless that override is set. The cache lands in
`<resolved pairs parent>/cache_pdmx`, which is `data/cache_pdmx` either way.

## 10. classical_clean: `data/pairs_classical_clean_manifest.csv`

**What it is.** The manifest of the genre-matched unpaired corpus behind the two live own checkpoints
(`ssl_classical_clean` epoch 13, and `ssl_tuplet20` which warm-starts from it). 23,783 rows, one per
PDMX piano-subset score whose `genres` contains `classical`, minus title-matched copies of the 14 ASAP
test pieces.

**How it was built (two steps, both on the GPU box).**

1. `scripts/build_classical_unpaired.py <list of .mxl paths> pairs_classical data/pairs_classical_manifest.csv`.
   For each listed score it renders a MIDI with `expressive_render.render` (seed = 1000 + position in
   the list), builds `_chunks.json` from the alignment, tokenizes and writes
   `data/cache_pdmx/<sha256(midi path)>.pkl`, then deletes the MIDI and the alignment JSON ("the unpaired
   branch masks timing anyway"). The id is `"c" + sha1(<the .mxl path string>)[:14]`. The manifest is
   written in completion order (`as_completed`), not list order. Result: 23,818 rows in
   `pairs_classical_manifest.csv` (23,832 `_chunks.json` on disk).
2. `scripts/build_classical_clean_manifest.py` drops rows whose `src_mxl` matches one of 14 composer-and-title
   regexes over the piano subset (one regex per ASAP test piece, deliberately aggressive). Result:
   23,783 rows (35 dropped) in `pairs_classical_clean_manifest.csv`.

Rows look like: `id = cd0bec5314f2427`, `src_mxl = /root/datasets/pdmx/mxl/1/11/Qm....mxl`,
`midi = /root/Music-ML-BayenLab/data/pairs_classical/cd0bec5314f2427.mid`, `mxl` = same as `src_mxl`,
`chunks = /root/Music-ML-BayenLab/data/pairs_classical/cd0bec5314f2427_chunks.json`,
`cache = /root/Music-ML-BayenLab/data/cache_pdmx/99531b98....pkl` (= sha256 of the `midi` string,
verified), real `n_notes` and token counts.

**What is and is not available.** In git: the manifest. In `gpu-corpora.tar.zst`: `data/pairs_classical/`
with 23,832 `_chunks.json` files and nothing else. NOT available anywhere: the rendered `.mid` files (the
builder deleted them) and the `data/cache_pdmx/*.pkl` pickles (never in git; not retrieved from the box
before it was terminated). The loader's fallback needs the `.mid` next to the `.mxl`, so this manifest
cannot be trained on as is.

**Rebuild route.** Fetch the `data` tier (PDMX). Build a list file from the manifest's `src_mxl` column
with `/root/datasets` replaced by `<datasets root>` (23,783 lines; the files exist because PDMX ships).
Run `scripts/build_classical_unpaired.py <list> pairs_classical data/pairs_classical_manifest.csv`, then
`scripts/build_classical_clean_manifest.py` (edit its `SUBSET`, `MANIFEST`, `OUT` constants if you want
different names), then train with `MANIFEST=<new clean manifest> scripts/run_ssl_classical_clean.sh`. Use
the manifest the rebuild writes, not the tracked one: the rebuilt ids, cache keys and `midi` strings all
derive from the new local paths. Expected cost: the box did 23,780 scores in about 20 minutes with 24
workers; a laptop will take hours.

The rebuild is a new version of the corpus, not a reproduction: the original source-list order is not
recorded (the manifest is in completion order), so the render seeds (`1000 + position`) differ, the
MIDIs differ, and the pickles are not byte-identical. For the unpaired SSL branch this does not matter in
substance (timing is masked; the score tokens and chunk boundaries are the same), but checkpoints
trained on the rebuilt corpus should be labelled as a new run. Also note the original list was 23,818
successful renders out of an unrecorded list length, and a 2026-06-09 rebuild on the box
(`legacy/benchmark/box_logs/build_unpaired.log`) got 23,780 of 23,783.

**License.** PDMX rows (`publicdomain` or `cc-zero`).

**Consumed by.** `scripts/run_ssl_classical_clean.sh` (ssl_classical_clean), `scripts/run_ssl_tuplet.sh`
(ssl_tuplet5/25/20/20e30 and the reshape runs via the tuplrate manifest), `scripts/run_ssl_recipe.sh`
(ssl_recipe), `scripts/compute_tuplet_rates.py` (reads the `cache` column; needs the pickles).
`pairs_classical_manifest.csv` (pre-filter) fed `scripts/run_ssl_classical.sh`, and
`pairs_classical_big_manifest.csv` (clean + scraped) fed `scripts/run_ssl_bigc.sh`.

**Path map.** `/root/Music-ML-BayenLab` rewrites to the project root, `/root/datasets` to
`<datasets root>`; `cache_dir_for` falls through to the default rule `<pairs parent>/cache_pdmx`, so
the loader looks in `data/cache_pdmx/`.

## 11. `data/pairs_classical_clean_tuplrate.csv`

**What it is.** The clean manifest with two extra columns: `tuplet_rate` (fraction of a score's notes
whose duration, rounded to the 1/24-quarter grid, is not a multiple of 3, that is, a tuplet duration)
and `n_notes_cache` (notes in the cached output stream). Same 23,783 rows, same order.

**Produced by.** `scripts/compute_tuplet_rates.py --manifest data/pairs_classical_clean_manifest.csv
--out data/pairs_classical_clean_tuplrate.csv`. It reads each row's `cache` pickle, so it needs the
missing pickles; after a rebuild (section 10), run it on the rebuilt manifest. Distribution on the box's
rebuilt corpus (`box_logs/balanced_driver.log`): 86.5 percent of scores have zero tuplets, mean rate
0.017, p95 0.107, p99 0.355.

**Consumed by.** `train.py fit --tuplet-gamma <g>` (the "reshape" lever: `PDMXDataset` multiplies the
length weight by `(tuplet_rate + 0.05) ** gamma` when the column exists), used by the ssl_reshape_g1,
ssl_reshape_g2 and ssl_combo runs (`MANIFEST=data/pairs_classical_clean_tuplrate.csv` in
`run_ssl_tuplet.sh`, per `RUNS.md`). The later `ssl_balanced` and `ssl_bal_*` runs also had the reshape
on, but their logs do not name the manifest; the same day's `box_logs/balanced_driver.log` wrote
`data/pairs_classical_rebuilt_tuplrate.csv` (23,780 rows, over the box's rebuilt corpus), which is not
in the export, so which of the two they read is unknown. This file also defines the tuplrich selection
(section 15): the 1,278 rows with `tuplet_rate >= 0.10`.

**Shipped as.** In git; identical copy inside `gpu-corpora.tar.zst`.

**Path map.** As section 10.

## 12. The gpu-corpora asset: six corpora, 13 CSVs, `data/preflight`

**What it is.** `gpu-corpora.tar.zst` (data tier, 1.41 GB compressed, 9.4 GB raw, 737,883 files) is a
`git archive` of the `data/` tree of the original repository's `gpu` branch, the branch the GPU box
committed to. The six corpus directories and seven of the CSVs were never on `main`. It extracts into
`data/`, adding six corpus directories, 13 CSV files (`assets/ASSETS.csv` says "12 manifests": the
thirteenth CSV is `pdmx_piano_subset.deduped.csv`, which is not a manifest) and `data/preflight/`. `fetch_assets.sh` extracts
with `tar -k`, so the files that already exist in git (identical copies) are left alone.

**License.** `pairs_pdmx`, `pairs_classical` and the PDMX rows of every manifest: PDMX terms
(`publicdomain` or `cc-zero`). `pairs_kern`, `pairs_kern_rubato`, `pairs_broad` and `pairs_scraped`
were built from online Humdrum kern and MusicXML sources (KernScores / craigsapp repositories, OpenScore,
the music21 corpus) whose individual terms were not recorded; treat them as internal research data and do
not redistribute.

### 12a. The six corpus directories (what each actually contains)

| Directory | Files | Per id | Ids on disk | Manifest rows | Source | Builder |
|---|---:|---|---:|---:|---|---|
| `pairs_pdmx` | 681,950 | `.mid`, `.mxl`, `_chunks.json`, `.alignment.json` | 170,487 | 170,420 (`_manifest.csv`, reconstructed) | PDMX piano subset (the deduped CSV is the one committed on the gpu branch; the exact command is not recorded) | `make_pairs.py` on the box; manifest by `legacy/scripts/rebuild_and_combine.py` |
| `pairs_classical` | 23,832 | `_chunks.json` only | 23,832 | 23,818 / 23,783 (manifests in `data/`) | `genres` contains classical, PDMX piano subset | `scripts/build_classical_unpaired.py` (deletes MIDI and alignment after caching) |
| `pairs_kern_rubato` | 21,690 | `.mid`, `_chunks.json`, `.alignment.json` | 7,230 (= 241 scores x 30 rubato seeds, ids `<kern id>_r<k>`) | 7,228 (inside `pairs_rubato_manifest.csv`) | the 241 `pairs_kern` MusicXML files | `scripts/render_all_kern.py` (K=30, seed 1000) |
| `pairs_scraped` | 5,400 | `_chunks.json` only | 5,400 | 5,397 (`pairs_scraped_manifest.csv`) | 12,163 scraped MusicXML consolidated to 5,422 solo-piano scores by `scripts/consolidate_scrape.py` (md5 dedup, at most 2 parts, 14-piece leak filter) | `build_classical_unpaired.py` (same deletion) |
| `pairs_broad` | 4,029 | `.mid`, `.musicxml`, `_chunks.json`, `.alignment.json` | 1,007 | 988 (`_manifest.csv`, reconstructed) | "broad" kern corpus: Scriabin, Chopin (incl. NIFC first editions), Mozart, Beethoven, Scarlatti, Joplin, Haydn, Hummel; leakage-audited | `scripts/build_kern_pairs.py` (kern to MusicXML via music21, hand-label fix by average pitch) |
| `pairs_kern` | 966 | `.mid`, `.musicxml`, `_chunks.json`, `.alignment.json` | 241 | 241 (`_manifest.csv`) | Mysterium (Scriabin, 189 scores) and chopin-mazurkas (52) kern repositories | `scripts/build_kern_pairs.py` |

Notes per directory:

- `pairs_pdmx` is the superset render (ids `000001` to `178719`, 170,487 present). Its `_manifest.csv`
  and `_errors.log` (36,724 lines) are inside the directory. Row fields are placeholders (see section 9).
- `pairs_classical` and `pairs_scraped` contain chunk files only. Chunk files are per-beat index lists,
  useless without the token cache they were computed for. Both corpora need the rebuild route of
  section 10 (for scraped, the 5,422 source files `/root/datasets/scraped/scr_*.musicxml` are NOT in the
  export, so `pairs_scraped` cannot be rebuilt at all).
- `pairs_kern_rubato` has MIDI but no score next to it: the manifest's `mxl` column points at
  `data/pairs_kern/<kern id>.musicxml`. The loader's cache fallback looks for a score beside the `.mid`,
  so these rows need the (missing) cache, or a re-render with `scripts/render_all_kern.py` (which needs
  `data/pairs_kern/*.musicxml`, shipped, and writes a new manifest).
- `pairs_kern`'s manifest records `cache = data/cache_kern/<sha>.pkl`; that directory is not shipped.
  The loader ignores the `cache` column and will rebuild into `data/cache_pdmx/` from `.mid` plus
  `.musicxml`, which are both present. `_errors.log` is inside the directory.
- `pairs_broad` has 1,007 ids on disk and 988 manifest rows (the manifest was reconstructed from ids
  that had a cache pickle at the time). `src_mxl` is empty, so the kern source of each pair is not
  recorded in the export.
- No cache directory is shipped for any of the six corpora. The original box-side caches
  (`data/cache_pdmx`, `data/cache_kern`) were never committed.

### 12b. The 13 CSVs

Six of them are byte-identical to files already in git or in another asset:
`pairs_classical_clean_manifest.csv`, `pairs_classical_clean_tuplrate.csv`, `pairs_gentle5k_manifest.csv`,
`pairs_unpaired_ssl_58k.csv`, `pdmx_eval_leak_blocklist.csv` (all in git) and
`pdmx_piano_subset.deduped.csv` (in `pdmx-subset-csvs.tar.zst`). The other seven exist only here:

| Manifest | Rows | Composition (by `midi` directory) | Built by | Used by |
|---|---:|---|---|---|
| `pairs_classical_manifest.csv` | 23,818 | `pairs_classical` | `build_classical_unpaired.py` | `scripts/run_ssl_classical.sh` (ssl_classical); input of `build_classical_clean_manifest.py` |
| `pairs_classical_big_manifest.csv` | 29,180 | 23,783 `pairs_classical` + 5,397 `pairs_scraped` | concatenation of the clean and scraped manifests | `scripts/run_ssl_bigc.sh` (ssl_bigc) |
| `pairs_scraped_manifest.csv` | 5,397 | `pairs_scraped` | `build_classical_unpaired.py` on `/root/datasets/scraped/*.musicxml` | merged into the big manifest |
| `pairs_unpaired_ssl_manifest.csv` | 170,420 | `pairs_pdmx` | `rebuild_and_combine.py` (same rows as `pairs_pdmx/_manifest.csv`) | `scripts/run_ssl_unpaired.sh` (ssl_unpaired); parent of the 58k file |
| `pairs_combined_manifest.csv` | 171,408 | 170,420 `pairs_pdmx` + 988 `pairs_broad` | `rebuild_and_combine.py` | the ar_full4 causal-AR build (`--dataset-type mixed`) |
| `pairs_upweighted_manifest.csv` | 113,640 | first 84,000 `pairs_pdmx` + 988 `pairs_broad` repeated 30 times | `legacy/scripts/build_upweighted.py` | ar_full5 (kern x30 repetition, negative result) |
| `pairs_rubato_manifest.csv` | 91,228 | 84,000 `pairs_pdmx` + 7,228 `pairs_kern_rubato` | `scripts/render_all_kern.py` / `scripts/rerender_rubato.py` | `scripts/run_ar_rubato.sh` (ar_rubato, diverged) |

Path styles are mixed. `pairs_classical*`, `pairs_scraped*` and the kern_rubato rows use absolute
`/root/Music-ML-BayenLab/...` strings; `pairs_pdmx`, `pairs_broad` and `pairs_kern` rows use relative
`data/...` strings; the kern_rubato rows' `mxl` column is relative (`data/pairs_kern/...`) while their
`midi` and `chunks` are absolute. The cache key is always `sha256(<midi string as written>)`.

### 12c. `data/preflight`

Three files, byte-identical to the copies in git (section 14).

**Path map for the whole asset.** Absolute `/root/Music-ML-BayenLab/...` resolves to the project root
automatically. Relative `data/...` strings resolve only when the working directory is the project root
or when `MUSICML_PATH_MAP` has entries such as
`data/pairs_pdmx/=<root>/data/pairs_pdmx/;data/pairs_broad/=<root>/data/pairs_broad/;data/pairs_kern/=<root>/data/pairs_kern/`
(prefix match, first rule wins; the built-in `data/pairs/` rule does not match these because of the
slash). Every cache lookup for these corpora falls through to the default rule and lands in
`data/cache_pdmx/`, which starts empty.

## 13. The benchmark test pieces (MAESTRO, tier-2 end-to-end evaluation)

**What it is.** Three real piano performances with audio, used by `benchmark/eval_tier2_e2e.py` to score
the full audio-to-score pipeline (`transcribe.py -t hft -b transformer`) and by `benchmark/eval_decomposed.py`
to separate transcription error from notation error. Ground truth: the ASAP `xml_score.musicxml` for
the two Chopin etudes (both in the released model's ASAP training set), and for Mazeppa either the
ASAP-edition score or the PDMX-sourced `benchmark/liszt_transcendental/gt_score.musicxml` (see
`benchmark/DECOMPOSED_FINDINGS.md` for why the PDMX edition inflates MUSTER).

**Source and version.** MAESTRO v3.0.0 (the metadata copy is in git at `benchmark/maestro-v3.0.0.csv`,
1,276 rows). Matching the performance MIDIs' durations and note counts to that CSV identifies the three
performances; the export does not record this choice explicitly, so the identification below is by
duration (119.15 s, 217.54 s, 494.71 s; exact to the tenth of a second):

| Folder | Piece | MAESTRO file (year, split) |
|---|---|---|
| `benchmark/chopin_op10` | Chopin, Etude Op.10 No.4 | `2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--3` (train) |
| `benchmark/chopin_op25` | Chopin, Etude Op.25 No.11 | `2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--3` (train) |
| `benchmark/liszt_transcendental` | Liszt, Transcendental Etude No.4 "Mazeppa" | `2008/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--4` (train) |

Each folder's `catalog.json` lists every MAESTRO performance of the whole opus (12 etudes each) with
file names and durations; the three used are among those entries.

**License.** CC BY-NC-SA 4.0 (MAESTRO). Keep inside the private repository.

**Files.** Per folder: `audio/<stem>.wav` (the MAESTRO recording), `midi/<stem>.midi` (the MAESTRO
performance MIDI, the ground-truth note list for Block 1), `midi/<stem>_hft.mid` (the hFT-Transformer
transcription of the audio; present in the export tree and in the asset), and the output folders `musescore/`,
`transformer/` (pdf and xml) produced by earlier runs. `*.pre.wav` files are preprocessed intermediates
and are gitignored. Note that `.gitignore` also lists `benchmark/*/midi/`, so the `_hft.mid` files
present in the export would not be committed; they come back with the asset.

**Consumed by.** `benchmark/eval_tier2_e2e.py`, `benchmark/eval_decomposed.py`, `benchmark/trackb_sweep.py`,
`scripts/content_dedup.py` (Mazeppa GT score is one of the 17 needles), the README quick start.

**Shipped as.** `benchmark-pieces.tar.zst` (quickstart, 128.5 MB) extracts the three `audio/*.wav` and
the `midi/*.midi` plus `midi/*_hft.mid` into their folders.

**Path map.** Not involved; the eval scripts build paths from `musicml_paths.REPO_ROOT`.

## 14. The held-out plan: `data/preflight/`

**What it is.** Three small files from the 2026-04-29/30 preflight before the first pretraining round:

- `asap_metadata.csv`: a byte-identical copy of the pinned ASAP clone's `metadata.csv` (1,067 rows),
  fetched directly from the commit so the overlap check could run before the clone existed.
- `step1_overlap_findings.md`: which benchmark pieces are in ASAP (Op.10/4: 22 performances; Op.25/11:
  19 according to the note, although `asap_metadata.csv` has 24 rows under `Chopin/Etudes_op_25/11/`;
  Mazeppa: none), the block-2 result on each, and the decision to proceed. Neither etude is one of
  the 14 ASAP test pieces, so both are in the released model's training data.
- `heldout_plan.md`: the ASAP composers to avoid, candidate held-out pieces from composers absent from
  ASAP (Mendelssohn, Saint-Saens, Albeniz; alternatives Granados, Medtner, Czerny), and the four
  verification steps (grep ASAP metadata, grep the PDMX subset, source a public-domain score, source a
  GT MIDI). The note ends by saying the three MAESTRO benchmark pieces remained the primary measure and
  the held-out curation was deferred; no held-out piece was ever added to the export.

**Consumed by.** People, not scripts. `heldout_plan.md` references `data/pdmx_piano_subset.csv` as the
second grep target.

**Shipped as.** In git; identical copy in `gpu-corpora.tar.zst`.

## 15. Known-missing corpora (manifest in git, data gone)

These manifests are tracked so the record of the runs survives; the files they index were on the GPU box
or on external disks and were not retrieved. Nothing in the release contains them. Every one of them
needs a rebuild before use, and in two cases a rebuild is impossible from the export alone.

| Manifest (rows) | What it indexed | Builder | Rebuild from the export? |
|---|---|---|---|
| `data/pairs_tuplrich_manifest.csv` (1,278) | PAIRED renders (`data/pairs_tuplrich/<id>.mid/.mxl/_chunks.json`, cache `data/cache_tuplrich/`) of the 1,278 classical_clean scores with `tuplet_rate >= 0.10` (the set matches exactly), rendered 2026-06-09 (`box_logs/render_tuplrich.log`: 1,268 new + 10 from an earlier partial run) | `make_pairs.py` on a 1,278-row CSV that is not in the export | Yes: select those rows from the tuplrate manifest (after section 10's rebuild), write a CSV with an `mxl` column relative to the PDMX root, run `make_pairs.py`. No script or launcher in the export consumes this manifest, so its intended run is not recorded. |
| `data/pairs_maestro_seg_manifest.csv` (919) | Short MAESTRO performance windows pseudo-labelled by the released model, one chunk each (`/root/datasets/maestro_segments/s<shard>_p<piece>_w<window>.{mid,musicxml,_chunks.json}`); columns `id, midi, mxl, chunks` only | `scripts/build_maestro_segments.py` (sharded over two GPUs) | Only with MAESTRO v3 (not shipped) and the released checkpoint; outputs are new. |
| `data/pairs_maestro_pseudo_manifest.csv` (200) | Pseudo scores of 200 whole MAESTRO performances (`/root/datasets/maestro_pseudo/*.musicxml`) tokenized as unpaired scores (`data/pairs_maestro_pseudo/`) | `scripts/pseudo_label_maestro.py` then `build_classical_unpaired.py` (chain in `scripts/build_and_train_st.sh`; the ssl_pseudo run injected them x12) | Only with MAESTRO v3. |
| `data/pairs_maestro_paired_manifest.csv` (4) | Whole-piece paired (real MAESTRO MIDI, released-model score) pairs that fit in one 512-token sequence; symlinks into MAESTRO | `scripts/build_maestro_paired.py` | Only with MAESTRO v3. |
| classical renders and cache behind `data/pairs_classical_clean_manifest.csv` | the `.mid` files (deleted by design) and `data/cache_pdmx/*.pkl` | section 10 | Yes, as a new version (section 10). |
| `pairs_scraped` cache and sources | `/root/datasets/scraped/scr_*.musicxml` (5,422 files) | `scripts/consolidate_scrape.py` from a `/tmp/scrape` tree | No: the scraped sources are not in the export. |
| MAESTRO v3.0.0 itself | 1,276 performances (MIDI and audio) | downloaded on the box (`box_logs/maestro_dl.log`, "midi=1276") | Download from the MAESTRO page (CC BY-NC-SA 4.0). Only the three benchmark performances and the metadata CSV ship. |

The three `build_maestro_*.py` and `pseudo_label_maestro.py` scripts carry a `HISTORICAL` header; their
defaults are the box paths (`/root/datasets/maestro-v3.0.0`, `/root/datasets/maestro_pseudo`).

The ten training runs with no surviving checkpoint anywhere (ssl_tuplet5, ssl_tuplet25, ssl_tuplet20e30,
ssl_combo, ssl_reshape_g2, ssl_bal_g0.5, ssl_bal_g1.0, ssl_balanced, ssl_pseudo, ssl_real07) all
trained on the classical_clean corpus or its tuplrate/pseudo variants, so reproducing any of them
requires the rebuild route first; their evaluation JSONs under `benchmark/` and curves under
`benchmark/curves/` are the surviving evidence.

## 16. Other files under `data/`

- `data/duration_priors.pt` (11.7 KB): log marginal P(duration bucket) and log P(duration | metrical
  phase) accumulated from ASAP train + validation scores only, by `scripts/compute_duration_priors.py`.
  Consumed by `benchmark/eval_padsweep.py` and the decode-lever and best-of-N / support-selection study
  scripts (`decode_lever_sweep_v9.py`, `routed_selector_v10.py`, `confirmatory_v10_1.py`,
  `oracle_prefix_v10_1.py`, `reevaluate_v10_dev_metric_v2.py`); its sha256 is pinned in
  `scripts/audit_confirmatory_v10_1.py`.
- `data/offset_phase_prior.pt` (1.8 KB): the cached offset-phase log-prior built by
  `scripts/rerank_offset.py` on first use.
- `data/pairs_april_errors.log`: see section 7.

Both `.pt` files are explicitly un-ignored in `.gitignore` (every other `*.pt` is ignored).

## 17. Cheat sheet: cache keys and path resolution

Two different key rules exist. Do not mix them up.

| Corpus family | Key | Directory | Built by |
|---|---|---|---|
| ASAP | `sha256(<data_dir>asap-dataset/<rel perf path> + "diffusion_2024_04_18")` with `data_dir = "./data/"` | `MIDI2ScoreTransformer/data/cache/` | `ASAPDataset` on first access; warm with `scripts/build_asap_cache.py` |
| Every synthetic or unpaired manifest | `sha256(<the manifest's midi string, verbatim>)` | `pathmap.cache_dir_for(midi string)`: `data/cache_pdmx_full`, `data/cache_pdmx_smoke`, `legacy/data/cache_pdmx` for the three shipped packs, else `<resolved pairs parent>/cache_pdmx` | `make_pairs.parse_and_cache`; `PDMXDataset._load_pair` on a miss |

`pathmap.resolve_path(s)` returns `s` if it exists as given, else applies the first matching prefix rule
from `MUSICML_PATH_MAP` (semicolon-separated `old=new` pairs) followed by the defaults:
`/root/Music-ML-BayenLab` and the macOS project path to the project root, `/root/datasets` and the
macOS datasets path to `<datasets root>`, and `data/pairs/` to `legacy/data/pairs/`. Strings are never
rewritten in the manifest or before hashing; only the open() path changes. `MUSICML_CACHE_MAP`
(`<resolved pairs dir>=<cache dir>`) overrides the cache directory rule.

To check a manifest resolves on a machine without training anything:

    python - <<'EOF'
    import csv, os, sys
    sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")
    from pathmap import resolve_path, cache_dir_for
    from dataset import sha256
    rows = list(csv.DictReader(open("data/pairs_gentle5k_manifest.csv")))[:3]
    for r in rows:
        midi = resolve_path(r["midi"]); pkl = cache_dir_for(r["midi"]) / (sha256(r["midi"]) + ".pkl")
        print(os.path.exists(midi), os.path.exists(pkl), midi, pkl)
    EOF

(`dataset.py` imports torch and the tokenizer, so run it in the project environment.)

## 18. File-type glossary

- `.mid`: the perturbed performance MIDI rendered by `scripts/expressive_render.py` (synthetic packs) or
  a real performance (ASAP, MAESTRO).
- `.mxl` / `.musicxml`: the engraved score, the training target. Synthetic packs copy the PDMX `.mxl`
  next to the MIDI; kern-derived packs hold a `.musicxml` converted by music21; ASAP uses
  `xml_score.musicxml`.
- `_chunks.json`: `{"midi": [[idx...] per beat], "mxl": [[idx...] per beat], "swapped": bool}`. For ASAP
  it is computed from beat annotations by `chunker.py`; for synthetic packs from the alignment
  (`midi[b] == mxl[b]`, `swapped: false`). Training crops and pads per chunk.
- `.alignment.json`: the renderer's per-note record: top-level `n_score_notes`, `n_midi_notes`,
  `n_measures`, `n_beats`, `max_ql`, and an `alignment` list with one entry per note (`midi_idx`,
  `score_idx`, `pitch`, `score_offset_ql`, `score_duration_ql`, `measure_idx`, `beat_idx`). Read only
  by the render scripts; 76 percent of the full pack's bytes.
- `<sha>.pkl`: `torch.save((input_stream, output_stream))`, the tokenizer's raw parse of MIDI and score
  before bucketing. The unit of all training I/O.
- `_manifest.csv`, `_errors.log`: written by `make_pairs.py` beside the pack; one error line per skipped
  source (`<id>\t<stage>:<message>\t<src>`).
