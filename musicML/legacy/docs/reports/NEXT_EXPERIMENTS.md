# Next experiments — the box-ready queue to push *past* SOTA parity

Ranked from a 25-proposal ideation pass in which each proposal's concrete code/data claims were checked
against the actual repo + the tested-negative ledger (see `legacy/docs/archive/TRIAGE_RESULT.json` for the
full machine-readable plan). The proven baseline
is `ssl_classical_clean` ep13 (Mozart 5.76 / Ravel 20.51 = parity; Scriabin 12.3 vs released 10.37 = residual).

**One rule from the synthesis:** *nothing is retrained until the inference/measurement map exists.* The cheap
levers produce the per-piece calibration frontier that tells every training run where to aim.

---

## DO NOW — inference-only, zero retrain (run the moment the box is up)

### 1. Pad-threshold sweep on all 14 ASAP pieces  ← **highest EV in the entire backlog; already implemented**
- One `run_padsweep.sh` invocation does three jobs: sweeps the now-live keep-gate {0.60…0.25} (lower → rescues
  Scriabin's dropped notes; higher → prunes Mozart's over-production), **measures all 14 test pieces** (closes
  the "4–6 of 14 measured" gap), and yields the per-piece calibration map every training run needs.
- Zero risk: 0.50 reproduces released behavior byte-for-byte (locally validated). See `PAD_THRESHOLD_FIX.md`.
- Run: `bash scripts/run_padsweep.sh 'checkpoints/ssl_classical_clean/*epoch=13*.ckpt' sslcc`, then A/B the released ckpt.
- *Merges 4 proposals* (2× pad-sweep, 2× full-14-piece-measurement) + the implemented pad-exposure item.
- Cost ~3–6 GPU-hr (×2 for the released A/B). Attacks: scriabin-missrate, mozart-overproduction, val≠MUSTER, unmeasured-pieces.

### 2. MUSTER-based checkpoint reselection across saved epochs
- val ≠ MUSTER is **proven** (ssl_bigc had the best val 0.4996 but worse MUSTER via Mozart 166-vs-36 tuplet
  over-production). ep13 was picked by val; an earlier, less-over-producing epoch may be MUSTER-better.
- Score every saved `ssl_classical_clean` / `ssl_bigc` epoch through the same 14-piece harness *at its own best
  threshold*; add note-count-ratio + tuplet-emission-ratio as an over-production proxy; pick the 14-piece-MUSTER min.
- Free win if an earlier epoch wins; worst case it validates ep13 and gives a cheap surrogate for future checkpointing.
- Run: `for c in checkpoints/ssl_classical_clean/*epoch=*.ckpt; do bash scripts/run_padsweep.sh "$c" "sslcc_$(basename $c)"; done`
- *Merges 3 reselection proposals.* Cost ~3–6 GPU-hr.

### 3. Per-head decoding sweep (greedy structure, mild temp/top-k on pitch)
- Greedy argmax on the **pitch** head collapses notes in dense polyphony (Scriabin); the pad/duration/offset
  heads must stay greedy to keep Mozart's rhythm spine and avoid over-production. Decouple the globally-shared
  `top_k`/`temperature` into per-head overrides.
- Needs a small code add (`head_overrides` threaded through `generate()`/`infer()`; default `None` = identical).
  **→ implemented + locally validated in these tests** (`scripts/test_perhead_decode.py`: `None` == explicit greedy
  on every stream; live when set). `eval_tuplet.py` now takes `--pitch-top-k` / `--pitch-temp`. Staged for the box.
- Run *after* the pad sweep, only on the miss-rate the pad-gate couldn't reach (don't burn the grid twice). E.g.:
  `for tk in 2 3 5; do for t in 1.0 1.1; do venv311/bin/python benchmark/eval_tuplet.py --ckpt <ckpt> --device cuda --pitch-top-k $tk --pitch-temp $t --limit-per 1 --out benchmark/perhead_k${tk}_t${t}.json; done; done`
- Cost ~4–8 GPU-hr. Attacks: scriabin-missrate, pitch-error.

---

## QUEUE — single retrains, ranked by EV, de-risked against the tested-negative ledger
*(Run only after the DO-NOW map; each is `needs_box`.)*

1. **Density / tuplet-rate RESHAPE of the existing unpaired corpus** — keep corpus *size* fixed (growth is what
   turned ssl_bigc negative) and reshape composition toward the ASAP-test density histogram. Prefer the **soft
   per-sample inverse-density `WeightedRandomSampler`** (~20 lines, `gamma=0` = byte-identical control) over
   manifest surgery. The cleanest single-knob test of the proven over-production failure mode.
   ⚠ *Blocker:* `pairs_classical_clean_manifest.csv` is **not present locally** — use `pairs_deduped_full`/
   `_manifest.csv` (52k rows, has `n_notes`/`n_measures`) or regenerate via `build_classical_clean_manifest.py`.
2. **Denoising-SSL** — corrupt the pitch-only surrogate with performance-like noise (density-scaled dropout /
   insertion / onset-jitter; `onset_jitter` already exists in `pdmx_dataset.py`) and reconstruct the clean
   score. Makes the model *practice* dense-passage recovery **without adding dense data** (sidesteps the ssl_bigc trap).
3. **Curriculum masking** — reveal quantized onset/duration early (the 1/24 grid encodes tuplet positions),
   anneal to pitch-only by ~60% of training. Inference path stays pitch-only, so no train/test surrogate mismatch.
4. **Auxiliary tuplet-bucket loss reweighting** — upweight non-dyadic offset buckets in the existing CE; no
   data-distribution change, so it can't re-trigger over-production. Moderate upside (tuplet tail is field-hard).
5. **Simple→dense exposure curriculum** — phase the corpus in by density (order, not amount). Below #1 because
   reshape is a cleaner single-knob test of the same composition hypothesis.

## QUEUE-LOW / BLOCKED
- **Noisy-student self-training on MAESTRO** (pseudo-PAIRS into the paired branch, ~3k:822, confidence-gated) —
  the principled way to break the 822-pair ceiling, **but blocked**: verify MAESTRO MIDIs are on the box and
  that per-stream generation confidences are dumpable before committing 15–20 GPU-hr. The most *product-relevant*
  lever (data is what separates an academic reproduction from a shipping product) — promote once unblocked.
- Difficulty-aware continuous conditioning token; DPO/KTO preference tuning; overlap-window voting; CPU post-hoc
  grammar-repair (box-free but confounded by the now-live pad-gate).

## SKIP — adversarially disqualified (don't spend GPU)
- **Multi-sample self-consistency voting** — *architecturally unsound*: output length is set by AR pad-stopping,
  not input length, so stochastic samples have varying lengths; the "trivially slot-aligned by input-note index"
  premise is false (would need DTW/Hungarian, not the claimed 40-line aggregator). The useful part (pad_prob
  voting) is already subsumed by the live threshold sweep.
- **Renderer-minted synthetic pairs at ≤5%** — synthetic-render already disproven (ar_full4/5 regressed,
  ar_rubato diverged) even below the proposed fraction; same domain-gap/over-production failure mode.
- **Beat-phase-consistent offset decoding** & **beat/downbeat rhythm refiner** — beat-conditioning already tested
  neutral-to-destructive; beat-alignment already present in the ASAP chunker; targets a secondary effect
  (Scriabin's residual is missing tuplet *production* ~0/99, not offset-bucket timing).
