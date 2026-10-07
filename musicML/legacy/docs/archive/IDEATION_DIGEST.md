## LENS: ssl-representation: make the masked-SSL that already won smarter (better surrogate input, curriculum masking, auxiliary structure heads, difficulty-aware conditioning/dropout) — all as concrete edits 
- [medium | ~1 training run, same length as ssl_classical_clean (~12-16 ] Curriculum masking of the unpaired surrogate: reveal quantized onset early, anneal to pitch-only
- [medium | 1-2 training runs (~6-10 A100-hours each). Code change is ~1] Auxiliary tuplet/beat-phase prediction head trained ONLY from the existing offset stream (no new labels, no architecture change at inference)
- [medium | 1 training run (~6-10 A100-hours) + a few CPU-cheap inferenc] Difficulty-aware conditioning token + per-sample decoder dropout (turn the binary c_i into a graded density signal)
- [high | Essentially FREE — no training. A handful of CPU/GPU validat] Checkpoint selection by a MUSTER-proxy (over-production calibration metric) instead of val/total — close the val!=MUSTER gap with zero new training
- [medium | 1 training run (~6-10 A100-hours) + one full 14-piece eval p] Denoising-SSL variant: corrupt the surrogate with realistic performance noise so the encoder learns to be robust to dense fast passages (and measure the full 14-piece standing)
## LENS: data-ceiling: break the 822-real-aligned-pair ceiling without hand-aligning, via self-training/pseudo-labels and renderer-minted exactly-aligned pairs, with strict confidence filtering and MUSTER-base
- [medium | ~6-10 GPU-hours pseudo-labelling (200hr / ~hundreds of piece] Noisy-student self-training: pseudo-label MAESTRO performances with ssl_classical_clean, keep only high-confidence pairs, fold into the REAL branch
- [low | ~3-5 GPU-hours rendering (CPU-bound, use the 252-core box) +] Renderer-minted EXACTLY-aligned pairs at a SMALL fraction in the PAIRED branch (alignment signal, domain gap subordinated)
- [high | NONE if epochs are on disk (pure inference, ~few GPU-hours o] MUSTER-selected checkpointing + over-production calibration sweep (turn val!=MUSTER into a free gain, zero training)
- [high | NONE (pure inference/eval) — ~a few GPU-hours plus handling ] Measure all 14 ASAP test pieces to establish true standing and locate the real residual (zero training, prerequisite for everything)
- [speculative | Incremental over proposal 1: ~2-3 extra GPU-hours (stratific] Confidence/difficulty-stratified pseudo-label curriculum: harvest MAESTRO by difficulty band to specifically attack the dense-passage miss-rate without amplifying error
## LENS: corpus-balance: fix OVER-PRODUCTION by corpus COMPOSITION (note-density / tuplet-rate distribution matching, curriculum, per-sample exposure weighting), not by adding more data
- [high | ~1 GPU-hour for the reshape+caching is already done (reuses ] Density-matched unpaired corpus: reshape (not grow) the SSL pool to the ASAP-test note-density histogram
- [high | Offline tuplet-rate pass: ~1-2 CPU-hours on the 252-core box] Tuplet-rate balancing: bucket the unpaired corpus by actual tuplet token fraction and hold it to the test rate
- [medium | Same total step budget as one ssl_classical_clean run (~6-10] Simple->dense curriculum on the unpaired stream (sort by density, anneal the cap)
- [high | Code: ~20 lines. Each gamma run ~6-10 A100-hours; a 3-point ] Per-sample inverse-density loss/exposure weighting in the WeightedRandomSampler (1-line lever)
- [high | Zero GPU training; pure inference + eval. ~2-4 GPU-hours of ] Over-production-aware checkpoint selection: evaluate all 14 test pieces + add a calibration metric, since val!=MUSTER
## LENS: inference-decoding
- [high | ~0 GPU-hours of training; ~10-20 min/piece x 6 thresholds x ] Expose the continuous pad-sigmoid in generate() and sweep the REAL note-keep threshold (the current --pad-threshold is a no-op)
- [medium | ~0 training. Inference sweep ~4-8 GPU-hours for the grid x 5] Per-head decoding: greedy pad/duration, low-temperature pitch — decouple the globally-shared top_k/temperature
- [medium | ~0 training. 3x inference cost = ~9-15 GPU-hours for the eva] Overlap-window voting / self-ensemble using the existing chunked AR loop to stabilize dense-passage notes
- [high | ~0 training. ~14 pieces x 3 thresholds x 2 models x ~15 min ] Measure the full 14-piece ASAP test split at multiple thresholds to find our true standing and the over-production frontier
- [speculative | ~0 training. Inference sweep ~6-10 GPU-hours for the small g] Beat-phase-consistent offset decoding: nudge the offset head toward the input MIDI's own onset grid on dense passages (speculative)
## LENS: literature-contrarian (2023-2026 piano AMT / MIDI-to-score / symbolic-music / rhythm-quantization literature, mapped onto the ssl_classical_clean asset stack)
- [high | Stage 1-2: none (CPU/box inference, minutes-hours). Stage 3:] Self-distilled pad-head recalibration: tune the per-note keep-gate against MUSTER, not val-loss
- [medium | none beyond N× inference (N=5-9; minutes per piece on A100).] Multi-sample self-consistency decode with note-level majority voting (free recall on dense passages)
- [medium | ~20-40 GPU-hours (sampling + a few hundred DPO/KTO steps; pr] DPO/KTO preference fine-tune against a rule-based over-production / miss penalty (no new aligned data)
- [medium | none (pure CPU post-processing).] Post-hoc notational-grammar corrector: a deterministic state-machine repair pass on the decoded score
- [speculative | Path A: ~none (inference + merge). Path B: ~10-20 GPU-hours.] Beat/downbeat-conditioned rhythm refiner as a teacher OR second-stage cleanup (the only published MUSTER-SOTA-beating method we haven't tried)