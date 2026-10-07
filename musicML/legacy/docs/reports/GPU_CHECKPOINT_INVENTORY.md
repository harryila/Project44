# GPU Checkpoint Inventory (2026-06-05)

Checkpoints are **not in git** (27GB total, `.gitignore`).

Repo path on the GPU box: `/root/Music-ML-BayenLab`

## Priority downloads (best models)

| Run | Checkpoint | Val loss | Notes |
|-----|-----------|----------|-------|
| **ssl_classical_clean** | `checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt` | 0.5125 | SOTA-parity baseline (Mozart/Ravel) |
| **ssl_tuplet20** | `checkpoints/ssl_tuplet20/ssl_tuplet20-epoch=01-val/total=0.5234.ckpt` | 0.5234 | Best tuplet-lever MeanER (~11.87 full-14) |
| **ssl_bigc** | `checkpoints/ssl_bigc/ssl_bigc-epoch=14-val/total=0.4975.ckpt` | 0.4975 | Best val (over-produces on MUSTER) |
| **released** | `checkpoints/MIDI2ScoreTF.ckpt` | — | Published SOTA reference |

## Other runs on box

| Run | Best ckpt | Val |
|-----|-----------|-----|
| ssl_recipe | epoch=12, total=0.5194 | 40k-step schedule (negative) |
| ssl_classical | epoch=10, total=0.5186 | Pop→classical genre fix |
| ssl_reshape_g1 | epoch=00, total=0.5322 | Tuplet-rate reshape γ=1 |
| ssl_v2 | epoch=04, total=0.5645 | Early SSL v2 |
| beat_asap_v2 | epoch=04, total=0.6739 | Beat conditioning |

## Missing (eval JSON exists, ckpt cleaned)

- `ssl_tuplet20e30` — eval in `benchmark/ssl_tuplet20e30_best.json`, checkpoint dir removed (disk cleanup)
- `ssl_tuplet25`, `ssl_combo`, `ssl_reshape_g2` — same pattern

## Training manifests (in git on branch `gpu-session-2026-06-05`)

- `data/pairs_classical_clean_manifest.csv` — 24k balanced classical SSL corpus
- `data/pairs_classical_clean_tuplrate.csv` — same + tuplet_rate column
- `data/pairs_unpaired_ssl_58k.csv` — unpaired score subset

Bulk pair directories (`data/pairs_classical/`, etc.) are regenerable via scripts; not in git.
