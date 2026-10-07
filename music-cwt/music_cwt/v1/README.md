# music_cwt — Compound Word Transformer for Piano Generation

A 10.2M-parameter autoregressive transformer that generates piano music in **kern format from a 12-head compound token representation. Trained on ~218k kern files (Chopin, Liszt, Schubert, Bach, etc.), then fine-tuned on high-quality Humdrum-encoded scores.

---

## Architecture

**Model**: Compound Word Transformer (CWT)  
Each musical event is encoded as a single **compound token** with 12 simultaneous heads, processed in two stages:

| Stage | Heads predicted |
|-------|----------------|
| 1 | H1 (Type) + H11 (Spine: RH/LH) |
| 2 | H2–H10 + H12, conditioned on Stage 1 output |

**12 heads:**

| Head | Name | Values (examples) |
|------|------|-------------------|
| H1 | Type | Note_Single, Chord_Start, Chord_Cont, Chord_End, Rest, Barline, Fioritura_Q |
| H2 | Duration | 4, 8, 16, 4., 8., q, ... |
| H3 | Pitch | c, cc, G#, e--, r, ... |
| H4 | Ornament | T, t, TR, M, W, S, ... |
| H5 | Articulation | ', ^, ~, `, I |
| H6 | Tie | Tie_Start, Tie_Continue, Tie_End |
| H7 | Slur | Slur_Start, Slur_End |
| H8 | Phrase | Phrase_Start, Phrase_End, &Phrase_Start, &Phrase_End |
| H9 | Voicing | /, \ |
| H10 | Beam | L, LL, J, JJ, K, k, ... |
| H11 | Spine | RH, LH |
| H12 | Dynamic | pp, p, mp, mf, f, ff, sfz, <, >, ... |

**Config:**

| Param | Value |
|-------|-------|
| d_model | 256 |
| n_layers | 9 |
| n_attn_heads | 8 |
| d_ff | 1536 |
| max_seq_len | 2048 |

---

## File Structure

```
music_cwt/
  tokenizers/
    multihead_tokenizer.py        # 12-head tokenizer: kern -> CompoundToken sequence
    tokenization_rules_reviewed.md  # authoritative spec for all 12 heads
  checkpoints/                    # .pt files downloaded from training server
  kern_data/                      # ~90 local .krn files (Chopin/Liszt subset for testing)
  melody_data/                    # melody-only kern files (unused in CWT training)
  generated/                      # output .krn and .mxl files from generate.py
  prompt/                         # MusicXML prompt files for OOS tests
  cwt_model.py                    # CompoundWordTransformer architecture
  train_cwt.py                    # pre-training loop (full corpus)
  finetune_humdrum.py             # fine-tuning on Humdrum-quality files only
  generate.py                     # inference: checkpoint + optional prompt -> .krn
  gen_and_convert.py              # generate .krn + convert to .mxl in one call (use this on Windows)
  postprocess.py                  # CompoundToken sequence -> valid **kern text
  summarize_logs.py               # parse train.log and show per-epoch loss table
  archi_model.md                  # detailed architecture design notes
  tokenization_rules_reviewed.md  # local copy of tokenization spec
```

---

## Training

Training runs on the Polytechnique server (`angleterre`, RTX A4000 16GB). The local `music_cwt/` folder maps to `~/taff/` on the server.

### Pre-training (full corpus)
```bash
# On server, in tmux
cd ~/taff
python3 -u train_cwt.py --kern-dir kern_data --checkpoint-dir checkpoints 2>&1 | tee train.log
```

### Fine-tuning (Humdrum files only)
```bash
python3 -u finetune_humdrum.py --kern-dir kern_data --checkpoint-dir checkpoints 2>&1 | tee finetune.log
```
Fine-tuning filters files with `!!!COM:` header (manually encoded, higher quality). Uses lr=1e-4 vs 3e-4 for pre-training.

**Hyperparams:** batch=4, grad_accum=8 (effective=32), ~45 min/epoch.  
Both scripts auto-resume from the latest valid checkpoint.

### Training status
| Phase | Epoch | Train loss | Val loss |
|-------|-------|-----------|---------|
| Pre-training | ep10 | 0.0990 | 0.0974 |
| Pre-training | ep25 | 0.0896 | 0.0906 |
| Pre-training | ep43 | 0.0853 | 0.0873 |
| Pre-training | ep50 | ~0.084 | ~0.087 |
| Fine-tuning | ft50 | — | — |

---

## Generation (Windows)

**Python to use:** `<your Python 3.12 with torch + music21>` (has torch + music21)

Always use `gen_and_convert.py` on Windows — it runs generation and MXL conversion in a single process, avoiding PowerShell sandbox issues.

```bash
# From kern prompt (recommended)
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step349566.pt \
  --output generated/out.krn \
  --prompt-krn kern_data/Chopin_nocturne72-1.krn \
  --prompt-measures 6 \
  --temperature 0.8 --top-k 30

# From MusicXML prompt (OOS test)
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step349566.pt \
  --output generated/oos_test.krn \
  --prompt-krn prompt/009_1et2-1a-W-002.musicxml \
  --prompt-measures 6 \
  --temperature 0.8 --top-k 30

# Cold generation (no prompt)
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step349566.pt \
  --output generated/cold.krn \
  --key K_0_maj --time M_3/4 --tempo MM_72
```

Key parameters:
- `--temperature` — 0.8 is more conservative, 1.0 is default. Lower = less repetitive.
- `--top-k` — 30 gives more focused sampling than default 50.
- `--prompt-measures` — number of measures to take as prompt (default 6).
- `--max-tokens` — tokens to generate after prompt (default 1200 ≈ 30-40 measures).

The `.mxl` output is placed next to the `.krn` file and can be opened directly in MuseScore.

---

## Generation Quality

Model prompted on the seven first measures of Chopin's Nocturne Op9 No1.

| Checkpoint | Prompt | Note_Single | Chords | Subjective quality |
|------------|--------|------------|--------|--------------------|
| ep4 | cold | 94% | 0% | Incoherent |
| ep8 | cold | 86% | 9% | Some chords appear |
| ep25 | cold | 84% | 11% | Correct chords |
| ep43 | cold | 79% | 16% | Repetitive, too many 32nd notes |
| ft50 | Nocturne 72 (kern) | 94% | 1% | Better melodically, sparse texture |
| ft50 | Nocturne op.9 n°2 (MXL, temp=0.8) | 41% | 52% | **Best so far** — musical, repeats themes with variation |

**Best settings:** `temperature=0.8, top-k=30` with a rich 2-hand MusicXML prompt.

### What works
- Repeats and varies musical phrases heard in the prompt — short-range musical memory is working
- With a rich prompt (both hands, chords), generates convincing harmonic texture
- Key/time signature correctly inferred from MXL prompt

### Known issues

**1. Incoherent rests** — silences appear at musically wrong moments  
Suspected cause: degenerate files in the PDMX training corpus with randomly placed rests from bad MXL conversion. **To investigate:** scan `pdmx_data/kern/` for files with >30% rest tokens.

**2. Repetition loops** — model gets stuck repeating the exact same pattern for many bars  
Suspected cause: PDMX corpus contains MIDI-loop-based arrangements where the same bars repeat verbatim. **To investigate:** detect files with >3 consecutive identical measures.

**3. LH always 0%** — left hand never generated  
Root cause identified: `pdmx_scrapper/mxl_to_kern.py` line 84 always assigns `parts[0]` as RH without checking the clef. For files where `parts[0]` is bass, RH/LH labels are swapped across all 213k converted files. The model learned a confused H11 (spine) representation.  
Fix: detect `isinstance(clef, BassClef)` and swap — already corrected in `generate.py`'s `_mxl_to_kern()`.

**4. Theme drift after ~20 bars** — long-range coherence degrades  
Cause: attention dilutes over long sequences. Would require architectural changes (RoPE) or longer training.

---

## Next Steps

| Priority | Task | Effort |
|----------|------|--------|
| High | Write diagnostic script: scan PDMX kern files for degenerate data (>30% rests, repeated measures) | 1h |
| High | Fix spine swap in `pdmx_scrapper/mxl_to_kern.py` + reconvert 213k files | 2h + 1 day GPU |
| High | Retrain from scratch on cleaned data | 3 days GPU |
| Medium | Add repetition penalty in `generate.py` | 1h |
| Low | RoPE positional encoding for better long-range coherence | 1 day |

---

## SSH Commands

```bash
# From WSL — always use /mnt/c/... for Windows paths
ssh poly-albatros          # gateway
ssh angleterre             # GPU machine (RTX A4000)

# Upload a file to server
scp <local copy of taff>/music_cwt/<file> poly-albatros:~/taff/<file>

# Download a checkpoint
scp poly-angleterre:~/taff/checkpoints/<ckpt>.pt \
  <local copy of taff>/music_cwt/checkpoints/
```

---

## Dependencies

```
torch >= 2.0
music21 >= 9.0
```

Python: `<your Python 3.12 with torch + music21>`
