import sys
sys.path.insert(0, 'tokenizers')
from multihead_tokenizer import MultiHeadTokenizer
from pathlib import Path

tok = MultiHeadTokenizer()
lengths = []
errors = []

for folder in ['kern_data', 'melody_data']:
    files = list(Path(folder).glob('*.krn'))
    print(f'{folder}: {len(files)} fichiers')
    for f in files:
        try:
            score = tok.tokenize_file(f)
            lengths.append(len(score.tokens))
        except Exception as e:
            errors.append((str(f), str(e)))

lengths.sort()
n = len(lengths)
total = sum(lengths)
print("\n=== METRIQUES ===")
print(f"Fichiers tokenises : {n}")
print(f"Erreurs            : {len(errors)}")
print(f"Total tokens       : {total:,}")
print(f"Moyenne            : {total // n if n else 0}")
print(f"Mediane (p50)      : {lengths[n // 2]}")
print(f"p75                : {lengths[int(n * 0.75)]}")
print(f"p90                : {lengths[int(n * 0.90)]}")
print(f"p95                : {lengths[int(n * 0.95)]}")
print(f"p99                : {lengths[int(n * 0.99)]}")
print(f"Max                : {lengths[-1]}")
print(f"Min                : {lengths[0]}")

buckets = [0, 128, 256, 512, 1024, 2048, 4096, 99999]
labels = ['<128', '128-256', '256-512', '512-1024', '1024-2048', '2048-4096', '>4096']
counts = [0] * len(labels)
for l in lengths:
    for i, (lo, hi) in enumerate(zip(buckets, buckets[1:])):
        if lo <= l < hi:
            counts[i] += 1
            break
print('\nDistribution:')
mx = max(counts) if counts else 1
for lbl, cnt in zip(labels, counts):
    bar = '#' * (cnt * 40 // mx)
    pct = cnt * 100 // n if n else 0
    print(f'  {lbl:12s} {cnt:5d} ({pct:2d}%)  {bar}')

if errors[:5]:
    print('\nPremieres erreurs:')
    for p, e in errors[:5]:
        print(f'  {p}: {e}')
