"""Content-fingerprint dedup of PDMX against the evaluation pieces.

Required before any PDMX training (see benchmark/LEAKAGE_AUDIT.md): metadata
blocklists miss alt-title copies (Liszt "Gondoliera" -> "Venezia e Napoli") and
embedded movements (Ravel "Ondine" inside full "Gaspard"). This catches them by
content.

Method (validated by the leakage audit's adversarial pass):
  1. For every eval ground-truth score, extract a TIME-SORTED pitch sequence
     (music21 flatten + sort by offset — NOT XML document order).
  2. Build transposition-invariant interval 5-grams (a "fingerprint").
  3. For each PDMX candidate (restricted to the eval composers — only those can be
     copies), compute containment = |needle ∩ candidate| / |needle|.
  4. Flag a leak when containment > 0.4 (positive controls: true leaks 0.78-0.92,
     unrelated same-composer < 0.10).

Output: data/pdmx_eval_leak_blocklist.csv (mxl paths to drop before training) +
a printed report. Run from the repo root.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import REPO_ROOT as _ROOT
REPO = Path(_ROOT)
sys.path.insert(0, str(REPO / "MIDI2ScoreTransformer/midi2scoretransformer"))
os.chdir(REPO)
import pandas as pd  # noqa: E402
import music21  # noqa: E402

ASAP = REPO / "MIDI2ScoreTransformer/data/asap-dataset"
PDMX_ROOT = Path(os.path.expanduser("~/datasets/pdmx"))
EVAL_COMPOSERS = ["bach", "beethoven", "brahms", "chopin", "debussy", "haydn",
                  "liszt", "mozart", "prokofiev", "rachmanin", "ravel", "schubert",
                  "schumann", "scriabin"]
THRESHOLD = 0.4
N = 5  # n-gram size


def fingerprint(path):
    """Time-sorted, transposition-invariant interval-N-gram set."""
    s = music21.converter.parse(str(path))
    notes = []
    for el in s.flatten().notes:
        off = float(el.offset)
        if el.isChord:
            for p in el.pitches:
                notes.append((off, p.midi))
        else:
            notes.append((off, el.pitch.midi))
    notes.sort(key=lambda x: (x[0], x[1]))
    pitches = [p for _, p in notes]
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    return set(tuple(intervals[i:i + N]) for i in range(len(intervals) - N + 1))


def build_needles():
    """17 eval references: 14 ASAP test pieces + 3 benchmark pieces."""
    from dataset import ASAPDataset
    refs = {}
    q = ASAPDataset(str(REPO / "MIDI2ScoreTransformer/data") + "/", "test")
    md = q.metadata
    seen = set()
    for i in range(len(md)):
        s = md.iloc[i]
        midi = s["performance_MIDI_external"].replace("{ASAP}", str(ASAP))
        score = os.path.dirname(midi) + "/xml_score.musicxml"
        if score in seen:
            continue
        seen.add(score)
        name = f"ASAP:{s['composer']}:{os.path.relpath(os.path.dirname(midi), './data/asap-dataset')}"
        refs[name] = score
    refs["BENCH:Chopin_Op10_No4"] = str(ASAP / "Chopin/Etudes_op_10/4/xml_score.musicxml")
    refs["BENCH:Chopin_Op25_No11"] = str(ASAP / "Chopin/Etudes_op_25/11/xml_score.musicxml")
    refs["BENCH:Liszt_Mazeppa"] = str(REPO / "benchmark/liszt_transcendental/gt_score.3staff.raw.musicxml")

    needles = {}
    for name, path in refs.items():
        try:
            fp = fingerprint(path)
            needles[name] = fp
            print(f"  needle {name[:48]:48s} {len(fp)} grams", flush=True)
        except Exception as e:
            print(f"  needle FAILED {name}: {e}", flush=True)
    return needles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-csv", default="data/pdmx_piano_subset.csv",
                    help="PDMX pool to dedup (the training pool).")
    ap.add_argument("--out", default="data/pdmx_eval_leak_blocklist.csv")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--report-above", type=float, default=0.2,
                    help="also record borderline candidates above this for inspection")
    args = ap.parse_args()

    print("Building eval fingerprints (needles)...")
    needles = build_needles()
    print(f"{len(needles)} needles built.\n")

    df = pd.read_csv(args.subset_csv, low_memory=False)
    comp = df["composer_name"].fillna("").str.lower()
    cand = df[comp.apply(lambda c: any(e in c for e in EVAL_COMPOSERS))].copy()
    print(f"Candidates (eval-composer piano rows): {len(cand)}\n")

    rows = []
    t0 = time.time()
    n_parsed = n_fail = 0
    for idx, (_, r) in enumerate(cand.iterrows()):
        mxl_rel = r["mxl"]
        mxl_abs = PDMX_ROOT / mxl_rel.lstrip("./")
        try:
            fp = fingerprint(mxl_abs)
            n_parsed += 1
        except Exception:
            n_fail += 1
            continue
        if not fp:
            continue
        best_c, best_eval = 0.0, None
        for name, needle in needles.items():
            if not needle:
                continue
            c = len(needle & fp) / len(needle)
            if c > best_c:
                best_c, best_eval = c, name
        if best_c >= args.report_above:
            rows.append({"mxl": mxl_rel, "composer": r.get("composer_name"),
                         "title": r.get("title"), "n_notes": r.get("n_notes"),
                         "containment": round(best_c, 4), "matched_eval": best_eval,
                         "is_leak": bool(best_c >= args.threshold)})
        if (idx + 1) % 100 == 0:
            n_leak = sum(1 for x in rows if x["is_leak"])
            print(f"  [{idx+1}/{len(cand)}] parsed={n_parsed} fail={n_fail} "
                  f"flagged={len(rows)} leaks={n_leak} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    out = pd.DataFrame(rows).sort_values("containment", ascending=False) if rows else pd.DataFrame(
        columns=["mxl", "composer", "title", "n_notes", "containment", "matched_eval", "is_leak"])
    out.to_csv(args.out, index=False)

    leaks = out[out["is_leak"]] if len(out) else out
    print("\n" + "=" * 78)
    print(f"CONTENT DEDUP — {n_parsed} parsed, {n_fail} parse-fail, {time.time()-t0:.0f}s")
    print(f"LEAKS (containment > {args.threshold}): {len(leaks)} PDMX rows to drop before training")
    print("=" * 78)
    for _, r in leaks.head(40).iterrows():
        print(f"  {r['containment']:.3f}  {str(r['matched_eval'])[:32]:32s} {str(r['title'])[:38]}")
    # per-eval coverage
    if len(leaks):
        print("\nEval pieces with >=1 content-leak found:")
        for name in sorted(set(leaks["matched_eval"])):
            n = (leaks["matched_eval"] == name).sum()
            print(f"  {name[:50]:50s} {n} copies")
    print(f"\nBorderline (0.2-{args.threshold}, inspect): {len(out)-len(leaks)}")
    print(f"Blocklist written: {args.out}")


if __name__ == "__main__":
    main()
