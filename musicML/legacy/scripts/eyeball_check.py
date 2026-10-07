"""Render N samples from the PDMX piano subset and compare timing/velocity
distributions to ASAP performance MIDI.

We're verifying that the synthetic perturbations land in the same ballpark
as real performance MIDI before scaling up to 5K pairs. Things to watch:
    - Median inter-onset interval (IOI) at similar tempo should be similar
    - Velocity range and spread should overlap
    - Duration / beat-length ratio should overlap
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
for p in (str(TOKENIZER_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from expressive_render import render


def midi_stats(path: Path) -> dict:
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted([n for inst in pm.instruments for n in inst.notes],
                   key=lambda n: (n.start, n.pitch))
    if len(notes) < 5:
        return {"n_notes": len(notes), "skip": True}
    onsets = [n.start for n in notes]
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b > a]
    durations = [n.end - n.start for n in notes]
    velocities = [n.velocity for n in notes]
    return {
        "n_notes": len(notes),
        "median_ioi_ms": round(1000 * statistics.median(iois), 1) if iois else None,
        "p10_ioi_ms": round(1000 * np.percentile(iois, 10), 1) if iois else None,
        "p90_ioi_ms": round(1000 * np.percentile(iois, 90), 1) if iois else None,
        "median_dur_ms": round(1000 * statistics.median(durations), 1),
        "p10_dur_ms": round(1000 * np.percentile(durations, 10), 1),
        "p90_dur_ms": round(1000 * np.percentile(durations, 90), 1),
        "median_velocity": int(statistics.median(velocities)),
        "min_velocity": int(min(velocities)),
        "max_velocity": int(max(velocities)),
        "song_seconds": round(notes[-1].end, 1),
    }


def find_asap_midis(asap_dir: Path, max_n: int = 5) -> list[Path]:
    """Find a few performance MIDIs in an ASAP-style layout."""
    midis = []
    if not asap_dir.exists():
        return midis
    for root, dirs, files in os.walk(asap_dir):
        for f in files:
            if f.endswith(".mid") and not f.startswith("midi_score"):
                midis.append(Path(root) / f)
                if len(midis) >= max_n:
                    return midis
    return midis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="data/pdmx_piano_subset.csv")
    ap.add_argument("--mxl-root", default="/Users/harry/datasets/pdmx/mxl_partial")
    ap.add_argument("--out-dir", default="data/render_test/eyeball")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--asap-dir", default="data/asap-dataset")
    args = ap.parse_args()

    df = pd.read_csv(args.subset)
    df = df.sample(n=min(len(df), args.n * 5), random_state=7)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for i, row in df.iterrows():
        mxl_rel = str(row["mxl"]).lstrip("./")
        src = Path(args.mxl_root) / mxl_rel
        if not src.exists():
            continue
        out = out_dir / f"sample_{len(rendered):02d}.mid"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = render(src, out, seed=42 + len(rendered))
        if not res.get("ok"):
            continue
        stats = midi_stats(out)
        stats.update({
            "src": str(src),
            "title": str(row.get("title", ""))[:60],
            "composer": str(row.get("composer_name", ""))[:30],
            "tracks": str(row.get("tracks", "")),
            "n_score_notes": int(row.get("n_notes", 0)),
            "n_bars": int(row.get("song_length.bars", 0)),
        })
        rendered.append(stats)
        if len(rendered) >= args.n:
            break

    print(f"\nRendered {len(rendered)} samples\n" + "-" * 60)
    for s in rendered:
        print(f"  {s['composer']:<25} {s['title']:<60}")
        print(f"    notes={s['n_notes']:5}  bars={s.get('n_bars', '?')}  tracks={s['tracks']}")
        print(f"    IOI(ms): med={s.get('median_ioi_ms')} p10={s.get('p10_ioi_ms')} p90={s.get('p90_ioi_ms')}")
        print(f"    dur(ms): med={s.get('median_dur_ms')} p10={s.get('p10_dur_ms')} p90={s.get('p90_dur_ms')}")
        print(f"    vel: med={s.get('median_velocity')} range=[{s.get('min_velocity')}, {s.get('max_velocity')}]")
        print()

    asap_midis = find_asap_midis(Path(args.asap_dir), max_n=5)
    if asap_midis:
        print("ASAP comparison:")
        print("-" * 60)
        for p in asap_midis:
            s = midi_stats(p)
            print(f"  {p.name}")
            print(f"    notes={s['n_notes']:5}")
            print(f"    IOI(ms): med={s.get('median_ioi_ms')} p10={s.get('p10_ioi_ms')} p90={s.get('p90_ioi_ms')}")
            print(f"    dur(ms): med={s.get('median_dur_ms')} p10={s.get('p10_dur_ms')} p90={s.get('p90_dur_ms')}")
            print(f"    vel: med={s.get('median_velocity')} range=[{s.get('min_velocity')}, {s.get('max_velocity')}]")
            print()
    else:
        # Fallback comparison: use one of our existing benchmark hFT MIDIs
        # as a proxy for "real performance MIDI distribution"
        print("ASAP not yet cloned. Comparing to existing hFT-transcribed MIDIs as proxy:")
        print("-" * 60)
        for stem in ["benchmark/chopin_op10/midi/Op10_No4_CsharpMinor_hft.mid",
                     "benchmark/chopin_op25/midi/Op25_No11_Aminor_hft.mid",
                     "benchmark/liszt_transcendental/midi/Transcendental_No4_Mazeppa_hft.mid"]:
            if Path(stem).exists():
                s = midi_stats(Path(stem))
                print(f"  {stem}")
                print(f"    notes={s['n_notes']:5}")
                print(f"    IOI(ms): med={s.get('median_ioi_ms')} p10={s.get('p10_ioi_ms')} p90={s.get('p90_ioi_ms')}")
                print(f"    vel: med={s.get('median_velocity')} range=[{s.get('min_velocity')}, {s.get('max_velocity')}]")
                print()

    out_json = out_dir / "stats.json"
    with out_json.open("w") as f:
        json.dump({"synthetic": rendered, "asap_or_proxy_count": len(asap_midis)}, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
