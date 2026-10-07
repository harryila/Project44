"""Track-B measurement: sweep hFT detection thresholds on a piece and score each
config's end-to-end output with MUSTER, to test whether lowering recall thresholds
dents the MissRate on dense repertoire. Honest before/after — keep only what moves.

Runs transcribe.py (real pipeline) with --no-render for speed, then MUSTER vs GT.
"""
import json, subprocess, sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import REPO_ROOT as _ROOT
REPO = Path(_ROOT)
sys.path.insert(0, str(REPO / "MIDI2ScoreTransformer/midi2scoretransformer"))
from muster import muster  # noqa: E402

PIECE = {
    "name": "Liszt_Mazeppa",
    "audio": REPO / "benchmark/liszt_transcendental/audio/Transcendental_No4_Mazeppa.wav",
    "gt": REPO / "benchmark/liszt_transcendental/gt_score.musicxml",  # 2-staff
}
# (label, extra transcribe.py args). Baseline already measured (MeanER 34.23) but
# re-run here for an apples-to-apples comparison under identical conditions.
CONFIGS = [
    ("baseline_0.50", []),
    ("onset_mpe_0.35", ["--hft-onset-threshold", "0.35", "--hft-mpe-threshold", "0.35"]),
    ("onset_mpe_0.25", ["--hft-onset-threshold", "0.25", "--hft-mpe-threshold", "0.25"]),
    ("offset_mode", ["--hft-offset-mode", "offset"]),
]

work = REPO / "benchmark/trackb_sweep_out"
work.mkdir(parents=True, exist_ok=True)
out_json = REPO / "benchmark/trackb_mazeppa_sweep.json"

results = []
t0 = time.time()
for label, extra in CONFIGS:
    pdf = work / f"{PIECE['name']}__{label}.pdf"
    mxl = pdf.with_suffix(".musicxml")
    ti = time.time()
    print(f"[{label}] transcribing ...", flush=True)
    cmd = [str(REPO / "venv311/bin/python"), str(REPO / "transcribe.py"),
           str(PIECE["audio"]), "-t", "hft", "-b", "transformer",
           "--no-render", "-o", str(pdf)] + extra
    subprocess.run(cmd, cwd=str(REPO), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rec = {"label": label, "args": extra, "transcribe_s": round(time.time() - ti, 1)}
    if mxl.exists():
        try:
            m = muster(str(mxl), str(PIECE["gt"]))
            rec["muster"] = m
            print(f"[{label}] MeanER={m.get('MeanER')} MissRate={m.get('MissRate')} "
                  f"ExtraRate={m.get('ExtraRate')} PitchER={m.get('PitchER')} ({rec['transcribe_s']}s)", flush=True)
        except Exception as e:
            rec["error"] = f"score: {e}"; print(f"[{label}] score error: {e}", flush=True)
    else:
        rec["error"] = "no musicxml produced"; print(f"[{label}] FAILED: no musicxml", flush=True)
    results.append(rec)
    json.dump({"piece": PIECE["name"], "results": results}, open(out_json, "w"), indent=2)

print("\n=== Track-B hFT-threshold sweep on Mazeppa (vs 2-staff GT) ===")
print(f"{'config':18s} {'MeanER':>7s} {'MissRate':>8s} {'ExtraRate':>9s} {'OnsetER':>7s} {'PitchER':>7s}")
for r in results:
    m = r.get("muster") or {}
    def f(k):
        v = m.get(k); return f"{v:7.2f}" if isinstance(v, (int, float)) else "    n/a"
    print(f"{r['label']:18s} {f('MeanER')} {f('MissRate')} {f('ExtraRate')} {f('OnsetER')} {f('PitchER')}")
print(f"\n{round(time.time()-t0,1)}s total -> {out_json}")
