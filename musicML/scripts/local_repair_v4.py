"""PAPER-A experiment: V4 LOCAL REPAIR (pre-registered in paper/bestofn_prereg.md, addendum).

Whole-sequence selection (V1-V3b) is dead: it rewards symbolic cleanliness while the real
gain in the candidate pool is COVERAGE in hard passages (cand02 = fewer MISSED tuplets, not
fewer onset errors). V4 therefore repairs locally: keep the argmax decode everywhere, detect
hard windows (donor disagreement + timing evidence), and splice in the donor slice that wins
a coverage-aware local score. The note-aligned 1:1 seq2seq makes splicing exact.

Frozen design (see prereg addendum): windows from d_i>=0.25 or e_i, pad +/-2, min 3, merge <4;
variants = donor (offset, duration, pad) slices, downbeat stays baseline; local score =
2.0 * kept-fraction + 1.0 * timing-fit; w_lik=0; drift gate MeanER <= baseline + 0.5.

Run (dev):
  venv311/bin/python scripts/local_repair_v4.py --pieces Scriabin --n 8 \
      --out benchmark/local_repair_v4_dev.json
Run (held-out, frozen):
  venv311/bin/python scripts/local_repair_v4.py --pieces Ravel Liszt Mozart --n 32 \
      --out benchmark/local_repair_v4_heldout.json
"""
import argparse, json, time
from pathlib import Path

import verifier_bestofn as VB  # reuses env setup (CPU/FP32/chdir), decode, muster, MTD
import numpy as np
import torch

N_PHASE = 24
THETA = 0.02
DUPLE = np.array([0.0, 0.25, 0.5, 0.75])
TRIPLE = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
W_COV, W_FIT, W_LIK = 2.0, 1.0, 0.0


def streams_np(y):
    return {
        "off": y["offset"].argmax(-1).reshape(-1).cpu().numpy(),
        "dur": y["duration"].argmax(-1).reshape(-1).cpu().numpy(),
        "db": y["downbeat"].argmax(-1).reshape(-1).cpu().numpy(),
        "pad": (y.get("pad_prob", y["pad"]).reshape(-1).cpu().numpy() > 0.5),
    }


def quarter_anchor_fit(off, db, kept, t, lo, hi):
    """Timing-fit of decoded phases vs input onsets inside [lo, hi): negative mean nearest-
    grid distance of the normalized onsets under the decode's own duple/triple label. Uses
    the same quarter anchoring as V3b. Returns 0.0 when no informative group exists."""
    n = len(off)
    measure = np.cumsum(db > 0)
    quarter = off // N_PHASE
    phase = off % N_PHASE
    gkey = measure * 1000 + quarter
    kept_idx = np.flatnonzero(kept)
    g_first = {}
    for g in np.unique(gkey[kept_idx]):
        idx = kept_idx[gkey[kept_idx] == g]
        g_first[int(g)] = float(t[idx].min())
    fits, weights = [], []
    in_win = kept_idx[(kept_idx >= lo) & (kept_idx < hi)]
    for g in np.unique(gkey[in_win]):
        idx = kept_idx[gkey[kept_idx] == g]   # whole quarter group, anchored properly
        if len(idx) == 0:
            continue
        anchor = g_first.get(int(g) + 1)
        if anchor is None:
            m_ = int(g) // 1000
            anchor = g_first.get((m_ + 1) * 1000)
        t0 = float(t[idx].min())
        if anchor is None or anchor <= t0:
            continue
        ph_u, t_u = [], []
        for p_ in np.unique(phase[idx]):
            sel = idx[phase[idx] == p_]
            ph_u.append(float(p_)); t_u.append(float(t[sel].min()))
        if len(ph_u) < 2:
            continue
        u = (np.array(sorted(t_u)) - t0) / (anchor - t0)
        u = u[(u >= -0.05) & (u < 1.05)]
        if len(u) < 2:
            continue
        says_triple = bool(np.any((np.array(ph_u) % 3) != 0))
        grid = TRIPLE if says_triple else DUPLE
        err = float(np.mean(np.min(np.abs(u[:, None] - grid[None, :]), axis=1)))
        fits.append(-err); weights.append(len(u))
    if not fits:
        return 0.0
    return float(np.average(fits, weights=weights))


def find_windows(base, donors, t):
    """Hard windows from donor disagreement + V3b-style evidence flags on the baseline."""
    n = len(base["off"])
    dis = np.zeros(n)
    for d in donors:
        dis += (d["off"][:n] != base["off"]).astype(float)
    dis /= max(len(donors), 1)

    # evidence flags: quarters where timing favors triple but baseline group is duple
    e = np.zeros(n, bool)
    measure = np.cumsum(base["db"] > 0)
    quarter = base["off"] // N_PHASE
    phase = base["off"] % N_PHASE
    gkey = measure * 1000 + quarter
    kept_idx = np.flatnonzero(base["pad"])
    g_first = {}
    for g in np.unique(gkey[kept_idx]):
        idx = kept_idx[gkey[kept_idx] == g]
        g_first[int(g)] = float(t[idx].min())
    for g in np.unique(gkey[kept_idx]):
        idx = kept_idx[gkey[kept_idx] == g]
        anchor = g_first.get(int(g) + 1)
        if anchor is None:
            anchor = g_first.get((int(g) // 1000 + 1) * 1000)
        t0 = float(t[idx].min())
        if anchor is None or anchor <= t0:
            continue
        ph_u, t_u = [], []
        for p_ in np.unique(phase[idx]):
            sel = idx[phase[idx] == p_]
            ph_u.append(float(p_)); t_u.append(float(t[sel].min()))
        if len(ph_u) < 3:
            continue
        u = (np.array(sorted(t_u)) - t0) / (anchor - t0)
        u = u[(u >= -0.05) & (u < 1.05)]
        if len(u) < 3:
            continue
        err_d = float(np.mean(np.min(np.abs(u[:, None] - DUPLE[None, :]), axis=1)))
        err_t = float(np.mean(np.min(np.abs(u[:, None] - TRIPLE[None, :]), axis=1)))
        says_triple = bool(np.any((np.array(ph_u) % 3) != 0))
        if (err_d - err_t) > THETA and not says_triple:
            e[idx] = True

    flag = (dis >= 0.25) | e
    # maximal runs, pad +/-2, min len 3, merge gaps < 4
    wins = []
    i = 0
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            wins.append([max(0, i - 2), min(n, j + 3)])
            i = j + 1
        else:
            i += 1
    merged = []
    for w in wins:
        if merged and w[0] - merged[-1][1] < 4:
            merged[-1][1] = w[1]
        else:
            merged.append(w)
    return [tuple(w) for w in merged if w[1] - w[0] >= 3], dis, e


def repair(y_base, base, donors_y, donors, t):
    """Splice best-variant (offset,duration,pad) per window into a copy of the baseline."""
    wins, dis, e = find_windows(base, donors, t)
    y_rep = {k: v.clone() for k, v in y_base.items()}
    chosen = []
    for (lo, hi) in wins:
        L = hi - lo
        best, best_s = None, None
        # baseline competes
        cov = float(np.mean(base["pad"][lo:hi]))
        fit = quarter_anchor_fit(base["off"], base["db"], base["pad"], t, lo, hi)
        base_s = W_COV * cov + W_FIT * fit
        best, best_s = ("base", None), base_s
        for di, d in enumerate(donors):
            # variant = baseline with donor slice on (off, dur, pad)
            off_v = base["off"].copy(); off_v[lo:hi] = d["off"][lo:hi]
            dur_v = base["dur"].copy(); dur_v[lo:hi] = d["dur"][lo:hi]
            pad_v = base["pad"].copy(); pad_v[lo:hi] = d["pad"][lo:hi]
            cov = float(np.mean(pad_v[lo:hi]))
            fit = quarter_anchor_fit(off_v, base["db"], pad_v, t, lo, hi)
            s = W_COV * cov + W_FIT * fit
            if s > best_s + 1e-12:
                best, best_s = ("donor", di), s
        if best[0] == "donor":
            di = best[1]
            def _splice(dst, src):
                if dst.ndim >= 2:
                    dst[..., lo:hi, :] = src[..., lo:hi, :]
                else:
                    dst[lo:hi] = src[lo:hi]
            for k in ("offset", "duration", "pad", "pad_prob", "raw_offset", "raw_duration"):
                if k in y_rep and k in donors_y[di]:
                    _splice(y_rep[k], donors_y[di][k])
            chosen.append({"lo": int(lo), "hi": int(hi), "donor": di, "score": best_s,
                           "base_score": base_s})
    return y_rep, wins, chosen


def tpf(rec, gt_total):
    missed = gt_total - rec["n_tuplet"]
    onset_ct = round(rec["onset_err_tuplet"] * rec["n_tuplet"] / 100.0) if rec["onset_err_tuplet"] is not None else 0
    return int(missed + onset_ct), int(missed), int(onset_ct)


def gt_tuplet_total(p, workdir):
    """Count GT tuplet notes from the MTD gt parse (piece-level constant)."""
    agg, gt_fmt3x, err_detail, auto_match = VB.MTD.run_muster_keep(
        VB.postprocess_score(VB.MultistreamTokenizer.detokenize_mxl(
            p["_y_base"], pad_threshold=0.5), inPlace=True), p["score"], workdir)
    gt_info = VB.MTD.parse_gt_fmt3x(gt_fmt3x)
    return sum(1 for gi in gt_info.values() if gi["tuplet"]), gt_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Scriabin"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/local_repair_v4.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    paths = VB.collect_paths("test")
    for p in paths:
        try:
            p["_x"] = VB.MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_t"] = VB.MultistreamTokenizer.parse_midi(p["midi"])["onset"].numpy()
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception:
            p["_x"] = None; p["n_notes"] = 1 << 30
    paths = [p for p in paths
             if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)]
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = set(), []
    for p in paths:
        key = (p["composer"], p["piece"])
        if key not in seen and p["_x"] is not None:
            seen.add(key); kept.append(p)
    paths = kept
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))

    model = VB.load_any_checkpoint(args.ckpt, "cpu"); model.eval(); model.to("cpu")
    overrides = {"offset": (args.topk, args.temp), "duration": (args.topk, args.temp)}
    work_root = VB.REPO / "benchmark" / "v4_work"
    results = {"ckpt": args.ckpt, "n": args.n, "topk": args.topk, "temp": args.temp,
               "weights": {"w_cov": W_COV, "w_fit": W_FIT, "w_lik": W_LIK}, "pieces": {}}
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        base = streams_np(y_base)
        gt_total, _ = gt_tuplet_total(p, work_root / tag / "gtcount")
        rec_b = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        tpf_b = tpf(rec_b, gt_total)
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
              f"(missed={tpf_b[1]} onset_err={tpf_b[2]}; gt_tuplets={gt_total})", flush=True)

        donors_y, donors = [], []
        for i in range(args.n):
            y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            donors_y.append(y_c); donors.append(streams_np(y_c))
            print(f"  donor{i:02d} decoded", flush=True)

        y_rep, wins, chosen = repair(y_base, base, donors_y, donors, p["_t"])
        n_win_notes = sum(hi - lo for lo, hi in wins)
        print(f"  windows: {len(wins)} covering {n_win_notes}/{p['n_notes']} notes; "
              f"{len(chosen)} repaired", flush=True)
        rec_r = VB.muster_piece(y_rep, p, work_root / tag / "repaired")
        tpf_r = tpf(rec_r, gt_total)
        drift_ok = rec_r["meaner_overall"] is not None and \
            rec_r["meaner_overall"] <= rec_b["meaner_overall"] + 0.5
        print(f"  repaired: MeanER={rec_r['meaner_overall']:.2f} TPF={tpf_r[0]} "
              f"(missed={tpf_r[1]} onset_err={tpf_r[2]}) drift_ok={drift_ok}", flush=True)

        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "gt_tuplets": gt_total,
            "baseline": {**rec_b, "tpf": tpf_b[0], "tpf_missed": tpf_b[1], "tpf_onset": tpf_b[2]},
            "repaired": {**rec_r, "tpf": tpf_r[0], "tpf_missed": tpf_r[1], "tpf_onset": tpf_r[2]},
            "windows": [{"lo": lo, "hi": hi} for lo, hi in wins],
            "chosen": chosen, "drift_ok": bool(drift_ok),
        }
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
