"""Exploratory V5 local repair after the V4 detector saturated.

V4 correctly failed its drift gate, but the failure was mechanical: raw donor-vs-baseline
offset disagreement fires almost everywhere under temperature sampling. V5 keeps the useful
part of V4 and removes the saturated detector:

  * detect windows only from timing evidence on the baseline, never raw donor disagreement;
  * bound all detection to the input note count, not the padded decoder length;
  * cap repair windows so a local patch cannot become a whole-piece splice;
  * splice donor downbeats together with offset/duration/pad for measure consistency;
  * use a small edit and bar-consistency penalty so the baseline wins unless the donor has
    better local coverage/timing evidence.

This script is intentionally separate from local_repair_v4.py so the failed V4 artifact
remains auditable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import local_repair_v4 as V4
import verifier_bestofn as VB


N_PHASE = 24
THETA = 0.02
DUPLE = np.array([0.0, 0.25, 0.5, 0.75])
TRIPLE = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])

W_COV = 2.0
W_FIT = 1.0
W_EDIT = 0.05
W_BAR = 0.25
ACCEPT_MARGIN = 0.005
PAD = 2
MERGE_GAP = 3
MIN_WINDOW = 3
MAX_WINDOW = 40


def evidence_flags(base, t, n_real):
    """Return baseline timing-evidence flags and group-level diagnostics.

    A note is flagged if its baseline (measure, quarter) group is duple, but the input
    timing fits the triple grid better than the duple grid by THETA.
    """
    n = min(n_real, len(base["off"]), len(t))
    e = np.zeros(n, bool)
    groups = []
    measure = np.cumsum(base["db"][:n] > 0)
    quarter = base["off"][:n] // N_PHASE
    phase = base["off"][:n] % N_PHASE
    gkey = measure * 1000 + quarter
    kept_idx = np.flatnonzero(base["pad"][:n])

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
            ph_u.append(float(p_))
            t_u.append(float(t[sel].min()))
        if len(ph_u) < 3:
            continue
        u = (np.array(sorted(t_u)) - t0) / (anchor - t0)
        u = u[(u >= -0.05) & (u < 1.05)]
        if len(u) < 3:
            continue
        err_d = float(np.mean(np.min(np.abs(u[:, None] - DUPLE[None, :]), axis=1)))
        err_t = float(np.mean(np.min(np.abs(u[:, None] - TRIPLE[None, :]), axis=1)))
        margin = err_d - err_t
        says_triple = bool(np.any((np.array(ph_u) % 3) != 0))
        if margin > THETA and not says_triple:
            e[idx] = True
            groups.append({
                "g": int(g),
                "lo": int(idx[0]),
                "hi": int(idx[-1] + 1),
                "n": int(len(idx)),
                "margin": round(margin, 4),
            })
    return e, groups


def windows_from_flags(flag, n_real):
    """Build capped local windows from evidence flags."""
    wins = []
    i = 0
    n = min(len(flag), n_real)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            wins.append([max(0, i - PAD), min(n, j + 1 + PAD)])
            i = j + 1
        else:
            i += 1

    merged = []
    for w in wins:
        if merged and w[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = w[1]
        else:
            merged.append(w)

    capped = []
    for lo, hi in merged:
        if hi - lo < MIN_WINDOW:
            continue
        while hi - lo > MAX_WINDOW:
            capped.append((lo, lo + MAX_WINDOW))
            lo += MAX_WINDOW
        if hi - lo >= MIN_WINDOW:
            capped.append((lo, hi))
    return capped


def bar_mismatch(off, db, lo, hi, n_real):
    """Fraction of local bar-boundary inconsistencies after a proposed splice."""
    lo_c = max(1, lo)
    hi_c = min(n_real, hi + 1)
    if hi_c <= lo_c:
        return 0.0
    prev = off[lo_c - 1:hi_c - 1]
    cur = off[lo_c:hi_c]
    should_downbeat = cur < prev
    says_downbeat = db[lo_c:hi_c] > 0
    return float(np.mean(should_downbeat != says_downbeat))


def variant_score(base, off, dur, db, pad, t, lo, hi, n_real):
    cov = float(np.mean(pad[lo:hi]))
    fit = V4.quarter_anchor_fit(off, db, pad, t, lo, hi)
    edit = float(np.mean(
        (off[lo:hi] != base["off"][lo:hi])
        | (dur[lo:hi] != base["dur"][lo:hi])
        | (db[lo:hi] != base["db"][lo:hi])
        | (pad[lo:hi] != base["pad"][lo:hi])
    ))
    bar = bar_mismatch(off, db, lo, hi, n_real)
    score = W_COV * cov + W_FIT * fit - W_EDIT * edit - W_BAR * bar
    return float(score), {
        "cov": round(cov, 4),
        "fit": round(float(fit), 4),
        "edit": round(edit, 4),
        "bar": round(bar, 4),
    }


def splice_slice(y_rep, donor_y, lo, hi):
    def _splice(dst, src):
        if dst.ndim >= 2:
            dst[..., lo:hi, :] = src[..., lo:hi, :]
        else:
            dst[lo:hi] = src[lo:hi]

    for k in (
        "offset",
        "duration",
        "downbeat",
        "pad",
        "pad_prob",
        "raw_offset",
        "raw_duration",
        "raw_downbeat",
    ):
        if k in y_rep and k in donor_y:
            _splice(y_rep[k], donor_y[k])


def repair(y_base, base, donors_y, donors, t, n_real):
    flags, groups = evidence_flags(base, t, n_real)
    wins = windows_from_flags(flags, n_real)
    y_rep = {k: v.clone() for k, v in y_base.items()}
    chosen = []

    for lo, hi in wins:
        base_s, base_det = variant_score(
            base, base["off"], base["dur"], base["db"], base["pad"], t, lo, hi, n_real
        )
        best = {
            "kind": "base",
            "donor": None,
            "score": base_s,
            "detail": base_det,
        }
        for di, d in enumerate(donors):
            off_v = base["off"].copy()
            dur_v = base["dur"].copy()
            db_v = base["db"].copy()
            pad_v = base["pad"].copy()
            off_v[lo:hi] = d["off"][lo:hi]
            dur_v[lo:hi] = d["dur"][lo:hi]
            db_v[lo:hi] = d["db"][lo:hi]
            pad_v[lo:hi] = d["pad"][lo:hi]
            s, det = variant_score(base, off_v, dur_v, db_v, pad_v, t, lo, hi, n_real)
            if s > best["score"] + ACCEPT_MARGIN:
                best = {
                    "kind": "donor",
                    "donor": di,
                    "score": s,
                    "detail": det,
                }
        if best["kind"] == "donor":
            splice_slice(y_rep, donors_y[best["donor"]], lo, hi)
            chosen.append({
                "lo": int(lo),
                "hi": int(hi),
                "donor": int(best["donor"]),
                "score": round(best["score"], 6),
                "base_score": round(base_s, 6),
                "detail": best["detail"],
                "base_detail": base_det,
            })

    return y_rep, wins, chosen, groups


def piece_paths(args):
    paths = VB.collect_paths("test")
    for p in paths:
        try:
            p["_x"] = VB.MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_t"] = VB.MultistreamTokenizer.parse_midi(p["midi"])["onset"].numpy()
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception:
            p["_x"] = None
            p["n_notes"] = 1 << 30
    paths = [
        p for p in paths
        if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)
    ]
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = set(), []
    for p in paths:
        key = (p["composer"], p["piece"])
        if key not in seen and p["_x"] is not None:
            seen.add(key)
            kept.append(p)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Scriabin"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/local_repair_v5_dev.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    paths = piece_paths(args)
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))

    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")
    overrides = {"offset": (args.topk, args.temp), "duration": (args.topk, args.temp)}
    work_root = VB.REPO / "benchmark" / "v5_work"
    results = {
        "ckpt": args.ckpt,
        "n": args.n,
        "topk": args.topk,
        "temp": args.temp,
        "weights": {
            "w_cov": W_COV,
            "w_fit": W_FIT,
            "w_edit": W_EDIT,
            "w_bar": W_BAR,
            "accept_margin": ACCEPT_MARGIN,
            "max_window": MAX_WINDOW,
        },
        "pieces": {},
    }
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        base = V4.streams_np(y_base)
        gt_total, _ = V4.gt_tuplet_total(p, work_root / tag / "gtcount")
        rec_b = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        tpf_b = V4.tpf(rec_b, gt_total)
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
              f"(missed={tpf_b[1]} onset_err={tpf_b[2]}; gt_tuplets={gt_total})", flush=True)

        donors_y, donors = [], []
        for i in range(args.n):
            y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            donors_y.append(y_c)
            donors.append(V4.streams_np(y_c))
            print(f"  donor{i:02d} decoded", flush=True)

        y_rep, wins, chosen, groups = repair(y_base, base, donors_y, donors, p["_t"], p["n_notes"])
        n_win_notes = sum(hi - lo for lo, hi in wins)
        print(f"  windows: {len(wins)} covering {n_win_notes}/{p['n_notes']} notes; "
              f"{len(chosen)} repaired", flush=True)

        rec_r = VB.muster_piece(y_rep, p, work_root / tag / "repaired")
        tpf_r = V4.tpf(rec_r, gt_total)
        drift_ok = (
            rec_r["meaner_overall"] is not None
            and rec_r["meaner_overall"] <= rec_b["meaner_overall"] + 0.5
        )
        gate_ok = drift_ok and tpf_r[0] <= 50
        print(f"  repaired: MeanER={rec_r['meaner_overall']:.2f} TPF={tpf_r[0]} "
              f"(missed={tpf_r[1]} onset_err={tpf_r[2]}) drift_ok={drift_ok} "
              f"gate_ok={gate_ok}", flush=True)

        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "gt_tuplets": gt_total,
            "baseline": {**rec_b, "tpf": tpf_b[0], "tpf_missed": tpf_b[1], "tpf_onset": tpf_b[2]},
            "repaired": {**rec_r, "tpf": tpf_r[0], "tpf_missed": tpf_r[1], "tpf_onset": tpf_r[2]},
            "windows": [{"lo": lo, "hi": hi} for lo, hi in wins],
            "evidence_groups": groups,
            "chosen": chosen,
            "drift_ok": bool(drift_ok),
            "gate_ok": bool(gate_ok),
        }
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
