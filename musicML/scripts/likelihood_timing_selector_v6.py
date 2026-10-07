"""V6 donor-only best-of-N selector.

V3b failed because its internal-coherence terms reward "do nothing" zero-triplet decodes.
V4/V5 local repair failed because the useful candidate gains are scattered and local
evidence windows do not isolate them reliably. V6 returns to whole-candidate selection, but
uses two signals that do not directly reward zero-triplet cleanliness:

  * teacher-forced rhythm reconstruction loss under the model itself;
  * V3b-style timing-contradiction evidence, bounded to the real input note span.

The selector ranks sampled donors only. The argmax baseline is reported and scored, but it
does not compete, matching the earlier best-of-N protocol.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import local_repair_v4 as V4
import verifier_bestofn as VB


N_PHASE = 24
THETA = 0.02
DUPLE = np.array([0.0, 0.25, 0.5, 0.75])
TRIPLE = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])


def streams_np(y):
    return V4.streams_np(y)


def _slice_streams(streams, lo, hi):
    out = {}
    for k, v in streams.items():
        if v.ndim == 1:
            out[k] = v[lo:hi]
        else:
            out[k] = v[lo:hi, ...]
    return out


def prepare_x(x):
    return {
        k: (v.unsqueeze(0) if v.ndim in (1, 2) else v).to("cpu")
        for k, v in x.items()
    }


def prepare_y(y, n_real):
    out = {}
    for k, v in y.items():
        if k == "pad_prob" or k.startswith("raw_"):
            continue
        vv = v[:n_real]
        if k == "pad":
            vv = vv.reshape(-1)
        out[k] = (vv.unsqueeze(0) if vv.ndim in (1, 2) else vv).to("cpu")
    return out


def _teacher_forced_rhythm_loss_chunk(model, x, y, n_chunk):
    xx = prepare_x(x)
    yy = prepare_y(y, n_chunk)
    with torch.no_grad():
        pred = model.forward(input_streams=xx, output_streams=yy)

    mask = yy["pad"].reshape(-1) > 0.5
    if int(mask.sum()) == 0:
        return {}, {}

    sums = {}
    counts = {}
    for k in ("offset", "duration", "downbeat"):
        logits = pred[k][0, :n_chunk]
        target = yy[k][0].argmax(-1)
        ce_sum = F.cross_entropy(logits[mask], target[mask], reduction="sum")
        sums[k] = float(ce_sum.detach())
        counts[k] = int(mask.sum())

    pad_logits = pred["pad"][0, :n_chunk].reshape(-1)
    pad_target = yy["pad"].reshape(-1).float()
    pad_sum = F.binary_cross_entropy_with_logits(pad_logits, pad_target, reduction="sum")
    sums["pad"] = float(pad_sum.detach())
    counts["pad"] = int(n_chunk)
    return sums, counts


def teacher_forced_rhythm_loss(model, x, y, n_real, chunk_size=None):
    """Weighted rhythm-stream reconstruction loss over the real input note span.

    Full-piece teacher forcing can exceed the checkpoint's positional embedding range on
    long pieces, so score in non-overlapping chunks bounded by the model's configured
    context. For the released AR decoder, chunk boundaries are a bounded approximation.
    """
    if chunk_size is None:
        enc_max = int(getattr(model.enc_config, "max_position_embeddings", 512))
        dec_max = int(getattr(model.dec_config, "max_position_embeddings", 512))
        chunk_size = min(enc_max, dec_max)
    sums = {k: 0.0 for k in ("offset", "duration", "downbeat", "pad")}
    counts = {k: 0 for k in ("offset", "duration", "downbeat", "pad")}
    for lo in range(0, n_real, chunk_size):
        hi = min(n_real, lo + chunk_size)
        x_chunk = _slice_streams(x, lo, hi)
        y_chunk = _slice_streams(y, lo, hi)
        chunk_sums, chunk_counts = _teacher_forced_rhythm_loss_chunk(
            model, x_chunk, y_chunk, hi - lo
        )
        for k, v in chunk_sums.items():
            sums[k] += v
            counts[k] += chunk_counts[k]

    if any(counts[k] == 0 for k in ("offset", "duration", "downbeat", "pad")):
        return float("inf"), {}

    parts = {k: sums[k] / counts[k] for k in sums}

    # Same relative weights as config.FEATURES for these streams, normalized for readability.
    total = (
        0.25 * parts["offset"]
        + 0.30 * parts["duration"]
        + 0.40 * parts["downbeat"]
        + parts["pad"]
    ) / 1.95
    return float(total), {k: round(v, 6) for k, v in parts.items()}


def timing_contradiction(y, t, n_real):
    """V3b timing contradiction, without V1/V2, bounded to real input notes."""
    s = streams_np(y)
    n = min(n_real, len(s["off"]), len(t))
    off = s["off"][:n]
    db = s["db"][:n]
    pad = s["pad"][:n]
    tt = np.asarray(t[:n], dtype=float)

    measure = np.cumsum(db > 0)
    quarter = off // N_PHASE
    phase = off % N_PHASE
    gkey = measure * 1000 + quarter
    kept_idx = np.flatnonzero(pad)

    g_first = {}
    for g in np.unique(gkey[kept_idx]):
        idx = kept_idx[gkey[kept_idx] == g]
        g_first[int(g)] = float(tt[idx].min())

    penalty = 0.0
    n_groups = 0
    n_contra = 0
    for g in np.unique(gkey[kept_idx]):
        idx = kept_idx[gkey[kept_idx] == g]
        anchor = g_first.get(int(g) + 1)
        if anchor is None:
            anchor = g_first.get((int(g) // 1000 + 1) * 1000)
        t0 = float(tt[idx].min())
        if anchor is None or anchor <= t0:
            continue
        ph_u, t_u = [], []
        for p_ in np.unique(phase[idx]):
            sel = idx[phase[idx] == p_]
            ph_u.append(float(p_))
            t_u.append(float(tt[sel].min()))
        if len(ph_u) < 3:
            continue
        t_s = np.array(t_u)[np.argsort(t_u)]
        if not np.all(np.diff(t_s) > 0):
            continue
        u = (t_s - t0) / (anchor - t0)
        u = u[(u >= -0.05) & (u < 1.05)]
        if len(u) < 3:
            continue
        err_d = float(np.mean(np.min(np.abs(u[:, None] - DUPLE[None, :]), axis=1)))
        err_t = float(np.mean(np.min(np.abs(u[:, None] - TRIPLE[None, :]), axis=1)))
        margin = err_d - err_t
        says_triple = bool(np.any((np.array(ph_u) % 3) != 0))
        n_groups += 1
        if margin > THETA and not says_triple:
            penalty += margin
            n_contra += 1
        elif margin < -THETA and says_triple:
            penalty += -margin
            n_contra += 1

    per_group = penalty / max(n_groups, 1)
    return float(per_group), {
        "timing_penalty": round(float(penalty), 6),
        "timing_per_group": round(float(per_group), 8),
        "timing_groups": int(n_groups),
        "timing_contra": int(n_contra),
    }


def zscores(vals):
    arr = np.asarray(vals, dtype=float)
    sd = float(arr.std())
    if sd == 0.0:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / sd


def select_candidate(records):
    model_z = zscores([r["model_loss"] for r in records])
    timing_z = zscores([r["timing_per_group"] for r in records])
    for r, mz, ez in zip(records, model_z, timing_z):
        r["model_z"] = round(float(mz), 6)
        r["timing_z"] = round(float(ez), 6)
        r["selector_score"] = round(float(mz + ez), 6)
    return min(records, key=lambda r: (r["selector_score"], r["i"]))


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
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/likelihood_timing_selector_v6_dev.json"))
    ap.add_argument("--eval-all", action="store_true")
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
    work_root = VB.REPO / "benchmark" / "v6_work"
    results = {
        "ckpt": args.ckpt,
        "n": args.n,
        "topk": args.topk,
        "temp": args.temp,
        "selector": "z(teacher_forced_rhythm_loss)+z(v3b_timing_per_group), donor-only",
        "pieces": {},
    }
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)

        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        gt_total, _ = V4.gt_tuplet_total(p, work_root / tag / "gtcount")
        rec_b = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        tpf_b = V4.tpf(rec_b, gt_total)
        base_loss, base_loss_parts = teacher_forced_rhythm_loss(model, p["_x"], y_base, p["n_notes"])
        base_ev, base_ev_detail = timing_contradiction(y_base, p["_t"], p["n_notes"])
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
              f"model_loss={base_loss:.4f} timing={base_ev:.6f}", flush=True)

        records = []
        donors_y = []
        for i in range(args.n):
            y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            donors_y.append(y_c)
            model_loss, loss_parts = teacher_forced_rhythm_loss(model, p["_x"], y_c, p["n_notes"])
            ev, ev_detail = timing_contradiction(y_c, p["_t"], p["n_notes"])
            rec = None
            if args.eval_all:
                rec = VB.muster_piece(y_c, p, work_root / tag / f"cand{i:02d}")
            records.append({
                "i": i,
                "seed": 1000 + i,
                "model_loss": round(model_loss, 8),
                "loss_parts": loss_parts,
                **ev_detail,
                "timing_per_group": ev,
                "metrics": rec,
            })
            print(f"  cand{i:02d}: model_loss={model_loss:.4f} timing={ev:.6f}", flush=True)

        pick = select_candidate(records)
        y_pick = donors_y[pick["i"]]
        rec_p = pick["metrics"]
        if rec_p is None:
            rec_p = VB.muster_piece(y_pick, p, work_root / tag / f"pick_cand{pick['i']:02d}")
        tpf_p = V4.tpf(rec_p, gt_total)
        gate_ok = tpf_p[0] <= 50 and rec_p["meaner_overall"] <= rec_b["meaner_overall"] + 0.5
        print(f"  --> pick cand{pick['i']:02d}: MeanER={rec_p['meaner_overall']:.2f} "
              f"TPF={tpf_p[0]} gate_ok={gate_ok}", flush=True)

        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "gt_tuplets": gt_total,
            "baseline": {
                **rec_b,
                "tpf": tpf_b[0],
                "tpf_missed": tpf_b[1],
                "tpf_onset": tpf_b[2],
                "model_loss": round(base_loss, 8),
                "loss_parts": base_loss_parts,
                **base_ev_detail,
            },
            "candidates": records,
            "selector_pick": {
                **pick,
                "metrics": {
                    **rec_p,
                    "tpf": tpf_p[0],
                    "tpf_missed": tpf_p[1],
                    "tpf_onset": tpf_p[2],
                },
            },
            "gate_ok": bool(gate_ok),
        }
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
