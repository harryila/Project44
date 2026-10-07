"""V9 decode-time lever sweep.

V8 showed a useful selector on Scriabin/Liszt, but Ravel's sampled donor pool had no
non-worse candidate. This script tests whether deterministic decode-time logit levers can
create a better pool before selection:

  * offset phase boost over the live top-k offset candidates;
  * optional duration marginal and metrical priors already wired into model.generate.

Every row is evaluated with the same MUSTER tuplet decomposition used by the selector
experiments, so the output records both MeanER and tuplet-passage failure (TPF).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import local_repair_v4 as V4
import likelihood_timing_selector_v6 as V6
import verifier_bestofn as VB


N_PHASE = 24


class OffsetPhaseBoost:
    """Boost selected offset phases among the model's current top-k candidates."""

    def __init__(self, lam, mode="eighth", topk=15):
        self.lam = float(lam)
        self.mode = mode
        self.topk = int(topk)
        if mode == "all":
            allowed = {p for p in range(N_PHASE) if p % 3 != 0}
        elif mode == "eighth":
            allowed = {8, 16}
        elif mode == "sixteenth":
            allowed = {4, 8, 16, 20}
        elif mode == "none":
            allowed = set()
        else:
            raise ValueError(f"unknown offset boost mode: {mode}")
        self.allowed = allowed
        self.phase = torch.tensor([b % N_PHASE for b in range(145)], dtype=torch.long)

    def __call__(self, logits, prev_off):
        if self.lam == 0.0 or not self.allowed:
            return logits
        bsz, vocab = logits.shape
        logits = logits.clone()
        _, topi = torch.topk(logits, min(self.topk, vocab), dim=-1)
        phase = self.phase.to(logits.device)
        allowed = torch.tensor(
            sorted(self.allowed), dtype=torch.long, device=logits.device
        )
        for row in range(bsz):
            idx = topi[row]
            mask = (phase[idx][:, None] == allowed[None, :]).any(dim=1)
            logits[row, idx[mask]] = logits[row, idx[mask]] + self.lam
        return logits


def parse_floats(values):
    out = []
    for value in values:
        out.append(float(value))
    return out


def config_grid(args):
    offset_lams = parse_floats(args.offset_lambdas)
    dur_taus = parse_floats(args.dur_taus)
    dur_metrical_lams = parse_floats(args.dur_metrical_lambdas)
    configs = []
    seen = set()
    for mode in args.offset_modes:
        for offset_lam in offset_lams:
            if mode == "none" and offset_lam != 0.0:
                continue
            for dur_tau in dur_taus:
                for dur_metrical_lam in dur_metrical_lams:
                    key = (
                        mode if offset_lam else "none",
                        round(offset_lam, 8),
                        round(dur_tau, 8),
                        round(dur_metrical_lam, 8),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append(
                        {
                            "offset_mode": key[0],
                            "offset_lambda": float(offset_lam),
                            "dur_tau": float(dur_tau),
                            "dur_metrical_lambda": float(dur_metrical_lam),
                        }
                    )
    return configs


def decode(model, x, cfg, args, priors):
    offset_rerank = None
    if cfg["offset_lambda"] and cfg["offset_mode"] != "none":
        offset_rerank = OffsetPhaseBoost(
            cfg["offset_lambda"], mode=cfg["offset_mode"], topk=args.offset_topk
        )
    dur_log_pi = priors.get("log_pi_dur") if cfg["dur_tau"] else None
    dur_metrical = (
        priors.get("log_p_dur_given_phase") if cfg["dur_metrical_lambda"] else None
    )
    torch.manual_seed(args.seed)
    with torch.no_grad():
        return VB.infer(
            x,
            model,
            overlap=args.overlap,
            chunk=args.chunk,
            verbose=False,
            kv_cache=True,
            dur_log_pi=dur_log_pi,
            dur_tau=cfg["dur_tau"],
            dur_metrical=dur_metrical,
            dur_metrical_lambda=cfg["dur_metrical_lambda"],
            offset_rerank=offset_rerank,
        )


def decoded_triplet_count(y, n_real):
    streams = V4.streams_np(y)
    n = min(n_real, len(streams["off"]))
    return int(((streams["off"][:n] % N_PHASE) % 3 != 0).sum())


def eval_config(model, p, cfg, args, priors, gt_total, work_root):
    y = decode(model, p["_x"], cfg, args, priors)
    rec = VB.muster_piece(y, p, work_root)
    tpf = V4.tpf(rec, gt_total)
    timing, timing_detail = V6.timing_contradiction(y, p["_t"], p["n_notes"])
    loss, loss_parts = V6.teacher_forced_rhythm_loss(model, p["_x"], y, p["n_notes"])
    return {
        **cfg,
        "meaner_overall": rec["meaner_overall"],
        "onset_err_tuplet": rec["onset_err_tuplet"],
        "onset_err_nontuplet": rec["onset_err_nontuplet"],
        "n_tuplet_matched": rec["n_tuplet"],
        "n_nontuplet_matched": rec["n_nontuplet"],
        "tpf": tpf[0],
        "tpf_missed": tpf[1],
        "tpf_onset": tpf[2],
        "decoded_triplet_count": decoded_triplet_count(y, p["n_notes"]),
        "timing_per_group": timing,
        "timing_detail": timing_detail,
        "model_loss": round(loss, 8),
        "loss_parts": loss_parts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Ravel"])
    ap.add_argument("--offset-modes", nargs="*", default=["none", "eighth"])
    ap.add_argument("--offset-lambdas", nargs="*", default=["0", "0.5", "1.0"])
    ap.add_argument("--offset-topk", type=int, default=15)
    ap.add_argument("--dur-taus", nargs="*", default=["0"])
    ap.add_argument("--dur-metrical-lambdas", nargs="*", default=["0"])
    ap.add_argument("--prior-path", default=str(VB.REPO / "data/duration_priors.pt"))
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/decode_lever_sweep_v9.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    priors = {}
    prior_path = Path(args.prior_path)
    if not prior_path.is_absolute():
        prior_path = VB.REPO / prior_path
    if prior_path.exists():
        priors = torch.load(prior_path, map_location="cpu", weights_only=False)

    configs = config_grid(args)
    paths = V6.piece_paths(args)
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))
    print(f"{len(configs)} configs", flush=True)

    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")

    work_root = VB.REPO / "benchmark" / "v9_work"
    results = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "offset_topk": args.offset_topk,
        "pieces": {},
    }
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        gt_total, _ = V4.gt_tuplet_total(p, work_root / tag / "gtcount")
        base_rec = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        base_tpf = V4.tpf(base_rec, gt_total)
        base_loss, base_parts = V6.teacher_forced_rhythm_loss(
            model, p["_x"], y_base, p["n_notes"]
        )
        base_timing, base_timing_detail = V6.timing_contradiction(
            y_base, p["_t"], p["n_notes"]
        )
        baseline = {
            "meaner_overall": base_rec["meaner_overall"],
            "onset_err_tuplet": base_rec["onset_err_tuplet"],
            "onset_err_nontuplet": base_rec["onset_err_nontuplet"],
            "n_tuplet_matched": base_rec["n_tuplet"],
            "n_nontuplet_matched": base_rec["n_nontuplet"],
            "tpf": base_tpf[0],
            "tpf_missed": base_tpf[1],
            "tpf_onset": base_tpf[2],
            "decoded_triplet_count": decoded_triplet_count(y_base, p["n_notes"]),
            "timing_per_group": base_timing,
            "timing_detail": base_timing_detail,
            "model_loss": round(base_loss, 8),
            "loss_parts": base_parts,
        }
        rows = []
        print(
            f"  baseline: MeanER={baseline['meaner_overall']:.2f} "
            f"TPF={baseline['tpf']} decTrip={baseline['decoded_triplet_count']} "
            f"timing={baseline['timing_per_group']:.6f}",
            flush=True,
        )
        for i, cfg in enumerate(configs):
            row = eval_config(
                model,
                p,
                cfg,
                args,
                priors,
                gt_total,
                work_root / tag / f"cfg{i:03d}",
            )
            rows.append(row)
            print(
                f"  cfg{i:03d} mode={cfg['offset_mode']} off={cfg['offset_lambda']:.3g} "
                f"tau={cfg['dur_tau']:.3g} met={cfg['dur_metrical_lambda']:.3g} "
                f"MeanER={row['meaner_overall']:.2f} TPF={row['tpf']} "
                f"decTrip={row['decoded_triplet_count']}",
                flush=True,
            )
            results["pieces"][f"{p['composer']}/{p['piece']}"] = {
                "gt_tuplets": gt_total,
                "baseline": baseline,
                "rows": rows,
            }
            json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
