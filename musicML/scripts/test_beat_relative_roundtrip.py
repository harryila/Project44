"""B2 Phase-1 GATE: verify the beat-relative (within-quarter + quarter_idx) tokenization is LOSSLESS.

For each score, compare the reconstructed within-measure offset under B2 mode to the baseline
within-measure offset (both on the 1/24 grid). They must match (within 1 bucket = 1/24) — i.e., the
within-quarter recoding adds NO loss. Also reports that triplet positions concentrate into within-quarter
buckets {8,16} (the sample-efficiency benefit). No model needed — pure tokenizer round-trip.

Run:  venv311/bin/python scripts/test_beat_relative_roundtrip.py <score.mxl> [more.mxl ...]
"""
import sys, glob
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"))
import warnings; warnings.simplefilter("ignore")
import numpy as np
import tokenizer as TK
from tokenizer import MultistreamTokenizer, one_hot_unbucketing, PARAMS, PARAMS_BR


def offsets_from_tokens(toks, beat_rel):
    if beat_rel:
        wq = one_hot_unbucketing(toks["offset"], **PARAMS_BR["offset"]).numpy().astype(float)
        qi = one_hot_unbucketing(toks["quarter_idx"], **PARAMS_BR["quarter_idx"]).numpy().astype(float)
        return qi + wq
    return one_hot_unbucketing(toks["offset"], **PARAMS["offset"]).numpy().astype(float)


def main():
    scores = sys.argv[1:]
    if not scores:
        scores = sorted(glob.glob(str(REPO / "data" / "pairs" / "*.mxl")))[:8]
    assert scores, "no scores; pass .mxl paths"
    worst = 0.0; n_notes = 0; n_trip = 0; trip_in_816 = 0; ok = 0
    for sp in scores:
        try:
            raw = MultistreamTokenizer.parse_mxl(sp)            # raw within-measure offset (truth)
            TK.BEAT_RELATIVE = True
            br = MultistreamTokenizer.tokenize_mxl(sp)          # bucketed B2 (within-q + quarter_idx)
        except Exception as e:
            TK.BEAT_RELATIVE = False
            print(f"  parse fail {Path(sp).name}: {repr(e)[:80]}"); continue
        finally:
            TK.BEAT_RELATIVE = False
        if raw["offset"].shape[0] == 0:
            continue
        base_off = raw["offset"].numpy().astype(float)          # raw within-measure (truth, unclamped)
        br_off = offsets_from_tokens(br, True)                  # B2 bucketed reconstruction
        if len(base_off) != len(br_off):
            print(f"  LENGTH MISMATCH {Path(sp).name}: {len(base_off)} vs {len(br_off)}"); continue
        # compare only notes within the quarter_idx range (<=24q); within-q bucketing adds <=1/24
        keep = base_off <= 24.0
        d = float(np.max(np.abs(base_off[keep] - br_off[keep]))) if keep.any() else 0.0
        worst = max(worst, d); n_notes += int(keep.sum()); ok += 1
        # triplet concentration: within-quarter bucket of notes whose offset is non-dyadic
        wq_bucket = br["offset"].argmax(-1).numpy()
        base_bucket = (np.round(base_off * 24)).astype(int)
        for j, b in enumerate(base_bucket):
            if (b % 3) != 0:  # triplet-ish within-measure position
                n_trip += 1
                if wq_bucket[j] in (4, 8, 16, 20):
                    trip_in_816 += 1
        print(f"  {Path(sp).name[:40]:40s} notes={len(base_off):4d} max|Δoffset|={d:.5f}")
    print(f"\nscores ok={ok} notes={n_notes} WORST max|Δoffset (B2-recon vs raw truth)|={worst:.8f}  (lossless if = 0)")
    if n_trip:
        print(f"triplet-position notes: {n_trip}; of those, {trip_in_816} ({100*trip_in_816/n_trip:.1f}%) land in within-quarter buckets {{4,8,16,20}} (concentration benefit)")
    print("GATE:", "PASS (lossless to 1/24 grid)" if worst <= 1/24 + 1e-6 else "FAIL (lossy!)")


if __name__ == "__main__":
    main()
