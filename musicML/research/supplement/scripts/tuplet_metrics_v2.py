"""Resolution-independent tuplet metrics for MUSTER outputs.

This module is deliberately separate from the frozen V10.1 selector dependencies.  It uses
the exact MusicXML time-modification tags exposed by ``muster_tuplet_decompose`` and counts
TPF directly from MUSTER's missed-note and onset-error events:

    TPF = missed explicit-tuplet GT notes + onset-error events on matched tuplet GT notes.
"""

import verifier_bestofn as VB


def muster_piece(y, piece, workdir):
    mxl = VB.postprocess_score(
        VB.MultistreamTokenizer.detokenize_mxl(y, pad_threshold=0.5), inPlace=True
    )
    agg, gt_fmt3x, err_detail, auto_match = VB.MTD.run_muster_keep(
        mxl, piece["score"], workdir
    )
    gt_info = VB.MTD.parse_gt_fmt3x(gt_fmt3x)
    matched = VB.MTD.parse_matched_gtids(auto_match, gt_info)
    err = VB.MTD.parse_err_detail(err_detail)
    dec = VB.MTD.decompose_piece(gt_info, matched, err, agg)
    return {
        "meaner_overall": dec["meaner_overall"],
        "onset_err_tuplet": dec["onset_err_tuplet"],
        "onset_err_nontuplet": dec["onset_err_nontuplet"],
        "n_tuplet": dec["n_tuplet"],
        "n_nontuplet": dec["n_nontuplet"],
        "n_onset_err_tuplet": dec["n_onset_err_tuplet"],
        "n_miss_tuplet": dec["n_miss_tuplet"],
        "n_gt_tuplet": dec["n_gt_tuplet"],
    }


def tpf(record):
    missed = int(record["n_miss_tuplet"])
    onset = int(record["n_onset_err_tuplet"])
    return missed + onset, missed, onset
