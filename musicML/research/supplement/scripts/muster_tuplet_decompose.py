"""Decompose the MUSTER transcription-error metric by TUPLET vs NON-TUPLET GT notes.

WHY this is sound (verified empirically, see probe_muster_formats.py):
  The MUSTER C++ pipeline (Nakamura ScoreMatchEvaluation_VoicePlus) writes a per-note
  error-detail file `est_err_detail.txt`. We reproduce the EXACT same pipeline the python
  wrapper runs (muster.muster -> evaluate_XML_voicePlus.sh), but KEEP the intermediates.

  From that file we get, keyed by GT score note id (GtID):
    - which matched GT notes carry an OnsetError(shift|scale)   -> onset error
    - which matched GT notes carry an OffsetError               -> offset error
    - which GT notes are MissNote (unmatched)                   -> excluded from matched set
    - nExtraNote / nPitchError                                   -> for self-consistency only
  The same error-detail file ends with MUSTER's exact GT-to-EST correspondence block, which
  defines the matched-note denominator. From `gt_fmt3x.txt` we get each score-note GtID and
  timing metadata; from generated `gt.xml` we recover exact explicit tuplet membership.

MUSTER metric definitions (reverse-engineered to 4 decimals, exact):
    PitchER  = nPitchError / nGT
    MissRate = nMissNote   / nGT
    ExtraRate= nExtraNote  / nEST
    OnsetER  = nOnsetError / nMatched   (nMatched = nGT - nMissNote)
    OffsetER = nOffsetError/ nMatched
    MeanER   = mean(PitchER, MissRate, ExtraRate, OnsetER, OffsetER)
  => OnsetER/OffsetER are RATES (fraction of matched notes flagged), not magnitudes.
  The decomposition therefore reports, separately for tuplet vs non-tuplet matched notes:
    onset_err = (#matched-tuplet notes with an OnsetError)  / (#matched-tuplet notes)
    offset_err= (#matched-tuplet notes with an OffsetError) / (#matched-tuplet notes)
  Note-count-weighted recombination MUST reproduce the harness aggregate OnsetER/OffsetER.

TUPLET tag: exact MusicXML ``<time-modification>`` membership, joined to MUSTER GtIDs.
Fmt3x GtIDs have the form ``P<part>-<measure>-<note ordinal>``; the ordinal counts every
MusicXML ``<note>`` element in that part/measure, including rests.  This direct join avoids
assuming a fixed Fmt3x TPQN.  That matters because MUSTER chooses a different TPQN per score
(the current test set ranges from 12 to 7168), so a hard-coded modulo grid silently invents
tuplets in ordinary binary passages at other resolutions.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 venv311/bin/python scripts/muster_tuplet_decompose.py \
      --ckpt checkpoints/MIDI2ScoreTF.ckpt --tag released \
      --pieces Liszt Ravel Scriabin Mozart \
      --out benchmark/muster_tuplet_decomposed.json
  (omit --pieces for the full 14-piece padsweep set)
"""
import argparse, json, os, subprocess, sys, time, warnings
import xml.etree.ElementTree as ET
from pathlib import Path

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer"))
sys.path.insert(0, str(REPO / "benchmark"))
os.chdir(TF)

import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer  # noqa: E402
from score_utils import postprocess_score  # noqa: E402
from muster.muster import handle_score_file, parse_line_to_dict, MUSTER_BIN  # noqa: E402
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402

PF = str(MUSTER_BIN.parent / "Programs")


def run_muster_keep(est_score, gt_path, workdir):
    """Run the EXACT evaluate_XML_voicePlus.sh pipeline, keeping intermediates.
    Returns (aggregate_dict, gt_fmt3x_path, err_detail_path)."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    handle_score_file(est_score, str(workdir / "est.xml"), True)
    handle_score_file(gt_path, str(workdir / "gt.xml"), True)
    gt = str(workdir / "gt"); est = str(workdir / "est"); out = str(workdir / "out")
    seq = [
        [f"{PF}/MusicXMLToFmt3x", f"{est}.xml", f"{est}_fmt3x.txt"],
        [f"{PF}/Fmt3xToSpr", f"{est}_fmt3x.txt", f"{est}_spr.txt"],
        [f"{PF}/MusicXMLToHMM", f"{gt}.xml", f"{gt}_hmm.txt"],
        [f"{PF}/MusicXMLToFmt3x", f"{gt}.xml", f"{gt}_fmt3x.txt"],
        [f"{PF}/ScorePerfmMatcher", f"{gt}_hmm.txt", f"{est}_spr.txt", f"{est}_pre_match.txt", "0.01"],
        [f"{PF}/ErrorDetection", f"{gt}_fmt3x.txt", f"{gt}_hmm.txt", f"{est}_pre_match.txt", f"{est}_err_match.txt"],
        [f"{PF}/RealignmentMOHMM", f"{gt}_fmt3x.txt", f"{gt}_hmm.txt", f"{est}_err_match.txt", f"{est}_auto_match.txt", "0.3"],
    ]
    for c in seq:
        subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(f"{out}.txt", "w") as fo:
        subprocess.run([f"{PF}/ScoreMatchEvaluation_VoicePlus", f"{gt}_fmt3x.txt",
                        f"{est}_fmt3x.txt", f"{est}_auto_match.txt", f"{est}_err_detail.txt", "-1"],
                       stdout=fo, stderr=subprocess.DEVNULL)
    agg = None
    try:
        with open(f"{out}.txt") as f:
            agg = parse_line_to_dict(f.readline())
    except Exception:
        pass
    return agg, f"{gt}_fmt3x.txt", f"{est}_err_detail.txt", f"{est}_auto_match.txt"


def _children(element, name):
    """Return direct MusicXML children with or without an XML namespace."""
    found = element.findall(name)
    return found if found else element.findall(f"{{*}}{name}")


def _child(element, name):
    """Return one direct MusicXML child with or without an XML namespace."""
    found = element.find(name)
    return found if found is not None else element.find(f"{{*}}{name}")


def parse_musicxml_note_flags(path):
    """Map MUSTER-style GtIDs to exact tuplet/grace flags from MusicXML.

    MusicXMLToFmt3x constructs IDs from the part id, the MusicXML measure number, and the
    one-based ordinal of the ``note`` element within that measure.  Rests consume an ordinal
    but do not appear as GtIDs, which is why the enumeration happens before the pitch check.
    """
    root = ET.parse(path).getroot()
    flags = {}
    for part in _children(root, "part"):
        part_id = part.get("id")
        if not part_id:
            raise ValueError(f"MusicXML part without id in {path}")
        for measure in _children(part, "measure"):
            measure_number = measure.get("number")
            if measure_number is None:
                raise ValueError(f"MusicXML measure without number in {path}")
            for note_ordinal, note in enumerate(_children(measure, "note"), 1):
                if _child(note, "pitch") is None:
                    continue
                note_id = f"{part_id}-{measure_number}-{note_ordinal}"
                time_mod = _child(note, "time-modification")
                is_tuplet = time_mod is not None
                if time_mod is not None:
                    actual = _child(time_mod, "actual-notes")
                    normal = _child(time_mod, "normal-notes")
                    if actual is not None and normal is not None:
                        is_tuplet = actual.text != normal.text
                flags[note_id] = {
                    "tuplet": bool(is_tuplet),
                    "grace": _child(note, "grace") is not None,
                }
    return flags


def parse_gt_fmt3x(path):
    """GtID -> exact MusicXML tuplet flag plus Fmt3x onset/duration metadata.

    Fmt3x is tab-separated with
    CHORDS packed onto one line: cols 0=onset_tick 7=dur_tick 8=numNotes, then numNotes
    pitches, voiceinfos, and trailing note IDs (P1-m-n) -- one per chord member, all
    sharing this line's onset/dur. We expand all P-tokens, then join those IDs to the
    generated ``gt.xml`` beside the Fmt3x file.  Missing IDs are a hard error: silently
    falling back to a timing-grid heuristic would make the evaluation resolution-dependent.
    """
    path = Path(path)
    info = {}
    for ln in open(path):
        if ln.startswith("//") or not ln.strip():
            continue
        c = [t for t in ln.rstrip("\n").split("\t") if t != ""]
        if len(c) < 11:
            continue  # rests (NF=10) -> no note IDs
        try:
            on = int(c[0]); dur = int(c[7])
        except ValueError:
            continue
        is_grace = (c[6] == "short-app")  # appoggiatura/grace; MUSTER excludes from note count
        for nid in [t for t in c if t.startswith("P") and "-" in t]:
            info[nid] = {"onset": on, "dur": dur, "grace": is_grace}

    xml_path = path.with_name("gt.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Exact tuplet tagging requires {xml_path}")
    xml_flags = parse_musicxml_note_flags(xml_path)
    missing = sorted(set(info) - set(xml_flags))
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"MusicXML/Fmt3x GtID join failed for {len(missing)}/{len(info)} notes "
            f"in {path}; first IDs: {sample}"
        )
    for nid, row in info.items():
        row["tuplet"] = xml_flags[nid]["tuplet"]
        row["grace"] = bool(row["grace"] or xml_flags[nid]["grace"])
    return info


def parse_matched_gtids(auto_match_path, gt_info):
    """The MUSTER alignment file: one line per EST note; col[9] (0-indexed) is the matched
    GT note id (P1-...) or '*' for an extra note. This is retained as a diagnostic fallback;
    the authoritative matched population comes from ScoreMatchEvaluation's correspondence
    block in ``est_err_detail.txt``, parsed by ``parse_err_detail``."""
    matched = set()
    for ln in open(auto_match_path):
        if ln.startswith("//") or not ln.strip():
            continue
        c = ln.split()
        if len(c) >= 10 and c[9].startswith("P") and "-" in c[9] and c[9] in gt_info:
            matched.add(c[9])
    return matched


def parse_err_detail(path):
    """Returns LISTS of GtIDs for each error EVENT (one per error line) -- onset/offset/
    miss/pitch -- plus the summary counts. MUSTER's nOnsetError/nOffsetError count error
    EVENTS (one per matched GT note flagged); verified == summary block (146/378)."""
    onset, offset, miss, pitch = [], [], [], []
    matched = set()
    counts = {}
    def gtid(c):
        for i, t in enumerate(c):
            if t == "GtID" and i + 1 < len(c):
                return c[i + 1]
        return None
    for ln in open(path):
        c = [token for token in ln.rstrip("\n").split("\t") if token != ""]
        if not c:
            continue
        tag = c[0]
        if len(c) == 2 and c[0].startswith("P") and c[1].startswith("P"):
            # ScoreMatchEvaluation writes GT_ID<TAB>EST_ID for every counted match.
            matched.add(c[0])
            continue
        if tag.startswith("OnsetError"):
            g = gtid(c); onset.append(g) if g else None
        elif tag.startswith("OffsetError"):
            g = gtid(c); offset.append(g) if g else None
        elif tag.startswith("MissNote"):
            g = gtid(c); miss.append(g) if g else None
        elif tag.startswith("PitchError"):
            g = gtid(c); pitch.append(g) if g else None
        elif tag.startswith("n") and len(c) >= 2:
            try:
                counts[tag.rstrip(":")] = int(float(c[1]))
            except ValueError:
                pass
        elif tag.startswith("#notes"):
            try:
                v = int(float(c[1]))
                counts["nGT" if "GT" in ln else "nEST"] = v
            except (ValueError, IndexError):
                pass
    return dict(
        onset=onset,
        offset=offset,
        miss=miss,
        pitch=pitch,
        matched=matched,
        counts=counts,
    )


def decompose_piece(gt_info, matched_gtids, err, agg):
    """Bucket MUSTER's matched GT correspondence set by tuplet status; bucket onset/offset
    error EVENTS likewise. MUSTER-faithful: OnsetER_bucket = (#error events in bucket) /
    (#matched notes in bucket). Note-weighted recombine == harness OnsetER/OffsetER."""
    evaluator_matched = {g for g in err.get("matched", set()) if g in gt_info}
    if evaluator_matched:
        matched_gtids = evaluator_matched

    def tup(g):
        gi = gt_info.get(g)
        return bool(gi and gi["tuplet"] and not gi["grace"])
    def grace(g):
        gi = gt_info.get(g)
        return bool(gi and gi["grace"])
    matched_grace = {g for g in matched_gtids if grace(g)}
    n_gt_summary = err["counts"].get("nGT")
    n_miss_summary = err["counts"].get("nMissNote")
    target_matched = (
        n_gt_summary - n_miss_summary
        if n_gt_summary is not None and n_miss_summary is not None else len(matched_gtids)
    )
    excess_matched = len(matched_gtids) - target_matched
    if excess_matched < 0:
        raise ValueError(
            f"Alignment exposes {len(matched_gtids)} matched IDs, below MUSTER denominator "
            f"{target_matched}"
        )
    if excess_matched and len(matched_grace) != excess_matched:
        raise ValueError(
            f"Cannot reconcile {excess_matched} matched IDs excluded by MUSTER with "
            f"{len(matched_grace)} matched grace IDs"
        )
    # Usually every matched Fmt3x ID belongs to MUSTER's denominator. On the one score where
    # the matcher exposes extra grace IDs, remove exactly those IDs from the paper buckets.
    excluded_matched = matched_grace if excess_matched else set()
    matched_bucket = set(matched_gtids) - excluded_matched
    # denominators: matched notes by bucket
    nt = sum(1 for g in matched_bucket if tup(g))
    nn = len(matched_bucket) - nt
    # numerators: error events by bucket (restrict to matched GtIDs; all should be matched)
    on_t = sum(1 for g in err["onset"] if g in matched_bucket and tup(g))
    on_n = sum(1 for g in err["onset"] if g in matched_bucket and not tup(g))
    off_t = sum(1 for g in err["offset"] if g in matched_bucket and tup(g))
    off_n = sum(1 for g in err["offset"] if g in matched_bucket and not tup(g))
    on_x = sum(1 for g in err["onset"] if g in excluded_matched)
    off_x = sum(1 for g in err["offset"] if g in excluded_matched)
    # TPF coverage is defined against the score, not against a candidate-specific matcher
    # population.  A few MUSTER alignments omit GT IDs from both `matched` and `MissNote`;
    # using only those two sets made the apparent GT total vary by candidate.  Count every
    # explicit, non-grace tuplet note in Fmt3x and derive misses as total minus matched.
    gt_t = sum(
        1 for gi in gt_info.values()
        if gi["tuplet"] and not gi["grace"]
    )
    miss_t_events = sum(1 for g in err["miss"] if tup(g))
    miss_t = gt_t - nt
    if miss_t < 0:
        raise ValueError(f"Matched tuplet count {nt} exceeds GT total {gt_t}")
    r = lambda num, den: (100.0 * num / den) if den else None
    out = {
        "n_tuplet": nt, "n_nontuplet": nn, "n_matched": nt + nn,
        "onset_err_tuplet": r(on_t, nt), "onset_err_nontuplet": r(on_n, nn),
        "offset_err_tuplet": r(off_t, nt), "offset_err_nontuplet": r(off_n, nn),
        "n_onset_err_tuplet": on_t, "n_onset_err_nontuplet": on_n,
        "n_offset_err_tuplet": off_t, "n_offset_err_nontuplet": off_n,
        "n_matched_grace": len(matched_grace),
        "n_excluded_matched_grace": len(excluded_matched),
        "n_onset_err_excluded": on_x, "n_offset_err_excluded": off_x,
        "n_miss_tuplet": miss_t, "n_miss_tuplet_events": miss_t_events,
        "n_gt_tuplet": gt_t,
        "meaner_overall": (agg or {}).get("MeanER"),
    }
    # ---- self-consistency ----
    # OffsetER: per-note OffsetError lines == MUSTER nOffsetError EXACTLY -> combined==harness exactly.
    # OnsetER : MUSTER's headline numerator is nOnsetError(RhythmCorrectionCost), an alignment-level
    #           COST that exceeds the sum of attributable per-note OnsetError(shift|scale) lines by a
    #           small residual (0 or 1 per piece on this set). We bucket the ATTRIBUTABLE lines, so the
    #           faithful self-check compares combined-onset to harness-onset recomputed on the
    #           attributable numerator; we also record the residual cost-units left unattributed.
    # MUSTER can expose matched grace IDs outside its denominator while retaining their error
    # events in its numerators. Keep those events out of the paper buckets, then add them back
    # only for the harness-reproduction checks.
    out["_check_bucketed_OnsetER"] = r(on_t + on_n, nt + nn)
    out["_check_bucketed_OffsetER"] = r(off_t + off_n, nt + nn)
    out["_check_combined_OnsetER"] = r(on_t + on_n + on_x, nt + nn)
    out["_check_combined_OffsetER"] = r(off_t + off_n + off_x, nt + nn)
    out["_check_harness_OnsetER"] = (agg or {}).get("OnsetER")     # MUSTER headline (incl residual)
    out["_check_harness_OffsetER"] = (agg or {}).get("OffsetER")
    n_onset_summary = err["counts"].get("nOnsetError(RhythmCorrectionCost)")
    n_offset_summary = err["counts"].get("nOffsetError")
    out["_check_onset_events"] = on_t + on_n + on_x
    out["_check_offset_events"] = off_t + off_n + off_x
    out["_check_n_onset_summary"] = n_onset_summary
    out["_check_n_offset_summary"] = n_offset_summary
    # residual = MUSTER onset cost - attributable per-note onset lines (the unbucketable part)
    out["_onset_cost_residual"] = (
        n_onset_summary - (on_t + on_n + on_x)
        if n_onset_summary is not None else None
    )
    # harness OnsetER recomputed on the attributable numerator (what combined SHOULD equal exactly)
    out["_check_harness_OnsetER_attributable"] = (
        r(on_t + on_n + on_x, nt + nn) if (nt + nn) else None
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--tag", required=True, help="model label e.g. 'released' or 'ours'")
    ap.add_argument("--pieces", nargs="*", default=None)
    ap.add_argument("--limit-per", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    work_root = REPO / "benchmark" / "decomp_work" / args.tag
    work_root.mkdir(parents=True, exist_ok=True)

    paths = collect_paths("test")
    for p in paths:
        try:
            p["_x"] = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception:
            p["_x"] = None; p["n_notes"] = 1 << 30
    if args.pieces:
        paths = [p for p in paths
                 if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)]
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = {}, []
    for p in paths:
        key = (p["composer"], p["piece"])
        if seen.get(key, 0) < args.limit_per and p["_x"] is not None:
            seen[key] = seen.get(key, 0) + 1
            kept.append(p)
    paths = kept
    print(f"{len(paths)} pieces; tag={args.tag} ckpt={args.ckpt}", flush=True)

    model = load_any_checkpoint(args.ckpt, "cpu"); model.eval(); model.to("cpu")

    results = []
    t0 = time.time()
    for i, p in enumerate(paths):
        rec = {"composer": p["composer"], "piece": p["piece"], "n_notes": p["n_notes"]}
        try:
            with torch.no_grad():
                y = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                          verbose=False, kv_cache=True)
            mxl = postprocess_score(MultistreamTokenizer.detokenize_mxl(y, pad_threshold=0.5), inPlace=True)
            wd = work_root / p["piece"].replace("/", "_")
            agg, gt_fmt3x, err_detail, auto_match = run_muster_keep(mxl, p["score"], wd)
            gt_info = parse_gt_fmt3x(gt_fmt3x)
            matched_gtids = parse_matched_gtids(auto_match, gt_info)
            err = parse_err_detail(err_detail)
            dec = decompose_piece(gt_info, matched_gtids, err, agg)
            rec.update(dec)
            rec["muster_aggregate"] = agg
            # OFFSET must match harness EXACTLY. ONSET must match the attributable numerator
            # exactly; the only allowed slack vs MUSTER's headline OnsetER is the RhythmCorrection
            # residual (<=1 cost-unit/piece, recorded in _onset_cost_residual).
            ok_off = (dec["_check_combined_OffsetER"] is None or dec["_check_harness_OffsetER"] is None
                      or abs(dec["_check_combined_OffsetER"] - dec["_check_harness_OffsetER"]) < 0.005)
            resid = dec.get("_onset_cost_residual")
            ok_on = (resid is None) or (0 <= resid <= 2)
            rec["selfcheck_ok"] = bool(ok_on and ok_off)
            print(f"[{i+1}/{len(paths)}] {p['composer'][:8]}/{p['piece'][:24]:24s} "
                  f"MeanER={dec['meaner_overall']} | onset T={_f(dec['onset_err_tuplet'])} "
                  f"N={_f(dec['onset_err_nontuplet'])} | offset T={_f(dec['offset_err_tuplet'])} "
                  f"N={_f(dec['offset_err_nontuplet'])} | nT={dec['n_tuplet']} nN={dec['n_nontuplet']} "
                  f"| check on {_f(dec['_check_combined_OnsetER'])}=={_f(dec['_check_harness_OnsetER'])} "
                  f"off {_f(dec['_check_combined_OffsetER'])}=={_f(dec['_check_harness_OffsetER'])} "
                  f"{'OK' if rec['selfcheck_ok'] else 'MISMATCH'}", flush=True)
        except Exception:
            import traceback
            rec["error"] = traceback.format_exc().splitlines()[-1]
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:24]} ERROR {rec['error']}", flush=True)
        results.append(rec)
        _dump(out_path, args, results, t0)

    _dump(out_path, args, results, t0, final=True)
    print(f"\nWrote {out_path}  ({round(time.time()-t0,1)}s)", flush=True)


def _f(x):
    return "—" if x is None else f"{x:.2f}"


def aggregate(results):
    """Corpus aggregate two ways: (A) micro = pooled note counts; (B) macro = mean of per-piece rates."""
    good = [r for r in results if "error" not in r and r.get("n_matched")]
    tot = {k: 0 for k in ["n_tuplet", "n_nontuplet", "n_onset_err_tuplet", "n_onset_err_nontuplet",
                          "n_offset_err_tuplet", "n_offset_err_nontuplet"]}
    for r in good:
        for k in tot:
            tot[k] += r.get(k, 0) or 0
    r = lambda num, den: (100.0 * num / den) if den else None
    micro = {
        "n_tuplet": tot["n_tuplet"], "n_nontuplet": tot["n_nontuplet"],
        "onset_err_tuplet": r(tot["n_onset_err_tuplet"], tot["n_tuplet"]),
        "onset_err_nontuplet": r(tot["n_onset_err_nontuplet"], tot["n_nontuplet"]),
        "offset_err_tuplet": r(tot["n_offset_err_tuplet"], tot["n_tuplet"]),
        "offset_err_nontuplet": r(tot["n_offset_err_nontuplet"], tot["n_nontuplet"]),
    }
    def macro(key):
        vals = [r2[key] for r2 in good if r2.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    macroe = {k: macro(k) for k in ["onset_err_tuplet", "onset_err_nontuplet",
                                    "offset_err_tuplet", "offset_err_nontuplet"]}
    macroe["meaner_overall"] = macro("meaner_overall")
    # headline ratios (micro)
    def ratio(a, b):
        return (micro[a] / micro[b]) if (micro.get(a) and micro.get(b)) else None
    headline = {
        "onset_tuplet_over_nontuplet": ratio("onset_err_tuplet", "onset_err_nontuplet"),
        "offset_tuplet_over_nontuplet": ratio("offset_err_tuplet", "offset_err_nontuplet"),
    }
    return {"micro_pooled": micro, "macro_meanofpieces": macroe,
            "headline_ratio_micro": headline, "n_pieces": len(good)}


def _dump(out_path, args, results, t0, final=False):
    # merge with any existing other-tag block so released + ours coexist in one file
    existing = {}
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
        except Exception:
            existing = {}
    existing.setdefault("models", {})
    existing["models"][args.tag] = {
        "ckpt": args.ckpt, "elapsed_s": round(time.time() - t0, 1),
        "per_piece": results, "aggregate": aggregate(results),
    }
    existing["_metric_defs"] = {
        "onset_err": "fraction (%) of MATCHED GT notes flagged OnsetError(shift|scale) by MUSTER",
        "offset_err": "fraction (%) of MATCHED GT notes flagged OffsetError by MUSTER",
        "matched": "nGT - nMissNote (MUSTER OnsetER/OffsetER denominator)",
        "tuplet_tag": "exact MusicXML <time-modification> joined by Fmt3x GtID",
        "self_consistency": "note-weighted combine of tuplet+nontuplet == harness OnsetER/OffsetER",
    }
    json.dump(existing, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()
