#!/usr/bin/env python3
"""Piano audio → sheet music transcription pipeline.

Chains: Audio (wav/mp3) → [transcriber] → MIDI → [backend] → PDF/PNG

Transcribers:
  basic-pitch — Spotify Basic Pitch CNN (default)
  mt3         — Google MT3 T5 Transformer (requires venv_mt3)

Backends:
  music21     — music21 heuristics + LilyPond rendering (default)
  musescore   — MuseScore CLI MIDI import + rendering
  transformer — MIDI2ScoreTransformer (neural) + MuseScore rendering
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pretty_midi

log = logging.getLogger("transcribe")

SUPPORTED_AUDIO = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
SUPPORTED_OUTPUT = {".pdf", ".png"}
TRANSCRIBERS = ["basic-pitch", "mt3", "transkun", "hft"]
BACKENDS = ["music21", "musescore", "transformer"]

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer"
CHECKPOINT_PATH = TRANSFORMER_DIR / "checkpoints" / "MIDI2ScoreTF.ckpt"
HFT_DIR = SCRIPT_DIR / "hFT-Transformer"
HFT_CONFIG = HFT_DIR / "corpus" / "config.json"
HFT_CHECKPOINT = HFT_DIR / "checkpoint" / "MAESTRO-V3" / "model_016_003.pkl"


# ---------------------------------------------------------------------------
# Phase 1: Audio → MIDI
# ---------------------------------------------------------------------------

def preprocess_audio(audio_path: Path, target_sr: int = 16000,
                     peak_dbfs: float = -3.0) -> Path:
    """Mono downmix + resample + peak normalize. Writes a sibling .pre.wav file
    and returns its path. Helpful for hFT, which is sensitive to recording
    level and expects 16 kHz mono.

    Peak normalization (not LUFS) is used to keep the dependency surface tiny
    and to avoid skewing the velocity prediction more than necessary.
    """
    import numpy as np
    import soxr
    try:
        import soundfile as sf
    except ImportError:
        log.error("--normalize-audio requires soundfile. Install with: pip install soundfile")
        sys.exit(1)

    out_path = audio_path.with_suffix(".pre.wav")
    log.info("      Preprocessing audio (mono, %d Hz, peak %.1f dBFS)...", target_sr, peak_dbfs)

    audio, sr = sf.read(str(audio_path), always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)

    peak = float(np.max(np.abs(audio))) or 1.0
    target_peak = 10.0 ** (peak_dbfs / 20.0)
    audio = audio * (target_peak / peak)

    sf.write(str(out_path), audio.astype("float32"), target_sr, subtype="PCM_16")
    log.info("      Preprocessed audio: %s", out_path)
    return out_path


def audio_to_midi_basic_pitch(audio_path: Path, midi_path: Path,
                              onset_threshold: float = 0.5,
                              frame_threshold: float = 0.3) -> int:
    """Run Basic Pitch inference and write a MIDI file."""
    log.info("[1/3] Audio → MIDI (Basic Pitch)")
    log.info("      Input:  %s", audio_path)
    log.info("      Thresholds: onset=%.2f, frame=%.2f", onset_threshold, frame_threshold)

    _root = logging.getLogger()
    _prev_level = _root.level
    _root.setLevel(logging.ERROR)

    import io, contextlib, warnings
    with contextlib.redirect_stderr(io.StringIO()), \
         contextlib.redirect_stdout(io.StringIO()), \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from basic_pitch.inference import predict
        model_output, midi_data, note_events = predict(
            str(audio_path),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

    _root.setLevel(_prev_level)
    midi_data.write(str(midi_path))

    n_notes = len(note_events)
    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes


def audio_to_midi_mt3(audio_path: Path, midi_path: Path) -> int:
    """Run MT3 Transformer inference and write a MIDI file."""
    log.info("[1/3] Audio → MIDI (MT3)")
    log.info("      Input:  %s", audio_path)

    from mt3_inference import transcribe_audio
    n_notes = transcribe_audio(str(audio_path), str(midi_path))

    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes


def audio_to_midi_transkun(audio_path: Path, midi_path: Path) -> int:
    """Run Transkun inference and write a MIDI file."""
    log.info("[1/3] Audio → MIDI (Transkun)")
    log.info("      Input:  %s", audio_path)

    import pkg_resources
    import torch
    import numpy as np
    import moduleconf
    from transkun.transcribe import readAudio, writeMidi

    default_weight = pkg_resources.resource_filename("transkun", "pretrained/2.0.pt")
    default_conf = pkg_resources.resource_filename("transkun", "pretrained/2.0.conf")

    device = "cpu"
    log.info("      Device: %s", device)

    conf_manager = moduleconf.parseFromFile(default_conf)
    TransKun = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    checkpoint = torch.load(default_weight, map_location=device)
    model = TransKun(conf=conf).to(device)
    if "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    fs, audio = readAudio(str(audio_path))
    if fs != model.fs:
        import soxr
        audio = soxr.resample(audio, fs, model.fs)

    x = torch.from_numpy(audio).to(device)
    with torch.no_grad():
        notes_est = model.transcribe(x)

    output_midi = writeMidi(notes_est)
    output_midi.write(str(midi_path))

    n_notes = len([n for n in notes_est if n.pitch > 0])
    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes


def audio_to_midi_hft(audio_path: Path, midi_path: Path,
                      thred_onset: float = 0.5, thred_mpe: float = 0.5,
                      thred_offset: float = 0.5,
                      mode_offset: str = "shorter") -> int:
    """Run hFT-Transformer inference and write a MIDI file.

    thred_onset / thred_mpe / thred_offset: detection thresholds (default 0.5).
    Lowering onset/mpe (~0.3-0.4) rescues soft/dense notes the 0.5 default drops
    — the single biggest Phase-1 recall knob. mode_offset ('shorter'|'offset'|
    'longer') controls note-duration estimation; 'shorter' clips sustained notes.
    """
    log.info("[1/3] Audio → MIDI (hFT-Transformer)")
    log.info("      Input:  %s", audio_path)

    if not HFT_DIR.is_dir():
        log.error("hFT-Transformer repo not found at %s", HFT_DIR)
        sys.exit(1)
    if not HFT_CHECKPOINT.is_file():
        log.error("hFT checkpoint not found at %s", HFT_CHECKPOINT)
        sys.exit(1)

    import json
    import pickle
    import torch
    import numpy as np

    hft_root = str(HFT_DIR)
    hft_model_dir = str(HFT_DIR / "model")
    if hft_root not in sys.path:
        sys.path.insert(0, hft_root)
    if hft_model_dir not in sys.path:
        sys.path.insert(0, hft_model_dir)
    from amt import AMT

    with open(str(HFT_CONFIG), "r") as f:
        config = json.load(f)
    config["input"]["min_value"] = float(np.log(config["feature"]["log_offset"]))

    amtobj = AMT(config, model_path=None)
    import io
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                            weights_only=False)
            return super().find_class(module, name)
    with open(str(HFT_CHECKPOINT), "rb") as f:
        model = CPUUnpickler(f).load()
    amtobj.model = model.to(amtobj.device)
    amtobj.model.eval()
    # Override hardcoded CUDA device refs in model submodules
    for module in amtobj.model.modules():
        if hasattr(module, "device") and isinstance(module.device, str):
            module.device = amtobj.device
    log.info("      Device: %s", amtobj.device)

    a_feature = amtobj.wav2feature(str(audio_path))
    log.info("      Feature frames: %d", a_feature.shape[0])

    out = amtobj.transcript(a_feature, mode="combination")
    onset_A, offset_A, mpe_A, vel_A = out[0], out[1], out[2], out[3]
    onset_B, offset_B, mpe_B, vel_B = out[4], out[5], out[6], out[7]

    log.info("      hFT thresholds: onset=%.2f mpe=%.2f offset=%.2f mode_offset=%s",
             thred_onset, thred_mpe, thred_offset, mode_offset)
    a_note = amtobj.mpe2note(
        a_onset=onset_B, a_offset=offset_B, a_mpe=mpe_B, a_velocity=vel_B,
        thred_onset=thred_onset, thred_offset=thred_offset, thred_mpe=thred_mpe,
        mode_velocity="ignore_zero", mode_offset=mode_offset,
    )

    amtobj.note2midi(a_note, str(midi_path))

    n_notes = len(a_note)
    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes, a_note


def audio_to_midi(audio_path: Path, midi_path: Path, transcriber: str = "basic-pitch",
                  onset_threshold: float = 0.5, frame_threshold: float = 0.3,
                  hft_onset_threshold: float = 0.5, hft_mpe_threshold: float = 0.5,
                  hft_offset_threshold: float = 0.5, hft_offset_mode: str = "shorter"):
    """Dispatch to the chosen transcriber. Returns (n_notes, raw_notes_or_None)."""
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    if transcriber == "basic-pitch":
        return audio_to_midi_basic_pitch(audio_path, midi_path,
                                         onset_threshold=onset_threshold,
                                         frame_threshold=frame_threshold), None
    elif transcriber == "mt3":
        return audio_to_midi_mt3(audio_path, midi_path), None
    elif transcriber == "transkun":
        return audio_to_midi_transkun(audio_path, midi_path), None
    elif transcriber == "hft":
        return audio_to_midi_hft(audio_path, midi_path,
                                 thred_onset=hft_onset_threshold,
                                 thred_mpe=hft_mpe_threshold,
                                 thred_offset=hft_offset_threshold,
                                 mode_offset=hft_offset_mode)
    else:
        raise ValueError(f"Unknown transcriber: {transcriber}")


def filter_midi(midi_path: Path, min_duration_ms: float = 50,
                min_velocity: int = 10, pitch_range: tuple = (21, 108),
                debounce_ms: float = 30) -> int:
    """Remove spurious notes from a MIDI file in-place.

    Returns the number of notes removed.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    total_before = sum(len(inst.notes) for inst in pm.instruments)

    for inst in pm.instruments:
        filtered = []
        for note in inst.notes:
            duration_ms = (note.end - note.start) * 1000
            if duration_ms < min_duration_ms:
                continue
            if note.velocity < min_velocity:
                continue
            if note.pitch < pitch_range[0] or note.pitch > pitch_range[1]:
                continue
            filtered.append(note)

        filtered.sort(key=lambda n: (n.pitch, n.start))
        deduped = []
        for note in filtered:
            if deduped and deduped[-1].pitch == note.pitch:
                gap_ms = (note.start - deduped[-1].start) * 1000
                if gap_ms < debounce_ms:
                    if note.velocity > deduped[-1].velocity:
                        deduped[-1] = note
                    continue
            deduped.append(note)

        inst.notes = deduped

    total_after = sum(len(inst.notes) for inst in pm.instruments)
    pm.write(str(midi_path))

    removed = total_before - total_after
    log.info("      MIDI filter: %d → %d notes (%d removed)", total_before, total_after, removed)
    return removed


# ---------------------------------------------------------------------------
# Phase 2 + 3 helpers
# ---------------------------------------------------------------------------

def find_lilypond() -> Optional[str]:
    """Locate the LilyPond binary."""
    path = shutil.which("lilypond")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/lilypond", "/usr/local/bin/lilypond"]:
        if Path(candidate).is_file():
            return candidate
    return None


def find_musescore() -> Optional[str]:
    """Locate the MuseScore binary."""
    env_path = os.environ.get("MUSICML_MUSESCORE")
    if env_path and Path(env_path).is_file():
        return env_path
    path = shutil.which("mscore")
    if path:
        return path
    for candidate in [
        "/opt/homebrew/bin/mscore",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/usr/local/bin/mscore",
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


def run_musescore(mscore_path: str, input_path: str, output_path: str) -> bool:
    """Run MuseScore in converter mode. Returns True on success."""
    env = os.environ.copy()
    if sys.platform == "linux":
        env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [mscore_path, "-o", output_path, input_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if result.returncode != 0:
        log.warning("      MuseScore CLI failed (exit code %d)", result.returncode)
        return False
    return True


def run_musescore_chunked(mscore_path: str, mxl_path: str, output_path: str,
                          chunk_size: int = 10) -> bool:
    """Fallback renderer: split score into chunks, render each, merge PDFs.

    MuseScore CLI crashes on large/complex MusicXML files. This splits the
    score into small measure chunks, renders each independently, and merges
    the resulting PDFs. Chunks that fail are skipped with a warning.
    """
    import tempfile
    from music21 import converter as m21_converter

    log.info("      Attempting chunked rendering (%d measures per chunk)...", chunk_size)
    score = m21_converter.parse(mxl_path)
    parts = score.parts
    if not parts:
        return False

    max_measures = max(len(list(p.getElementsByClass("Measure"))) for p in parts)
    pdfs = []
    failed_ranges = []

    for start in range(0, max_measures, chunk_size):
        end = min(start + chunk_size, max_measures)
        chunk = score.measures(start + 1, end)

        tmp_mxl = tempfile.mktemp(suffix=".musicxml")
        chunk.write("musicxml", fp=tmp_mxl)

        tmp_pdf = tempfile.mktemp(suffix=".pdf")
        if run_musescore(mscore_path, tmp_mxl, tmp_pdf):
            pdfs.append(tmp_pdf)
        else:
            failed_ranges.append(f"m{start+1}-{end}")

        os.unlink(tmp_mxl)

    if not pdfs:
        log.error("      All chunks failed to render.")
        return False

    if failed_ranges:
        log.warning("      Skipped chunks: %s", ", ".join(failed_ranges))

    if len(pdfs) == 1:
        os.rename(pdfs[0], output_path)
    else:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for pdf_path in pdfs:
            writer.append(pdf_path)
        writer.write(output_path)
        writer.close()
        for p in pdfs:
            os.unlink(p)

    log.info("      Chunked render: %d/%d chunks OK", len(pdfs), len(pdfs) + len(failed_ranges))
    return True


# ---------------------------------------------------------------------------
# Backend: music21 (heuristics + LilyPond)
# ---------------------------------------------------------------------------

def backend_music21(midi_path: Path, output_path: Path) -> Path:
    from music21 import converter, environment, note, chord

    log.info("[2/3] MIDI → Score (music21)")
    score = converter.parse(str(midi_path))

    key = score.analyze("key")
    log.info("      Key: %s", key)

    time_sigs = score.recurse().getElementsByClass("TimeSignature")
    ts_str = time_sigs[0].ratioString if time_sigs else "not detected"
    log.info("      Time signature: %s", ts_str)

    parts = score.parts
    log.info("      Parts: %d", len(parts))

    notes = score.recurse().getElementsByClass((note.Note, chord.Chord))
    log.info("      Notes/chords: %d", len(notes))

    tempos = score.recurse().getElementsByClass("MetronomeMark")
    if tempos:
        t = tempos[0]
        log.info("      Tempo: %s BPM", int(t.number) if t.number else t)

    log.info("[3/3] Score → Sheet Music (LilyPond)")
    ext = output_path.suffix.lower()
    lily_fmt = "lily.pdf" if ext == ".pdf" else "lily.png"

    lilypond_path = find_lilypond()
    if lilypond_path:
        log.info("      LilyPond: %s", lilypond_path)
        env = environment.Environment()
        env["lilypondPath"] = lilypond_path
        try:
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=output_path.suffix, dir=str(output_path.parent))
            os.close(tmp_fd)
            os.unlink(tmp_path)
            tmp_stem = tmp_path.rsplit(".", 1)[0]

            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull_fd, 2)
            try:
                written = Path(str(score.write(lily_fmt, fp=tmp_stem)))
            finally:
                os.dup2(saved_stderr, 2)
                os.close(devnull_fd)
                os.close(saved_stderr)
            if written.exists():
                written.rename(output_path)
            ly_source = Path(tmp_stem)
            if ly_source.exists():
                ly_source.unlink()
            log.info("      Output: %s", output_path)
            return output_path
        except Exception as exc:
            log.warning("      LilyPond rendering failed: %s", exc)
            log.warning("      Falling back to MusicXML export.")
    else:
        log.warning("      LilyPond not found. Falling back to MusicXML export.")

    xml_path = output_path.with_suffix(".musicxml")
    written = Path(str(score.write("musicxml", fp=str(xml_path))))
    log.info("      Output (MusicXML): %s", written)
    return written


# ---------------------------------------------------------------------------
# Backend: MuseScore CLI
# ---------------------------------------------------------------------------

def backend_musescore(midi_path: Path, output_path: Path) -> Path:
    mscore = find_musescore()
    if not mscore:
        log.error("MuseScore not found. Install via: brew install --cask musescore")
        sys.exit(1)

    log.info("[2/3] MIDI → Score + Render (MuseScore)")
    log.info("      MuseScore: %s", mscore)

    if not run_musescore(mscore, str(midi_path), str(output_path)):
        log.error("      MuseScore failed to render %s", midi_path)
        sys.exit(1)

    log.info("      Output: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Backend: MIDI2ScoreTransformer
# ---------------------------------------------------------------------------

def _fix_time_signatures(score, diagnostics: list):
    """Override time signatures with the majority-predicted meter.

    MIDI2ScoreTransformer's downbeat predictions can be noisy — e.g. the first
    few measures may predict 3/4 while the rest correctly predict 4/4. This
    function takes a majority vote over all predicted bar lengths and replaces
    outlier time signatures.

    NOTE: This assumes a single predominant time signature for the whole piece.
    Pieces with genuine meter changes (Bartók, some Prokofiev, late Liszt)
    would have intentional time signature changes overridden. Not solved here.
    """
    from music21 import meter
    from collections import Counter

    if not diagnostics:
        return score

    bar_lengths = [d["bar_length_ql"] for d in diagnostics]
    rounded = [round(bl) for bl in bar_lengths]
    majority_ql = Counter(rounded).most_common(1)[0][0]

    if majority_ql <= 0:
        return score

    frac_map = {1: "1/4", 2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/4"}
    if majority_ql not in frac_map:
        return score

    majority_ts_str = frac_map[majority_ql]
    log.info("      Time sig fix: majority vote = %s (%d/%d measures)",
             majority_ts_str,
             rounded.count(majority_ql), len(rounded))

    for part in score.parts:
        first_measure = True
        for m in part.getElementsByClass("Measure"):
            existing = m.getElementsByClass("TimeSignature")
            if first_measure:
                if existing:
                    if existing[0].ratioString != majority_ts_str:
                        m.remove(existing[0])
                        m.insert(0, meter.TimeSignature(majority_ts_str))
                else:
                    m.insert(0, meter.TimeSignature(majority_ts_str))
                first_measure = False
            else:
                for ts in existing:
                    if ts.ratioString != majority_ts_str:
                        m.remove(ts)

    return score


def backend_transformer(midi_path: Path, output_path: Path, raw_notes: list = None,
                        pad_threshold: float = 0.5,
                        top_k: int = 1,
                        temperature: float = 1.0,
                        chunk_size: int = 512,
                        overlap: int = 128,
                        no_render: bool = False) -> Path:
    if not TRANSFORMER_DIR.is_dir():
        log.error("MIDI2ScoreTransformer repo not found at %s", TRANSFORMER_DIR)
        log.error("Clone it: git clone https://github.com/TimFelixBeyer/MIDI2ScoreTransformer.git")
        sys.exit(1)
    if not CHECKPOINT_PATH.is_file():
        log.error("Model checkpoint not found at %s", CHECKPOINT_PATH)
        log.error("Download from: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases")
        sys.exit(1)

    mscore = find_musescore()
    if not mscore:
        log.error("MuseScore not found (required for transformer rendering).")
        log.error("Install via: brew install --cask musescore")
        sys.exit(1)

    log.info("[2/3] MIDI → Score (MIDI2ScoreTransformer)")

    # Add the transformer source to Python path
    transformer_src = str(TRANSFORMER_DIR / "midi2scoretransformer")
    if transformer_src not in sys.path:
        sys.path.insert(0, transformer_src)

    import torch
    from config import MyModelConfig
    # Patch for newer transformers compatibility
    if not hasattr(MyModelConfig, '_attn_implementation_internal'):
        MyModelConfig._attn_implementation_internal = None
    torch.serialization.add_safe_globals([MyModelConfig])

    from tokenizer import MultistreamTokenizer
    from utils import infer
    from score_utils import postprocess_score
    from models.roformer import Roformer

    # MPS produces numerically incorrect pad logits in this model's attention
    # layers, causing every note to be masked out. Force CPU for correctness.
    device = "cpu"
    log.info("      Device: %s", device)
    log.info("      Loading model...")

    model = Roformer.load_from_checkpoint(
        str(CHECKPOINT_PATH), map_location=device, weights_only=False,
    )
    model.eval()
    model.to(device)

    if raw_notes is not None:
        log.info("      Tokenizing notes directly (bypassing MIDI file)...")
        x = MultistreamTokenizer.tokenize_notes(raw_notes)
    else:
        log.info("      Tokenizing MIDI...")
        x = MultistreamTokenizer.tokenize_midi(str(midi_path))

    log.info("      Running inference (%d notes, chunk=%d, overlap=%d, top_k=%d, T=%.2f)...",
             x["pitch"].shape[0], chunk_size, overlap, top_k, temperature)
    y_hat = infer(x, model, verbose=False, kv_cache=True,
                  overlap=overlap, chunk=chunk_size,
                  top_k=top_k, temperature=temperature)

    log.info("      Decoding to score (pad_threshold=%.2f)...", pad_threshold)
    diagnostics = []
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat, _diagnostics=diagnostics, pad_threshold=pad_threshold)
    mxl = _fix_time_signatures(mxl, diagnostics)
    mxl = postprocess_score(mxl, inPlace=True)

    # Write MusicXML, then render to PDF/PNG with MuseScore
    mxl_path = output_path.with_suffix(".musicxml")
    mxl.write("musicxml", fp=str(mxl_path))
    log.info("      MusicXML saved: %s", mxl_path)

    if no_render:
        log.info("      --no-render: skipping MuseScore (MusicXML only)")
        return mxl_path

    log.info("[3/3] Score → Sheet Music (MuseScore)")
    log.info("      MuseScore: %s", mscore)

    if run_musescore(mscore, str(mxl_path), str(output_path)):
        log.info("      Output: %s", output_path)
        return output_path
    else:
        log.warning("      Full render failed. Trying chunked rendering...")
        if run_musescore_chunked(mscore, str(mxl_path), str(output_path)):
            log.info("      Output: %s", output_path)
            return output_path
        else:
            log.warning("      MusicXML preserved at: %s", mxl_path)
            return mxl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio to sheet music.",
        epilog="Example: python transcribe.py input.wav -o score.pdf --backend musescore",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Path to input audio file (wav, mp3, ogg, flac, m4a). "
             "Not required when --midi-input is used.",
    )
    parser.add_argument(
        "--midi-input",
        type=Path,
        default=None,
        help="Skip Phase 1 and use this MIDI file directly for Phase 2+3.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: outputs/<backend>/<input_stem>.pdf). Extension sets format.",
    )
    parser.add_argument(
        "-m", "--keep-midi",
        type=Path,
        default=None,
        help="Path for intermediate MIDI file (default: <input_stem>.mid beside output)",
    )
    parser.add_argument(
        "-t", "--transcriber",
        choices=TRANSCRIBERS,
        default="basic-pitch",
        help="Audio-to-MIDI transcriber: basic-pitch (Spotify CNN), "
             "mt3 (Google T5 Transformer, requires venv_mt3), "
             "transkun (event-based, continuous timestamps), or "
             "hft (hFT-Transformer, sub-frame precision). Default: basic-pitch.",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Basic Pitch onset detection threshold (0-1). Higher = fewer false onsets. Default: 0.5.",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="Basic Pitch frame detection threshold (0-1). Higher = fewer phantom notes. Default: 0.3.",
    )
    parser.add_argument(
        "-b", "--backend",
        choices=BACKENDS,
        default="music21",
        help="Score conversion backend: music21 (heuristics + LilyPond), "
             "musescore (MuseScore MIDI import), "
             "transformer (MIDI2ScoreTransformer neural model). Default: music21.",
    )
    # MIDI2ScoreTransformer (transformer backend) tuning flags --------------
    parser.add_argument(
        "--pad-threshold",
        type=float,
        default=0.5,
        help="Transformer backend: sigmoid threshold for the pad stream during "
             "detokenization. Lower (~0.3-0.4) rescues notes the model is unsure "
             "about. Default: 0.5 (paper default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Transformer backend: top-k sampling. 1 = greedy (paper default). "
             "Try 5 with --temperature 0.8 for slightly more diverse decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Transformer backend: sampling temperature. 1.0 = no scaling. "
             "Lower = sharper, higher = flatter. Default: 1.0.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Transformer backend: per-chunk decode length. Larger chunks give "
             "the model more long-range context (better time-sig / barlines) "
             "but slower. Default: 512 (paper default). Try 1024.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Transformer backend: overlap between adjacent chunks. Higher = "
             "more cross-chunk consistency. Default: 128. Must be < --chunk-size.",
    )
    # hFT (Phase 1) detection tuning ----------------------------------------
    parser.add_argument(
        "--hft-onset-threshold",
        type=float,
        default=0.5,
        help="hFT only: onset detection threshold (0-1). Lower (~0.3-0.4) "
             "rescues soft/dense notes the 0.5 default misses — the biggest "
             "Phase-1 recall knob. Default: 0.5.",
    )
    parser.add_argument(
        "--hft-mpe-threshold",
        type=float,
        default=0.5,
        help="hFT only: multi-pitch (frame) threshold (0-1). Lower rescues "
             "notes in dense passages. Default: 0.5.",
    )
    parser.add_argument(
        "--hft-offset-threshold",
        type=float,
        default=0.5,
        help="hFT only: offset detection threshold (0-1). Default: 0.5.",
    )
    parser.add_argument(
        "--hft-offset-mode",
        choices=["shorter", "offset", "longer"],
        default="shorter",
        help="hFT only: note-duration estimation. 'shorter' (default) clips "
             "sustained notes; 'offset'/'longer' preserve duration better for "
             "pedaled/sustained repertoire.",
    )
    # Audio preprocessing (Phase 1) -----------------------------------------
    parser.add_argument(
        "--normalize-audio",
        action="store_true",
        help="hFT only: downmix to mono, resample to 16 kHz, peak-normalize "
             "to -3 dBFS before transcription. Helps when input recordings "
             "have very different levels.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="transformer backend: save the MusicXML but skip the MuseScore "
             "render step. Faster for evaluation (the scored artifact is the "
             "MusicXML, not the PDF).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backend = args.backend
    skip_phase1 = args.midi_input is not None

    raw_notes = None

    if skip_phase1:
        midi_path = args.midi_input
        if not midi_path.is_file():
            log.error("MIDI input file not found: %s", midi_path)
            sys.exit(1)
        output_path = args.output or (
            SCRIPT_DIR / "outputs" / backend / (midi_path.stem + ".pdf")
        )
        log.info("Skipping Phase 1 — using existing MIDI: %s", midi_path)
    else:
        if args.input is None:
            log.error("Either provide an audio file or use --midi-input.")
            sys.exit(1)
        audio_path: Path = args.input
        if not audio_path.is_file():
            log.error("Input file not found: %s", audio_path)
            sys.exit(1)
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO:
            log.error(
                "Unsupported audio format '%s'. Supported: %s",
                audio_path.suffix, ", ".join(sorted(SUPPORTED_AUDIO)),
            )
            sys.exit(1)

        output_path = args.output or (
            SCRIPT_DIR / "outputs" / backend / (audio_path.stem + ".pdf")
        )
        midi_path = args.keep_midi or output_path.with_name(audio_path.stem + ".mid")

        # Phase 1: Audio → MIDI
        transcriber = args.transcriber
        raw_notes = None

        if args.normalize_audio and transcriber == "hft":
            audio_path = preprocess_audio(audio_path, target_sr=16000, peak_dbfs=-3.0)

        try:
            n_notes, raw_notes = audio_to_midi(
                audio_path, midi_path, transcriber=transcriber,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                hft_onset_threshold=args.hft_onset_threshold,
                hft_mpe_threshold=args.hft_mpe_threshold,
                hft_offset_threshold=args.hft_offset_threshold,
                hft_offset_mode=args.hft_offset_mode)
        except Exception as exc:
            log.error("Audio → MIDI failed: %s", exc)
            sys.exit(1)

        filter_midi(midi_path)

    if output_path.suffix.lower() not in SUPPORTED_OUTPUT:
        log.error(
            "Unsupported output format '%s'. Supported: %s",
            output_path.suffix, ", ".join(sorted(SUPPORTED_OUTPUT)),
        )
        sys.exit(1)

    print()

    # Phase 2 + 3: MIDI → Score → Render (backend-dependent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Backend: %s", backend)
    print()

    try:
        if backend == "music21":
            backend_music21(midi_path, output_path)
        elif backend == "musescore":
            backend_musescore(midi_path, output_path)
        elif backend == "transformer":
            backend_transformer(
                midi_path, output_path, raw_notes=raw_notes,
                pad_threshold=args.pad_threshold,
                top_k=args.top_k,
                temperature=args.temperature,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                no_render=args.no_render,
            )
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
