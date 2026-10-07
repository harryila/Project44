"""Generate a piano harmonization from a melody prompt and convert it to MXL."""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import torch

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tokenizers"))

from cwt_model import CompoundWordTransformer, ModelConfig  # noqa: E402
from extract_melody import extract_top_spine  # noqa: E402
from generate import _ids_to_tokens, _mxl_to_kern  # noqa: E402
from multihead_tokenizer import (  # noqa: E402
    HEADS as HEAD_NAMES,
    MetaPrefix,
    MultiHeadTokenizer,
    TokenizedScore,
    Vocabulary,
    make_sep,
)
from postprocess import kern_file  # noqa: E402


def _load_model(ckpt_path: Path, device: torch.device) -> CompoundWordTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), ModelConfig) else ModelConfig()
    model = CompoundWordTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] {ckpt_path} epoch={ckpt.get('epoch', '?')} step={ckpt.get('step', '?')}")
    print(f"[model] {model.count_params():,} params")
    return model


def _cut_after_measures(tokens, n_measures: int):
    cutoff = len(tokens)
    bar_count = 0
    for i, tok in enumerate(tokens):
        if tok.h1 == "Barline":
            bar_count += 1
            if bar_count > n_measures:
                cutoff = i
                break
    return tokens[:cutoff], bar_count


def _melody_prompt(path: Path, n_measures: int, vocab: Vocabulary, device: torch.device):
    if path.suffix.lower() in (".xml", ".musicxml", ".mxl"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = _mxl_to_kern(path)
        print(f"[prompt] MXL -> {path}")

    kern_text = path.read_text(encoding="utf-8")
    melody_kern = extract_top_spine(kern_text)
    if not melody_kern.strip():
        raise RuntimeError("melody extraction produced an empty prompt")

    with tempfile.NamedTemporaryFile("w", suffix=".krn", delete=False, encoding="utf-8") as tmp:
        tmp.write(melody_kern)
        melody_path = Path(tmp.name)

    tok = MultiHeadTokenizer()
    melody_score = tok.tokenize_file(melody_path)
    melody_tokens, seen_bars = _cut_after_measures(melody_score.tokens, n_measures)
    prompt_tokens = melody_tokens + [make_sep()]

    ids = [vocab.encode_token(t) for t in prompt_tokens]
    prompt = {
        h: torch.tensor([[row[i] for row in ids]], dtype=torch.long, device=device)
        for i, h in enumerate(HEAD_NAMES)
    }

    meta = MetaPrefix(
        key_sig=melody_score.meta.key_sig,
        time_sig=melody_score.meta.time_sig,
        tempo=melody_score.meta.tempo,
        clef_rh="CLEF_G",
        clef_lh="CLEF_F",
        style=melody_score.meta.style,
    )
    print(f"[prompt] melody tokens={len(melody_tokens)} + SEP, bars_seen={seen_bars}")
    print(f"[prompt] meta key={meta.key_sig} time={meta.time_sig} tempo={meta.tempo}")
    return prompt, meta, melody_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CWT harmonization from melody prompt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt-krn", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt-measures", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    print(f"[config] device={device} max_tokens={args.max_tokens} temp={args.temperature} top_k={args.top_k}")

    ckpt_path = Path(args.checkpoint).expanduser()
    prompt_path = Path(args.prompt_krn).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(ckpt_path, device)
    vocab = Vocabulary()
    prompt, meta, melody_path = _melody_prompt(prompt_path, args.prompt_measures, vocab, device)

    generated = model.generate(
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    gen_tokens = _ids_to_tokens(generated, vocab)
    print(f"[gen] generated tokens={len(gen_tokens)}")

    score = TokenizedScore(meta=meta, tokens=gen_tokens)
    kern_file(score, output)
    print(f"[debug] melody prompt kern: {melody_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from music21 import converter

        mxl_path = output.with_suffix(".mxl")
        parsed = converter.parse(str(output))
        parsed.write("musicxml", str(mxl_path))
        print(f"[mxl] OK {mxl_path} notes={len(list(parsed.flatten().notes))}")


if __name__ == "__main__":
    main()
