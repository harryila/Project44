"""
generate.py — Génère un fichier **kern depuis un checkpoint CWT.

Usage :
    python generate.py --checkpoint <ckpt> --output output.krn
    python generate.py --checkpoint <ckpt> --prompt-krn kern_data/Chopin_nocturne72-1.krn
    python generate.py --checkpoint <ckpt> --key K_b-e-a-_min --time M_3/4 --tempo MM_72

Options :
    --checkpoint      Chemin vers le fichier .pt (obligatoire)
    --output          Fichier de sortie .krn  (défaut : generated.krn)
    --max-tokens      Nombre de tokens à générer (défaut : 1200)
    --temperature     Température du sampling (défaut : 1.0)
    --top-k           Top-k sampling (défaut : 50)
    --prompt-krn      Fichier kern source pour le prompt (6 premières mesures)
    --prompt-measures Nombre de mesures à utiliser comme prompt (défaut : 6)
    --key             Token de tonalité   ex: K_b-e-a-_min  (défaut : K_0_maj)
    --time            Token de métrique   ex: M_3/4          (défaut : M_4/4)
    --tempo           Token de tempo      ex: MM_72           (défaut : MM_120)
    --seed            Graine aléatoire    (défaut : 42)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import (  # noqa: E402
    MetaPrefix, TokenizedScore, Vocabulary, HEADS as HEAD_NAMES,
    MultiHeadTokenizer, CompoundToken,
)
from cwt_model import CompoundWordTransformer, ModelConfig  # noqa: E402
from postprocess import kern_file  # noqa: E402

# ── Helpers ────────────────────────────────────────────────────────────────────

def _barline_seed(vocab: Vocabulary, device: torch.device) -> dict[str, torch.Tensor]:
    """Crée un seed minimal : un token Barline (H1=Barline, H2-H12=NULL)."""
    null_tok = {h: 0 for h in HEAD_NAMES}
    null_tok["h1"] = vocab._to_id["h1"]["Barline"]
    null_tok["h11"] = vocab._to_id["h11"]["Staff_1"]   # v3 : H11 = Staff_1..4 (plus de RH/LH)
    for h in HEAD_NAMES:
        if h not in ("h1", "h11") and "NULL" in vocab._to_id[h]:
            null_tok[h] = vocab._to_id[h]["NULL"]
    return {h: torch.tensor([[null_tok[h]]], dtype=torch.long, device=device)
            for h in HEAD_NAMES}


def _mxl_to_kern(mxl_path: Path) -> Path:
    """MXL -> kern via l'UNIQUE convertisseur exhaustif pdmx_scrapper/mxl_to_kern.py.
    NE PAS reimplementer ici (cf. CLAUDE.md : un seul convertisseur MXL<->kern, lossless)."""
    import tempfile
    repo = Path(__file__).resolve().parents[2]   # .../taff
    for sub in (".", "pdmx_scrapper", "midi2kern"):
        p = str(repo / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import mxl_to_kern as _exhaustive
    dst = Path(tempfile.mkdtemp()) / (Path(mxl_path).stem + ".krn")
    _name, ok, err = _exhaustive.convert_file(Path(mxl_path), dst)
    if not ok:
        raise RuntimeError(f"Conversion MXL->kern echouee ({err})")
    return dst


def _prompt_from_kern(
    path: Path,
    n_measures: int,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[CompoundToken], MetaPrefix]:
    """
    Tokenise les n_measures premières mesures d'un fichier kern ou MusicXML.
    Retourne (tenseurs prompt, tokens pour reconstruction, meta).
    """
    # Conversion MXL → kern si nécessaire
    if path.suffix.lower() in (".xml", ".musicxml", ".mxl"):
        import tempfile, warnings
        warnings.filterwarnings("ignore")
        path = _mxl_to_kern(path)
        print(f"[prompt] MXL converti -> {path}")

    tok   = MultiHeadTokenizer()
    score = tok.tokenize_file(path)

    # Couper après n_measures barlines
    cutoff      = 0
    bar_count   = 0
    for i, t in enumerate(score.tokens):
        if t.h1 == "Barline":
            bar_count += 1
            if bar_count > n_measures:
                cutoff = i
                break
    else:
        cutoff = len(score.tokens)

    prefix_tokens = score.tokens[:cutoff]
    print(f"[prompt] {path.name} — {bar_count} mesures, {len(prefix_tokens)} tokens de prompt")

    # Encoder en tenseurs (batch=1)
    prompt: dict[str, list[int]] = {h: [] for h in HEAD_NAMES}
    for t in prefix_tokens:
        for h in HEAD_NAMES:
            val = getattr(t, h)
            prompt[h].append(vocab._to_id[h].get(val, 0))

    tensors = {h: torch.tensor([ids], dtype=torch.long, device=device)
               for h, ids in prompt.items()}

    return tensors, prefix_tokens, score.meta


def _ids_to_tokens(generated: dict[str, list[int]], vocab: Vocabulary) -> list[CompoundToken]:
    """Convertit les IDs générés en liste de CompoundTokens."""
    n = len(generated["h1"])
    tokens = []
    for i in range(n):
        ids = [generated[h][i] for h in HEAD_NAMES]
        tokens.append(vocab.decode_token(ids))
    return tokens


def _load_model(ckpt_path: Path, device: torch.device) -> CompoundWordTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ckpt and isinstance(ckpt["cfg"], ModelConfig):
        cfg = ckpt["cfg"]
    else:
        cfg = ModelConfig()
        print("[warn] Config absente du checkpoint — utilisation des défauts.")
    model = CompoundWordTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    step  = ckpt.get("step",  "?")
    print(f"[model] Chargé depuis {ckpt_path.name}  (epoch {epoch}, step {step})")
    print(f"[model] {model.count_params():,} paramètres")
    return model


# ── Génération principale ──────────────────────────────────────────────────────

def generate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] device={device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.exists():
        print(f"Checkpoint introuvable : {ckpt_path}")
        sys.exit(1)

    model = _load_model(ckpt_path, device)
    vocab = Vocabulary()

    # ── Prompt ────────────────────────────────────────────────────────────────
    prefix_tokens: list[CompoundToken] = []

    if args.prompt_krn:
        prompt_path = Path(args.prompt_krn).expanduser()
        if not prompt_path.exists():
            print(f"Fichier prompt introuvable : {prompt_path}")
            sys.exit(1)
        prompt, prefix_tokens, src_meta = _prompt_from_kern(
            prompt_path, args.prompt_measures, vocab, device
        )
        # Meta du fichier source (sauf si l'utilisateur force --key/--time/--tempo)
        key   = args.key   if args.key   != "K_0_maj"  else src_meta.key_sig
        time  = args.time  if args.time  != "M_4/4"    else src_meta.time_sig
        tempo = args.tempo if args.tempo != "MM_120"   else src_meta.tempo
        meta  = MetaPrefix(
            key_sig  = key,
            time_sig = time,
            tempo    = tempo,
            clef_rh  = src_meta.clef_rh,
            clef_lh  = src_meta.clef_lh,
            style    = args.style or src_meta.style,
        )
    else:
        prompt = _barline_seed(vocab, device)
        meta   = MetaPrefix(
            key_sig  = args.key,
            time_sig = args.time,
            tempo    = args.tempo,
            clef_rh  = "CLEF_G",
            clef_lh  = "CLEF_F",
            style    = args.style,
        )

    print(f"[gen] max_tokens={args.max_tokens}  temperature={args.temperature}  top_k={args.top_k}")
    print(f"[gen] meta: key={meta.key_sig}  time={meta.time_sig}  tempo={meta.tempo}")
    print("[gen] Génération en cours...")

    generated = model.generate(
        prompt         = prompt,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
    )

    # Décoder + concaténer avec le préfixe
    gen_tokens  = _ids_to_tokens(generated, vocab)
    all_tokens  = prefix_tokens + gen_tokens
    print(f"[gen] {len(gen_tokens)} tokens générés  ({len(prefix_tokens)} tokens de prompt + {len(gen_tokens)} générés = {len(all_tokens)} total)")

    score  = TokenizedScore(meta=meta, tokens=all_tokens)
    output = Path(args.output).expanduser()
    kern_file(score, output)

    from collections import Counter
    h1_counts = Counter(t.h1 for t in gen_tokens)
    print("\n[stats] Distribution H1 (tokens générés) :")
    for typ, cnt in h1_counts.most_common():
        print(f"  {typ:<16} {cnt:5d}  ({cnt*100//len(gen_tokens):2d}%)")

    # H4 (Ornement) — test direct : le modèle génère-t-il des ornements ?
    h4_counts = Counter(t.h4 for t in gen_tokens)
    n_orn = sum(c for v, c in h4_counts.items() if v not in ("NULL", ""))
    print(f"\n[stats] Distribution H4 (Ornement) — {n_orn} ornement(s) sur {len(gen_tokens)} tokens :")
    for typ, cnt in h4_counts.most_common():
        flag = "" if typ in ("NULL", "") else "  <-- ornement"
        print(f"  {typ:<16} {cnt:5d}{flag}")

    # H5 (Articulation) — idem
    h5_counts = Counter(t.h5 for t in gen_tokens)
    n_art = sum(c for v, c in h5_counts.items() if v not in ("NULL", ""))
    print(f"\n[stats] Distribution H5 (Articulation) — {n_art} articulation(s) :")
    for typ, cnt in h5_counts.most_common():
        flag = "" if typ in ("NULL", "") else "  <-- articulation"
        print(f"  {typ:<16} {cnt:5d}{flag}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Génération CWT → kern")

    p.add_argument("--checkpoint",      required=True,     help="Fichier .pt du checkpoint")
    p.add_argument("--output",          default="generated/generated.krn")
    p.add_argument("--max-tokens",      type=int,   default=1200)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--top-k",           type=int,   default=50)
    p.add_argument("--prompt-krn",      default="",        help="Fichier kern pour le prompt")
    p.add_argument("--prompt-measures", type=int,   default=6, help="Nb de mesures de prompt")
    p.add_argument("--key",             default="K_0_maj", help="ex: K_b-e-a-_min")
    p.add_argument("--time",            default="M_4/4",   help="ex: M_3/4")
    p.add_argument("--tempo",           default="MM_120",  help="ex: MM_72")
    p.add_argument("--style",           default="",        help="ST_chopin | ST_liszt | ''")
    p.add_argument("--seed",            type=int,   default=42)

    generate(p.parse_args())


if __name__ == "__main__":
    main()
