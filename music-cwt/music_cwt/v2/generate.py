"""
generate.py â€” GÃ©nÃ¨re un fichier **kern depuis un checkpoint CWT.

Usage :
    python generate.py --checkpoint <ckpt> --output output.krn
    python generate.py --checkpoint <ckpt> --prompt-krn kern_data/Chopin_nocturne72-1.krn
    python generate.py --checkpoint <ckpt> --key K_b-e-a-_min --time M_3/4 --tempo MM_72

Options :
    --checkpoint      Chemin vers le fichier .pt (obligatoire)
    --output          Fichier de sortie .krn  (dÃ©faut : generated.krn)
    --max-tokens      Nombre de tokens Ã  gÃ©nÃ©rer (dÃ©faut : 1200)
    --temperature     TempÃ©rature du sampling (dÃ©faut : 1.0)
    --top-k           Top-k sampling (dÃ©faut : 50)
    --prompt-krn      Fichier kern source pour le prompt (6 premiÃ¨res mesures)
    --prompt-measures Nombre de mesures Ã  utiliser comme prompt (dÃ©faut : 6)
    --key             Token de tonalitÃ©   ex: K_b-e-a-_min  (dÃ©faut : K_0_maj)
    --time            Token de mÃ©trique   ex: M_3/4          (dÃ©faut : M_4/4)
    --tempo           Token de tempo      ex: MM_72           (dÃ©faut : MM_120)
    --seed            Graine alÃ©atoire    (dÃ©faut : 42)
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

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _barline_seed(vocab: Vocabulary, device: torch.device) -> dict[str, torch.Tensor]:
    """CrÃ©e un seed minimal : un token Barline (H1=Barline, H2-H12=NULL)."""
    null_tok = {h: 0 for h in HEAD_NAMES}
    null_tok["h1"] = vocab._to_id["h1"]["Barline"]
    null_tok["h11"] = vocab._to_id["h11"]["RH"]
    for h in HEAD_NAMES:
        if h not in ("h1", "h11") and "NULL" in vocab._to_id[h]:
            null_tok[h] = vocab._to_id[h]["NULL"]
    return {h: torch.tensor([[null_tok[h]]], dtype=torch.long, device=device)
            for h in HEAD_NAMES}


def _mxl_to_kern(mxl_path: Path) -> Path:
    """Convertit un fichier MusicXML en kern 2-spines via music21.

    Pas de squeeze/merge multi-staff ici : le prompt doit deja etre ramene a
    deux parties propres en amont. On prend parts[0] et parts[1], avec un swap
    simple si la premiere partie est en clef de basse.
    """
    import tempfile
    from collections import defaultdict
    from music21 import converter as m21conv, clef as m21clef, key as m21key
    from music21 import meter as m21meter, note as m21note, chord as m21chord
    from music21 import tempo as m21tempo

    def _ql_to_kern_dur(ql: float) -> str:
        mapping = {4.0:"1", 3.0:"2.", 2.0:"2", 1.5:"4.", 1.0:"4",
                   0.75:"8.", 0.5:"8", 0.375:"16.", 0.25:"16",
                   0.1875:"32.", 0.125:"32", 0.0625:"64"}
        ql = round(ql, 6)
        if ql in mapping:
            return mapping[ql]
        return mapping[min(mapping.keys(), key=lambda x: abs(x - ql))]

    def _pitch_to_kern(p) -> str:
        name  = p.step
        alter = p.accidental.modifier if p.accidental else ""
        alter = alter.replace("sharp", "#").replace("flat", "-").replace("natural", "")
        octave = p.octave
        if octave >= 4:
            base = name.lower() * (octave - 3)
        else:
            base = name.upper() * (4 - octave)
        return base + alter

    def _element_to_kern(el) -> str:
        dur = _ql_to_kern_dur(el.duration.quarterLength)
        prefix, suffix = "", ""
        if hasattr(el, "tie") and el.tie:
            if el.tie.type == "start":
                prefix = "["
            elif el.tie.type == "stop":
                suffix = "]"
            elif el.tie.type == "continue":
                prefix = "_"
        if isinstance(el, m21note.Rest):
            return dur + "r"
        if isinstance(el, m21note.Note):
            return prefix + dur + _pitch_to_kern(el.pitch) + suffix
        if isinstance(el, m21chord.Chord):
            return " ".join(prefix + dur + _pitch_to_kern(p) + suffix for p in el.pitches)
        return None

    def _first_clef(part):
        clefs = part.flatten().getElementsByClass(m21clef.Clef)
        return clefs[0] if clefs else None

    def _clef_token(part):
        c = _first_clef(part)
        if c is None:
            return "*clefG2"
        if isinstance(c, (m21clef.BassClef, m21clef.Bass8vaClef, m21clef.Bass8vbClef)):
            return "*clefF4"
        return "*clefG2"

    s = m21conv.parse(str(mxl_path))
    parts = list(s.parts)
    if len(parts) < 2:
        raise ValueError(f"Score doit avoir 2 parties (RH+LH), trouve {len(parts)}")

    rh_part, lh_part = parts[0], parts[1]
    c0 = _first_clef(rh_part)
    if isinstance(c0, (m21clef.BassClef, m21clef.Bass8vaClef, m21clef.Bass8vbClef)):
        rh_part, lh_part = lh_part, rh_part

    rh_meas = list(rh_part.getElementsByClass("Measure"))
    lh_meas = list(lh_part.getElementsByClass("Measure"))
    n_measures = min(len(rh_meas), len(lh_meas))

    clef_rh = _clef_token(rh_part)
    clef_lh = _clef_token(lh_part)
    key_str = "*k[]"
    ts_str = "*M4/4"
    tempo_str = "*MM120"

    ks = s.flatten().getElementsByClass(m21key.KeySignature)
    if ks:
        sharps = ks[0].sharps
        if sharps > 0:
            acc = ["f#", "c#", "g#", "d#", "a#", "e#", "b#"][:sharps]
        elif sharps < 0:
            acc = ["b-", "e-", "a-", "d-", "g-", "c-", "f-"][:-sharps]
        else:
            acc = []
        key_str = f"*k[{''.join(acc)}]" if acc else "*k[]"

    ts = s.flatten().getElementsByClass(m21meter.TimeSignature)
    if ts:
        ts_str = f"*M{ts[0].ratioString}"

    mm = s.flatten().getElementsByClass(m21tempo.MetronomeMark)
    if mm and mm[0].number:
        tempo_str = f"*MM{int(mm[0].number)}"

    lines = [
        "**kern\t**kern",
        "*staff1\t*staff2",
        f"{clef_rh}\t{clef_lh}",
        f"{key_str}\t{key_str}",
        f"{ts_str}\t{ts_str}",
        f"{tempo_str}\t{tempo_str}",
    ]

    for i in range(n_measures):
        lines.append(f"={i + 1}\t={i + 1}")
        events = defaultdict(lambda: [None, None])

        for el in rh_meas[i].flatten().notesAndRests:
            offset = round(float(el.offset), 6)
            tok = _element_to_kern(el)
            if tok is not None:
                events[offset][0] = tok

        for el in lh_meas[i].flatten().notesAndRests:
            offset = round(float(el.offset), 6)
            tok = _element_to_kern(el)
            if tok is not None:
                events[offset][1] = tok

        for offset in sorted(events):
            rh_s, lh_s = events[offset]
            lines.append(f"{rh_s or '.'}\t{lh_s or '.'}")

    lines += ["==", "*-\t*-"]

    tmp = tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w", encoding="utf-8")
    tmp.write("\n".join(lines))
    tmp.close()
    return Path(tmp.name)

def _prompt_from_kern(
    path: Path,
    n_measures: int,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[CompoundToken], MetaPrefix]:
    """
    Tokenise les n_measures premiÃ¨res mesures d'un fichier kern ou MusicXML.
    Retourne (tenseurs prompt, tokens pour reconstruction, meta).
    """
    # Conversion MXL â†’ kern si nÃ©cessaire
    if path.suffix.lower() in (".xml", ".musicxml", ".mxl"):
        import tempfile, warnings
        warnings.filterwarnings("ignore")
        path = _mxl_to_kern(path)
        print(f"[prompt] MXL converti -> {path}")

    tok   = MultiHeadTokenizer()
    score = tok.tokenize_file(path)

    # Couper aprÃ¨s n_measures barlines
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
    print(f"[prompt] {path.name} â€” {bar_count} mesures, {len(prefix_tokens)} tokens de prompt")

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
    """Convertit les IDs gÃ©nÃ©rÃ©s en liste de CompoundTokens."""
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
        print("[warn] Config absente du checkpoint â€” utilisation des dÃ©fauts.")
    model = CompoundWordTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    step  = ckpt.get("step",  "?")
    print(f"[model] ChargÃ© depuis {ckpt_path.name}  (epoch {epoch}, step {step})")
    print(f"[model] {model.count_params():,} paramÃ¨tres")
    return model


# â”€â”€ GÃ©nÃ©ration principale â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    print("[gen] GÃ©nÃ©ration en cours...")

    generated = model.generate(
        prompt         = prompt,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = args.top_k,
    )

    # DÃ©coder + concatÃ©ner avec le prÃ©fixe
    gen_tokens  = _ids_to_tokens(generated, vocab)
    all_tokens  = prefix_tokens + gen_tokens
    print(f"[gen] {len(gen_tokens)} tokens gÃ©nÃ©rÃ©s  ({len(prefix_tokens)} tokens de prompt + {len(gen_tokens)} gÃ©nÃ©rÃ©s = {len(all_tokens)} total)")

    score  = TokenizedScore(meta=meta, tokens=all_tokens)
    output = Path(args.output).expanduser()
    kern_file(score, output)

    from collections import Counter
    h1_counts = Counter(t.h1 for t in gen_tokens)
    print("\n[stats] Distribution H1 (tokens gÃ©nÃ©rÃ©s) :")
    for typ, cnt in h1_counts.most_common():
        print(f"  {typ:<16} {cnt:5d}  ({cnt*100//len(gen_tokens):2d}%)")

    # H4 (Ornement) â€” test direct : le modÃ¨le gÃ©nÃ¨re-t-il des ornements ?
    h4_counts = Counter(t.h4 for t in gen_tokens)
    n_orn = sum(c for v, c in h4_counts.items() if v not in ("NULL", ""))
    print(f"\n[stats] Distribution H4 (Ornement) â€” {n_orn} ornement(s) sur {len(gen_tokens)} tokens :")
    for typ, cnt in h4_counts.most_common():
        flag = "" if typ in ("NULL", "") else "  <-- ornement"
        print(f"  {typ:<16} {cnt:5d}{flag}")

    # H5 (Articulation) â€” idem
    h5_counts = Counter(t.h5 for t in gen_tokens)
    n_art = sum(c for v, c in h5_counts.items() if v not in ("NULL", ""))
    print(f"\n[stats] Distribution H5 (Articulation) â€” {n_art} articulation(s) :")
    for typ, cnt in h5_counts.most_common():
        flag = "" if typ in ("NULL", "") else "  <-- articulation"
        print(f"  {typ:<16} {cnt:5d}{flag}")


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    p = argparse.ArgumentParser(description="GÃ©nÃ©ration CWT â†’ kern")

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

