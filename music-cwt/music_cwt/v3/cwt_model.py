"""
Compound Word Transformer — architecture multi-head pour piano kern.
Spec : archi.md  |  Tokenisation : tokenizers/multihead_tokenizer.py
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulaires (tokenization_rules_reviewed.md) ─────────────────────────────
# Dérivés DIRECTEMENT du tokeniseur : VOCAB_SIZES, NULL_IDS et les index de
# contrainte H1 étaient autrefois hardcodés et se sont désynchronisés au fil des
# ajouts de heads (H13 Pedal, H14 Tempo, clefs mid-pièce, Fioritura_q…). On les
# calcule maintenant à partir de ALL_HEAD_VOCABS pour qu'ils ne dérivent plus.

sys.path.insert(0, str(Path(__file__).parent / "tokenizers"))
from multihead_tokenizer import ALL_HEAD_VOCABS, HEADS, H1_VOCAB, H11_VOCAB  # noqa: E402

VOCAB_SIZES: dict[str, int] = {h: len(ALL_HEAD_VOCABS[h]) for h in HEADS}

HEAD_NAMES = list(HEADS)

# Stage 1 = heads structuraux (conditionnent tous les autres).
# h17 (voix) en STAGE1 : c'est l'identite structurelle du token (comme h11 portee).
# Necessaire pour que l'ancre Staff_1/V1 soit predite meme sur un token Sustain
# (ou tous les autres heads sont NULL).
STAGE1 = ["h1", "h11", "h17"]
# Stage 3 = heads fin-grain conditionnes sur la duree+pitch (et donc Stage 1+2 deja choisis)
# Motivation : ornement et articulation dependent fortement de la duree et de la hauteur
# (un trille sur une noire pointee ; un staccato sur une croche aigue, etc.)
STAGE3 = ["h4", "h5"]
# Stage 2 = tout le reste, conditionne sur Stage 1
STAGE2 = [h for h in HEAD_NAMES if h not in STAGE1 and h not in STAGE3]

# Heads de Stage 2 utilises comme conditionnement supplementaire pour Stage 3 :
# H2 (Duration) et H3 (Pitch) — choix justifie dans archi_model.md.
STAGE3_COND = ["h2", "h3"]

# Index NULL par head, ou -1 si le head n'a pas de NULL (h1, h3, h11).
# Utilisé pour appliquer les contraintes structurelles dans la loss.
NULL_IDS: dict[str, int] = {
    h: (ALL_HEAD_VOCABS[h].index("NULL") if "NULL" in ALL_HEAD_VOCABS[h] else -1)
    for h in HEAD_NAMES
}

# Contraintes structurelles : si condition, forcer ces heads à NULL dans la loss.
H1_BARLINE_ID      = H1_VOCAB.index("Barline")
H1_SEP_ID          = H1_VOCAB.index("SEP")
H1_SUSTAIN_ID      = H1_VOCAB.index("Sustain")   # v3 Phase 2 — ancre "tient", heads NULL
H1_REST_ID         = H1_VOCAB.index("Rest")
H1_FIORITURA_Q_ID  = H1_VOCAB.index("Fioritura_Q")
H1_FIORITURA_q_ID  = H1_VOCAB.index("Fioritura_q")
H1_CHORD_ATOMIC_ID = H1_VOCAB.index("Chord_Atomic")
# H12/H13/H14 portés par tous les tokens (moment musical commun) — pas de mask Staff_1/V1.
H11_STAFF1_ID      = H11_VOCAB.index("Staff_1")  # garde pour postprocess
H17_V1_ID          = ALL_HEAD_VOCABS["h17"].index("V1")  # garde pour postprocess


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # v2 — config ~16M params (vs ~10M en v1) :
    # d_model 256 -> 320  (head_dim = 40), n_layers 9 -> 10, d_ff 1536 -> 1600
    d_model:      int   = 320
    n_layers:     int   = 10
    n_attn_heads: int   = 8     # head_dim = d_model / n_attn_heads = 40
    d_ff:         int   = 1600
    max_seq_len:  int   = 2048
    dropout:      float = 0.1


# ── Compound Embedding (concat + projection) ──────────────────────────────────

class CompoundEmbedding(nn.Module):
    """
    Concatenation des embeddings des N heads, puis projection lineaire vers d_model.
    x_t = Linear([embed_1(tok_1) || embed_2(tok_2) || ... || embed_N(tok_N)])

    Remplace la somme de v1 : plus expressif, le modele apprend a ponderer chaque
    head au lieu de tout mixer additivement. Cout : (N * d_model) -> d_model
    parametres pour la projection.

    Pas de positional embedding ici : RoPE est applique dans l'attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeds = nn.ModuleDict(
            {h: nn.Embedding(VOCAB_SIZES[h], cfg.d_model) for h in HEAD_NAMES}
        )
        self.proj = nn.Linear(len(HEAD_NAMES) * cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        # tokens : dict head -> (B, T) LongTensor
        parts = [self.embeds[h](tokens[h]) for h in HEAD_NAMES]  # liste de (B, T, d)
        cat = torch.cat(parts, dim=-1)                            # (B, T, N*d)
        return self.drop(self.proj(cat))                          # (B, T, d)


# ── RoPE (Rotary Positional Embedding) ────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).
    Encode la position dans les vecteurs Q/K de l'attention en appliquant
    une rotation 2D par paire de dimensions. Pas de parametres appris.

    Avantages vs positional embedding appris :
      - Meilleure generalisation aux sequences plus longues que celles vues a l'entrainement
      - Encode la position relative implicitement via la rotation
      - Pas de parametres a apprendre
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        # Frequences inverses : 1 / base^(2i/d) pour i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, d/2)
        # Format duplique pour appliquer la rotation : (T, d) avec chaque freq deux fois
        cos = freqs.cos().repeat_interleave(2, dim=-1)     # (T, d)
        sin = freqs.sin().repeat_interleave(2, dim=-1)     # (T, d)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Permute les paires (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # Reconstruct interleaved : (-x2, x1, -x4, x3, ...)
        out = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k : (B, n_heads, T, head_dim)
        T = q.size(-2)
        if T > self._cached_len:
            self._build_cache(T)
            # Re-deplacer les buffers sur le bon device
            self.cos_cache = self.cos_cache.to(q.device)
            self.sin_cache = self.sin_cache.to(q.device)
        cos = self.cos_cache[:T].to(q.dtype)  # (T, d)
        sin = self.sin_cache[:T].to(q.dtype)
        # Broadcast : (1, 1, T, d) sur (B, n_heads, T, d)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ── Transformer blocks ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        assert cfg.d_model % cfg.n_attn_heads == 0
        self.n_heads  = cfg.n_attn_heads
        self.head_dim = cfg.d_model // cfg.n_attn_heads
        self.scale    = math.sqrt(self.head_dim)
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = rope  # partage entre tous les layers (pas de parametres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE : encode la position dans Q et K avant le produit attention
        q, k = self.rope(q, k)

        # Masque causal calculé à la volée — évite de stocker 9 × 2048² en VRAM
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, rope)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── Couche de sortie 3 étapes ─────────────────────────────────────────────────

class ThreeStageOutputLayer(nn.Module):
    """
    Stage 1 : predit H1 (Type) + H11 (Spine) depuis h_t.
    Stage 2 : predit le reste (H2, H3, H6-H10, H12-H14) depuis h_t + embed(stage1).
    Stage 3 : predit H4 (Ornement) + H5 (Articulation) depuis cond_stage2 + embed(H2, H3).

    Motivation Stage 3 : ornements et articulations dependent fortement de la duree
    (H2) et de la hauteur (H3). Un trille a plus de sens sur une noire pointee que
    sur une triple-croche ; un staccato a plus de sens sur une croche aigue que
    sur une ronde grave. En les conditionnant explicitement sur H2/H3, on capture
    cette dependance fine au lieu de la laisser au seul backbone d'attention.

    En training  : teacher forcing — on passe gt_stage1 et gt_stage3_cond.
    En inférence : argmax des stages précédents utilisé comme conditionnement.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Stage 1
        self.stage1 = nn.ModuleDict(
            {h: nn.Linear(cfg.d_model, VOCAB_SIZES[h], bias=False) for h in STAGE1}
        )
        # Embeddings de conditionnement Stage 1 -> Stage 2 (distincts des inputs)
        self.cond_embed_s1 = nn.ModuleDict(
            {h: nn.Embedding(VOCAB_SIZES[h], cfg.d_model) for h in STAGE1}
        )
        self.ln_cond_s2 = nn.LayerNorm(cfg.d_model)
        # Stage 2
        self.stage2 = nn.ModuleDict(
            {h: nn.Linear(cfg.d_model, VOCAB_SIZES[h], bias=False) for h in STAGE2}
        )
        # Embeddings de conditionnement Stage 2 -> Stage 3 (uniquement les heads STAGE3_COND)
        self.cond_embed_s2 = nn.ModuleDict(
            {h: nn.Embedding(VOCAB_SIZES[h], cfg.d_model) for h in STAGE3_COND}
        )
        self.ln_cond_s3 = nn.LayerNorm(cfg.d_model)
        # Stage 3
        self.stage3 = nn.ModuleDict(
            {h: nn.Linear(cfg.d_model, VOCAB_SIZES[h], bias=False) for h in STAGE3}
        )

    def forward(
        self,
        h: torch.Tensor,                                              # (B, T, d_model)
        gt_stage1: Optional[dict[str, torch.Tensor]] = None,          # (B, T) par head
        gt_stage3_cond: Optional[dict[str, torch.Tensor]] = None,     # (B, T) par head STAGE3_COND
    ) -> dict[str, torch.Tensor]:
        logits: dict[str, torch.Tensor] = {}

        # ── Stage 1 — independant ────────────────────────────────────────────
        for head in STAGE1:
            logits[head] = self.stage1[head](h)  # (B, T, vocab)

        # ── Conditionnement Stage 1 -> Stage 2 ───────────────────────────────
        cond_s2 = h
        if gt_stage1 is not None:
            for head in STAGE1:
                cond_s2 = cond_s2 + self.cond_embed_s1[head](gt_stage1[head])
        else:
            for head in STAGE1:
                pred = logits[head].argmax(dim=-1)  # (B, T)
                cond_s2 = cond_s2 + self.cond_embed_s1[head](pred)
        cond_s2 = self.ln_cond_s2(cond_s2)

        # ── Stage 2 — conditionne sur Stage 1 ────────────────────────────────
        for head in STAGE2:
            logits[head] = self.stage2[head](cond_s2)

        # ── Conditionnement Stage 2 -> Stage 3 (h2 + h3) ─────────────────────
        cond_s3 = cond_s2
        if gt_stage3_cond is not None:
            for head in STAGE3_COND:
                cond_s3 = cond_s3 + self.cond_embed_s2[head](gt_stage3_cond[head])
        else:
            for head in STAGE3_COND:
                pred = logits[head].argmax(dim=-1)
                cond_s3 = cond_s3 + self.cond_embed_s2[head](pred)
        cond_s3 = self.ln_cond_s3(cond_s3)

        # ── Stage 3 — conditionne sur Stage 1+2 ──────────────────────────────
        for head in STAGE3:
            logits[head] = self.stage3[head](cond_s3)

        return logits


# ── Modèle principal ──────────────────────────────────────────────────────────

class CompoundWordTransformer(nn.Module):

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg    = cfg
        self.embed  = CompoundEmbedding(cfg)
        # RoPE partage entre toutes les couches (pas de parametres appris)
        head_dim = cfg.d_model // cfg.n_attn_heads
        self.rope   = RotaryEmbedding(head_dim, max_seq_len=cfg.max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)]
        )
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.output = ThreeStageOutputLayer(cfg)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        gt_stage1: Optional[dict[str, torch.Tensor]] = None,
        gt_stage3_cond: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.output(x, gt_stage1, gt_stage3_cond)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        tokens: dict[str, torch.Tensor],              # (B, T)
        pad_lengths: Optional[torch.Tensor] = None,   # (B,) longueurs réelles
    ) -> torch.Tensor:
        """
        Loss autoregressive : prédit tokens[1:] depuis tokens[:-1].
        - Contraintes structurelles : heads forcées NULL ne contribuent pas.
        - pad_lengths : positions de padding en fin de séquence sont exclues.
        """
        inp = {h: v[:, :-1] for h, v in tokens.items()}   # (B, T-1)
        tgt = {h: v[:, 1:]  for h, v in tokens.items()}   # (B, T-1)

        gt_s1       = {h: tgt[h] for h in STAGE1}
        gt_s3_cond  = {h: tgt[h] for h in STAGE3_COND}
        logits = self.forward(inp, gt_stage1=gt_s1, gt_stage3_cond=gt_s3_cond)

        B, T = tgt["h1"].shape

        # Masque de padding : position t valide ssi t < pad_lengths[b] - 1
        if pad_lengths is not None:
            pos       = torch.arange(T, device=pad_lengths.device)        # (T,)
            pad_mask  = pos.unsqueeze(0) < (pad_lengths.unsqueeze(1) - 1) # (B, T)
        else:
            pad_mask = torch.ones(B, T, dtype=torch.bool, device=tgt["h1"].device)

        loss = torch.tensor(0.0, device=tgt["h1"].device)

        for h in HEAD_NAMES:
            null_id = NULL_IDS[h]
            tgt_h   = tgt[h]
            log_h   = logits[h]

            mask = pad_mask.clone()

            if null_id >= 0:
                h1  = tgt["h1"]
                h11 = tgt["h11"]

                if h not in STAGE1:
                    mask = mask & ~((h1 == H1_BARLINE_ID) | (h1 == H1_SEP_ID)
                                    | (h1 == H1_SUSTAIN_ID))

                if h in ("h4", "h5", "h8"):
                    mask = mask & ~(h1 == H1_REST_ID)

                if h in ("h4", "h8"):
                    mask = mask & ~((h1 == H1_FIORITURA_Q_ID)
                                    | (h1 == H1_FIORITURA_q_ID))

                # v3 — Chord_Atomic : H6/H7/H8 forces NULL (tie/slur/phrase
                # per-note interdits sur un accord atomise). H4/H5 (ornement/
                # articulation) PEUVENT etre portes = decoration commune a tout
                # l'accord (relaxation 2026-06) -> appris normalement.
                if h in ("h6", "h7", "h8"):
                    mask = mask & ~(h1 == H1_CHORD_ATOMIC_ID)

                # v3 — H15 (Chord Quality) et H16 (Chord Voicing) ne sont actifs
                # QUE quand H1 = Chord_Atomic. Hors de ce cas, la valeur est
                # toujours NULL et on n'apprend rien dessus.
                if h in ("h15", "h16"):
                    mask = mask & (h1 == H1_CHORD_ATOMIC_ID)


            if mask.any():
                loss = loss + F.cross_entropy(log_h[mask], tgt_h[mask])

        return loss / len(HEAD_NAMES)

    # ── Génération ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: dict[str, torch.Tensor],   # (1, T_prompt) par head
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> dict[str, list[int]]:
        """Génération autoregressive token par token."""
        self.eval()
        ctx = {h: v.clone() for h, v in prompt.items()}
        generated: dict[str, list[int]] = {h: [] for h in HEAD_NAMES}

        for _ in range(max_new_tokens):
            # Tronquer au max_seq_len si nécessaire
            trunc = {h: v[:, -self.cfg.max_seq_len:] for h, v in ctx.items()}

            # Inférence (gt_stage1=None → stage 2 conditionné par argmax stage 1)
            logits = self.forward(trunc, gt_stage1=None)

            new_tok: dict[str, torch.Tensor] = {}
            for h in HEAD_NAMES:
                l = logits[h][:, -1, :] / max(temperature, 1e-8)   # (1, vocab)
                if top_k > 0:
                    topk_vals, _ = torch.topk(l, min(top_k, l.size(-1)))
                    l = l.masked_fill(l < topk_vals[:, -1:], float("-inf"))
                probs = F.softmax(l, dim=-1)
                new_tok[h] = torch.multinomial(probs, num_samples=1)  # (1, 1)

            for h in HEAD_NAMES:
                ctx[h] = torch.cat([ctx[h], new_tok[h]], dim=1)
                generated[h].append(new_tok[h].item())

        return generated

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Test rapide ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = CompoundWordTransformer(cfg)
    n     = model.count_params()
    print(f"Paramètres : {n:,}  (~{n / 1e6:.1f} M)")
    print(f"Config     : d_model={cfg.d_model}, layers={cfg.n_layers}, "
          f"heads={cfg.n_attn_heads}, d_ff={cfg.d_ff}, seq_len={cfg.max_seq_len}")

    B, T = 2, 64
    tokens = {h: torch.randint(0, VOCAB_SIZES[h], (B, T)) for h in HEAD_NAMES}

    # Forward pass + loss
    loss = model.compute_loss(tokens)
    print(f"\nLoss sur batch aléatoire : {loss.item():.4f}")

    # Shape des logits
    inp    = {h: v[:, :-1] for h, v in tokens.items()}
    gt_s1  = {h: tokens[h][:, 1:] for h in STAGE1}
    logits = model(inp, gt_stage1=gt_s1)
    print("\nLogits shapes :")
    for h, l in logits.items():
        print(f"  {h:4s} : {tuple(l.shape)}  (vocab {VOCAB_SIZES[h]})")
