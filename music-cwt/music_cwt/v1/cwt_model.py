"""
Compound Word Transformer — architecture multi-head pour piano kern.
Spec : archi.md  |  Tokenisation : tokenizers/multihead_tokenizer.py
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulaires (tokenization_rules_reviewed.md) ─────────────────────────────

VOCAB_SIZES: dict[str, int] = {
    "h1":  8,    # Note_Single/Chord_Start/Chord_Cont/Chord_End/Fioritura_Q/Rest/Barline/SEP
    "h2":  24,   # durées
    "h3":  282,  # hauteurs (pitch fused)
    "h4":  14,   # ornements
    "h5":  6,    # articulations
    "h6":  4,    # liaison (tie)
    "h7":  3,    # slur
    "h8":  5,    # phrase
    "h9":  3,    # voicing (stem)
    "h10": 9,    # beam
    "h11": 2,    # spine (RH / LH)
    "h12": 19,   # dynamique
}

HEAD_NAMES = list(VOCAB_SIZES.keys())

# Stage 1 = heads structuraux (conditionnent les autres)
STAGE1 = ["h1", "h11"]
STAGE2 = [h for h in HEAD_NAMES if h not in STAGE1]

# Index NULL par head (dernier token de chaque vocab dans le tokenizer)
# Utilisé pour appliquer les contraintes structurelles dans la loss.
# Ces index correspondent aux IDs définis dans Vocabulary (multihead_tokenizer.py).
# Valeur -1 = head sans NULL (h1 et h11 n'ont pas de NULL).
NULL_IDS: dict[str, int] = {
    "h1":  -1,   # pas de NULL
    "h2":  23,
    "h3":  -1,   # pas de NULL (r/rr sont des pitches valides)
    "h4":  13,
    "h5":  5,
    "h6":  3,
    "h7":  2,
    "h8":  4,
    "h9":  2,
    "h10": 8,
    "h11": -1,   # pas de NULL
    "h12": 18,
}

# Contraintes structurelles : si condition, forcer ces heads à NULL dans la loss
# Format : liste de (head_conditionnante, valeur_ID, [heads_à_NULLer])
# Les IDs H1 correspondent à l'ordre dans Vocabulary.H1_VOCAB.
H1_BARLINE_ID = 6   # index de "Barline" dans H1
H1_SEP_ID     = 7   # index de "SEP"
H1_REST_ID    = 5   # index de "Rest"
H1_FIORITURA_ID = 4 # index de "Fioritura_Q"
H11_LH_ID     = 1   # index de "LH" dans H11


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    d_model:      int   = 256
    n_layers:     int   = 9
    n_attn_heads: int   = 8     # head_dim = d_model / n_attn_heads = 32
    d_ff:         int   = 1536
    max_seq_len:  int   = 2048
    dropout:      float = 0.1


# ── Compound Embedding ────────────────────────────────────────────────────────

class CompoundEmbedding(nn.Module):
    """
    Somme des embeddings des 12 heads + positional embedding appris.
    x_t = pos_embed(t) + Σ_h embed_h(token_h_t)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeds = nn.ModuleDict(
            {h: nn.Embedding(VOCAB_SIZES[h], cfg.d_model) for h in HEAD_NAMES}
        )
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        # tokens : dict head -> (B, T) LongTensor
        vals = list(tokens.values())
        B, T = vals[0].shape
        device = vals[0].device

        pos = torch.arange(T, device=device)
        x = self.pos_embed(pos).unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)
        for h, emb in self.embeds.items():
            x = x + emb(tokens[h])
        return self.drop(x)


# ── Transformer blocks ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_attn_heads == 0
        self.n_heads  = cfg.n_attn_heads
        self.head_dim = cfg.d_model // cfg.n_attn_heads
        self.scale    = math.sqrt(self.head_dim)
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Masque causal calculé à la volée — évite de stocker 9 × 2048² en VRAM
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
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


# ── Couche de sortie 2 étapes ─────────────────────────────────────────────────

class TwoStageOutputLayer(nn.Module):
    """
    Stage 1 : prédit H1 (Type) et H11 (Spine) depuis h_t.
    Stage 2 : prédit les 10 autres heads depuis h_t + embed(H1) + embed(H11).

    En training  : teacher forcing — on passe gt_stage1 (vrais labels décalés).
    En inférence : gt_stage1=None → argmax de stage 1 utilisé comme conditionnement.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.stage1 = nn.ModuleDict(
            {h: nn.Linear(cfg.d_model, VOCAB_SIZES[h], bias=False) for h in STAGE1}
        )
        # embeddings de conditionnement (distincts des embeddings d'entrée)
        self.cond_embed = nn.ModuleDict(
            {h: nn.Embedding(VOCAB_SIZES[h], cfg.d_model) for h in STAGE1}
        )
        self.ln_cond = nn.LayerNorm(cfg.d_model)
        self.stage2 = nn.ModuleDict(
            {h: nn.Linear(cfg.d_model, VOCAB_SIZES[h], bias=False) for h in STAGE2}
        )

    def forward(
        self,
        h: torch.Tensor,                         # (B, T, d_model)
        gt_stage1: Optional[dict[str, torch.Tensor]] = None,  # (B, T) par head
    ) -> dict[str, torch.Tensor]:
        logits: dict[str, torch.Tensor] = {}

        # Stage 1 — indépendant
        for head in STAGE1:
            logits[head] = self.stage1[head](h)  # (B, T, vocab)

        # Conditionnement pour stage 2
        if gt_stage1 is not None:
            # Training : teacher forcing avec les vrais labels
            cond = h
            for head in STAGE1:
                cond = cond + self.cond_embed[head](gt_stage1[head])
        else:
            # Inférence : argmax des prédictions de stage 1
            cond = h
            for head in STAGE1:
                pred = logits[head].argmax(dim=-1)  # (B, T)
                cond = cond + self.cond_embed[head](pred)

        cond = self.ln_cond(cond)

        # Stage 2 — conditionné
        for head in STAGE2:
            logits[head] = self.stage2[head](cond)  # (B, T, vocab)

        return logits


# ── Modèle principal ──────────────────────────────────────────────────────────

class CompoundWordTransformer(nn.Module):

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg    = cfg
        self.embed  = CompoundEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.output = TwoStageOutputLayer(cfg)
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
    ) -> dict[str, torch.Tensor]:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.output(x, gt_stage1)

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

        gt_s1  = {h: tgt[h] for h in STAGE1}
        logits = self.forward(inp, gt_stage1=gt_s1)

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
                    mask = mask & ~((h1 == H1_BARLINE_ID) | (h1 == H1_SEP_ID))

                if h in ("h4", "h5", "h8"):
                    mask = mask & ~(h1 == H1_REST_ID)

                if h in ("h4", "h8"):
                    mask = mask & ~(h1 == H1_FIORITURA_ID)

                if h == "h12":
                    mask = mask & ~(h11 == H11_LH_ID)

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
