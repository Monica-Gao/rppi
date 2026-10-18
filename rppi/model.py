from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.bool(), fill_value)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float, cross: bool = False):
        super().__init__()
        self.cross = cross
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(x)
        if self.cross:
            assert kv is not None
            k = self.norm_kv(kv)
            out, _ = self.attn(q, k, k, key_padding_mask=None if kv_mask is None else ~kv_mask.bool(), need_weights=False)
        else:
            out, _ = self.attn(q, q, q, key_padding_mask=None if x_mask is None else ~x_mask.bool(), need_weights=False)
        x = x + out
        if x_mask is not None:
            x = x * x_mask.unsqueeze(-1).to(x.dtype)
        x = x + self.ff(self.norm_ff(x))
        if x_mask is not None:
            x = x * x_mask.unsqueeze(-1).to(x.dtype)
        return x


class CompatibilityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 2,
        cross_layers: int = 2,
        ff_dim: int = 768,
        dropout: float = 0.2,
        chunk_size: int = 16,
        sequence_length: int = 795,
        projection_dim: int = 256,
        shortcut_adversary: bool = False,
        degree_bins: int = 4,
        shared_cross_attention: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.chunk_size = chunk_size
        self.max_tokens = sequence_length // chunk_size
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, self.max_tokens + 1, hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.self_layers = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, ff_dim, dropout, cross=False) for _ in range(num_layers)]
        )
        self.shared_cross_attention = shared_cross_attention
        self.cross_a = nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, ff_dim, dropout, cross=True) for _ in range(cross_layers)]
        )
        self.cross_b = self.cross_a if shared_cross_attention else nn.ModuleList(
            [EncoderLayer(hidden_dim, num_heads, ff_dim, dropout, cross=True) for _ in range(cross_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.attn_pool = AttentionPooling(hidden_dim)
        summary_dim = hidden_dim * 4
        compat_dim = summary_dim * 4
        self.compat_projector = nn.Sequential(
            nn.LayerNorm(compat_dim),
            nn.Linear(compat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, projection_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.shortcut_adversary = shortcut_adversary
        if shortcut_adversary:
            self.same_species_head = nn.Sequential(
                nn.LayerNorm(projection_dim),
                nn.Linear(projection_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.degree_head = nn.Sequential(
                nn.LayerNorm(projection_dim),
                nn.Linear(projection_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, degree_bins),
            )

    def _chunk(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, dim = x.shape
        usable = (seq_len // self.chunk_size) * self.chunk_size
        x = x[:, :usable]
        if mask is not None:
            mask = mask[:, :usable].bool()
        x = x.view(bsz, usable // self.chunk_size, self.chunk_size, dim)
        if mask is None:
            return x.mean(dim=2), None
        m = mask.view(bsz, usable // self.chunk_size, self.chunk_size)
        mf = m.unsqueeze(-1).to(x.dtype)
        pooled = (x * mf).sum(dim=2) / mf.sum(dim=2).clamp_min(1.0)
        return pooled, m.any(dim=2)

    def _embed(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.dropout(F.gelu(self.input_proj(self.input_norm(x))))
        x, chunk_mask = self._chunk(x, mask)
        bsz, ntok, _ = x.shape
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        token_mask = None
        if chunk_mask is not None:
            token_mask = torch.cat(
                [torch.ones((bsz, 1), dtype=torch.bool, device=x.device), chunk_mask],
                dim=1,
            )
        x = x + self.pos[:, : ntok + 1]
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return x, token_mask

    def _summary(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        cls = x[:, 0]
        body = x[:, 1:]
        body_mask = None if mask is None else mask[:, 1:].bool()
        if body_mask is None:
            mean = body.mean(dim=1)
            maxv = body.max(dim=1).values
        else:
            mf = body_mask.unsqueeze(-1).to(body.dtype)
            mean = (body * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)
            maxv = body.masked_fill(~body_mask.unsqueeze(-1), torch.finfo(body.dtype).min).max(dim=1).values
            maxv = torch.where(body_mask.any(dim=1, keepdim=True), maxv, torch.zeros_like(maxv))
        return torch.cat([cls, mean, maxv, self.attn_pool(body, body_mask)], dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        shortcut_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        z1, tm1 = self._embed(x1, mask1)
        z2, tm2 = self._embed(x2, mask2)
        for layer in self.self_layers:
            z1 = layer(z1, x_mask=tm1)
            z2 = layer(z2, x_mask=tm2)
        for la, lb in zip(self.cross_a, self.cross_b):
            new1 = la(z1, kv=z2, x_mask=tm1, kv_mask=tm2)
            new2 = lb(z2, kv=z1, x_mask=tm2, kv_mask=tm1)
            z1, z2 = new1, new2
        z1 = self.out_norm(z1)
        z2 = self.out_norm(z2)
        s1 = self._summary(z1, tm1)
        s2 = self._summary(z2, tm2)
        diff = s1 - s2
        compat_raw = torch.cat([s1 + s2, torch.abs(diff), s1 * s2, diff.pow(2)], dim=1)
        embedding = self.compat_projector(compat_raw)
        logits = self.classifier(embedding).squeeze(-1)
        out = {"logits": logits, "embedding": embedding}
        if self.shortcut_adversary:
            reversed_embedding = GradientReversal.apply(embedding, shortcut_lambda)
            out["same_species_logits"] = self.same_species_head(reversed_embedding).squeeze(-1)
            out["degree_logits"] = self.degree_head(reversed_embedding)
        return out


def build_model(config: Dict) -> CompatibilityEncoder:
    return CompatibilityEncoder(**config)
