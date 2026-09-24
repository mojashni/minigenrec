"""Shared item encoder and scoring head. Architecture shared across runs, not weights."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inv(y: float) -> float:
    # raw so softplus(raw) == y
    return math.log(math.expm1(y))


class ItemEncoder(nn.Module):
    """id | text | hybrid. Cold items force m_i=0 so id never enters the score."""

    def __init__(
        self,
        n_items: int,
        dim: int,
        mode: str = "id",
        text_dim: int | None = None,
        cold_mask: torch.Tensor | None = None,
        init_lambda: float = 0.1,
    ):
        super().__init__()
        if mode not in ("id", "text", "hybrid"):
            raise ValueError(mode)
        self.mode = mode
        self.dim = dim
        self.id_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.register_buffer(
            "cold_mask",
            cold_mask.clone().bool() if cold_mask is not None else torch.zeros(n_items, dtype=torch.bool),
            persistent=True,
        )
        if mode in ("text", "hybrid"):
            if text_dim is None:
                raise ValueError("text_dim required for text/hybrid")
            self.w_text = nn.Linear(text_dim, dim, bias=False)
        else:
            self.w_text = None
        if mode == "hybrid":
            self.raw_lambda = nn.Parameter(torch.tensor(softplus_inv(init_lambda)))
        else:
            self.raw_lambda = None

    def forward(self, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        # returns [N, D] for the full catalog
        id_part = F.normalize(self.id_emb.weight, dim=-1, eps=1e-6)
        if self.mode == "id":
            return id_part
        assert text_emb is not None and self.w_text is not None
        text_part = F.normalize(self.w_text(text_emb), dim=-1, eps=1e-6)
        if self.mode == "text":
            return text_part
        assert self.raw_lambda is not None
        lam = F.softplus(self.raw_lambda)
        m = (~self.cold_mask).to(dtype=text_part.dtype).unsqueeze(-1)
        return F.normalize(text_part + lam * m * id_part, dim=-1, eps=1e-6)

    def encode_ids(self, item_ids: torch.Tensor, catalog: torch.Tensor) -> torch.Tensor:
        # item_ids [B, L], catalog [N, D] -> [B, L, D]
        return catalog[item_ids]


class ScoreHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 20.0):
        super().__init__()
        self.w_user = nn.Linear(in_dim, out_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, h: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.w_user(h), dim=-1, eps=1e-6)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        scores = scale * (u @ item_emb.T)
        return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
