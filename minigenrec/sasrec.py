"""SASRec user encoder. Same events + rating emb as the LLM path."""

from __future__ import annotations

import torch
import torch.nn as nn


class SASRecEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 64,
        dropout: float = 0.2,
        n_ratings: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, dim)
        self.rating_emb = nn.Embedding(n_ratings, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        item_vecs: torch.Tensor,
        ratings: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # item_vecs [B, L, D], ratings [B, L] ints 1..5, pad_mask True = pad
        b, length, _ = item_vecs.shape
        if length > self.max_len:
            raise ValueError(f"seq len {length} > max_len {self.max_len}")
        pos = torch.arange(length, device=item_vecs.device).unsqueeze(0).expand(b, -1)
        x = item_vecs + self.pos_emb(pos) + self.rating_emb(ratings.clamp(0, 5))
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.dropout(x)
        # causal mask: True means "ignore"
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        # last non-pad position; all-pad rows -> zeros
        lengths = (~pad_mask).sum(dim=1)
        idx = lengths.clamp(min=1) - 1
        h = out[torch.arange(b, device=out.device), idx]
        return torch.where(lengths.unsqueeze(-1) > 0, h, torch.zeros_like(h))


class SASRecRanker(nn.Module):
    """Full catalog scorer. ItemEncoder lives on this module (independent weights per run)."""

    def __init__(self, item_encoder, score_head, sasrec: SASRecEncoder):
        super().__init__()
        self.item_encoder = item_encoder
        self.score_head = score_head
        self.sasrec = sasrec

    def forward(
        self,
        item_ids: torch.Tensor,
        ratings: torch.Tensor,
        pad_mask: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        catalog = self.item_encoder(text_emb)
        seq = self.item_encoder.encode_ids(item_ids, catalog)
        h = self.sasrec(seq, ratings, pad_mask)
        return self.score_head(h, catalog)
