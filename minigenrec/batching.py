"""Batch helpers: same events for every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from minigenrec.config import MAX_EVENTS_K
from minigenrec.data import Dataset, select_history


def history_for_row(dataset: Dataset, row, k: int = MAX_EVENTS_K, full_history: bool = False):
    uid = int(row["user_id"])
    hend = int(row["hist_end"])
    items = dataset.warm_items[uid][:hend]
    ratings = dataset.warm_ratings[uid][:hend]
    if full_history:
        return items.copy(), ratings.copy()
    return select_history(items, ratings, k)


def collate_history(
    dataset: Dataset,
    batch: pd.DataFrame,
    k: int = MAX_EVENTS_K,
    full_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns item_ids, ratings, pad_mask (True=pad), targets_full."""
    seqs = [history_for_row(dataset, batch.iloc[i], k=k, full_history=full_history) for i in range(len(batch))]
    max_len = max((len(s[0]) for s in seqs), default=0)
    max_len = max(max_len, 1)
    b = len(batch)
    item_ids = torch.zeros(b, max_len, dtype=torch.long)
    ratings = torch.zeros(b, max_len, dtype=torch.long)
    pad_mask = torch.ones(b, max_len, dtype=torch.bool)
    for i, (items, rats) in enumerate(seqs):
        n = len(items)
        if n == 0:
            continue
        item_ids[i, -n:] = torch.from_numpy(np.asarray(items, dtype=np.int64).copy())
        ratings[i, -n:] = torch.from_numpy(np.asarray(rats, dtype=np.int64).copy())
        pad_mask[i, -n:] = False
    targets = torch.from_numpy(batch["target_full"].to_numpy(dtype=np.int64, copy=True))
    return item_ids, ratings, pad_mask, targets


def warm_ce_loss(logits_full: torch.Tensor, targets_full: torch.Tensor, warm_to_full: torch.Tensor, full_to_warm: torch.Tensor) -> torch.Tensor:
    """Softmax over warm catalog only. Cold items get no gradient as negatives."""
    warm_logits = logits_full.index_select(1, warm_to_full)
    targets_warm = full_to_warm[targets_full]
    if (targets_warm < 0).any():
        raise ValueError("cold target in train loss")
    return torch.nn.functional.cross_entropy(warm_logits, targets_warm)
