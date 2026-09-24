"""Batch helpers: same events for every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from minigenrec.config import MAX_EVENTS_K, MAX_LEN
from minigenrec.data import Dataset, select_history
from minigenrec.verbalize import build_prompt, build_title_map, event_line, fit_events_to_budget


class HistorySelector:
    """Canonical, memoized event selection shared by every model/title mode."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        k: int = MAX_EVENTS_K,
        full_history: bool = False,
        max_len: int = MAX_LEN,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.k = k
        self.full_history = full_history
        self.max_len = max_len
        self.title_maps = {
            mode: build_title_map(dataset.movies, mode)
            for mode in ("real", "shuffled", "none")
        }
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def select(self, row) -> tuple[np.ndarray, np.ndarray]:
        uid = int(row["user_id"])
        hend = int(row["hist_end"])
        key = (uid, hend)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        items = self.dataset.warm_items[uid][:hend]
        ratings = self.dataset.warm_ratings[uid][:hend]
        if self.full_history:
            items, ratings = items.copy(), ratings.copy()
        else:
            items, ratings = select_history(items, ratings, self.k)

        # Use the strictest of all title modes. This makes ID-only SASRec and
        # every title ablation see exactly the same events.
        drop = 0
        for title_map in self.title_maps.values():
            lines = [
                event_line(title_map[int(self.dataset.full_to_movie[int(fid)])], int(r))
                for fid, r in zip(items, ratings)
            ]
            kept = fit_events_to_budget(lines, self.tokenizer, self.max_len)
            drop = max(drop, len(lines) - len(kept))
        selected = (items[drop:].copy(), ratings[drop:].copy())
        self._cache[key] = selected
        return selected


def history_for_row(
    dataset: Dataset,
    row,
    k: int = MAX_EVENTS_K,
    full_history: bool = False,
    selector: HistorySelector | None = None,
):
    """Return selected events; prefer a canonical HistorySelector for experiments."""
    if selector is not None:
        return selector.select(row)
    uid = int(row["user_id"])
    hend = int(row["hist_end"])
    items = dataset.warm_items[uid][:hend]
    ratings = dataset.warm_ratings[uid][:hend]
    if full_history:
        items, ratings = items.copy(), ratings.copy()
    else:
        items, ratings = select_history(items, ratings, k)
    return items, ratings


def collate_history(
    dataset: Dataset,
    batch: pd.DataFrame,
    k: int = MAX_EVENTS_K,
    full_history: bool = False,
    selector: HistorySelector | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns item_ids, ratings, pad_mask (True=pad), targets_full."""
    seqs = [
        history_for_row(
            dataset,
            batch.iloc[i],
            k=k,
            full_history=full_history,
            selector=selector,
        )
        for i in range(len(batch))
    ]
    max_len_b = max((len(s[0]) for s in seqs), default=0)
    max_len_b = max(max_len_b, 1)
    b = len(batch)
    item_ids = torch.zeros(b, max_len_b, dtype=torch.long)
    ratings = torch.zeros(b, max_len_b, dtype=torch.long)
    pad_mask = torch.ones(b, max_len_b, dtype=torch.bool)
    for i, (items, rats) in enumerate(seqs):
        n = len(items)
        if n == 0:
            continue
        # Right-pad so SASRec positions always start at zero and `lengths - 1`
        # points at the most recent event, independent of the other batch rows.
        item_ids[i, :n] = torch.from_numpy(np.asarray(items, dtype=np.int64).copy())
        ratings[i, :n] = torch.from_numpy(np.asarray(rats, dtype=np.int64).copy())
        pad_mask[i, :n] = False
    targets = torch.from_numpy(batch["target_full"].to_numpy(dtype=np.int64, copy=True))
    return item_ids, ratings, pad_mask, targets


def prompts_from_history(dataset, items, ratings, title_map: dict[int, str]) -> str:
    lines = [event_line(title_map[int(dataset.full_to_movie[int(fid)])], int(r)) for fid, r in zip(items, ratings)]
    return build_prompt(lines)


def warm_ce_loss(logits_full: torch.Tensor, targets_full: torch.Tensor, warm_to_full: torch.Tensor, full_to_warm: torch.Tensor) -> torch.Tensor:
    """Softmax over warm catalog only. Cold items get no gradient as negatives."""
    warm_logits = logits_full.index_select(1, warm_to_full)
    targets_warm = full_to_warm[targets_full]
    if (targets_warm < 0).any():
        raise ValueError("cold target in train loss")
    loss = torch.nn.functional.cross_entropy(warm_logits, targets_warm)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite warm-catalog cross-entropy loss")
    return loss
