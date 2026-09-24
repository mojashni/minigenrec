"""Catalog evaluation. Masks come only from candidate_mask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minigenrec.config import TOP_K
from minigenrec.data import Dataset, candidate_mask
from minigenrec.metrics import metrics_from_ranks, ranks


def mean_hit(dataset: Dataset, split_df: pd.DataFrame, score_fn, batch_size: int) -> float:
    """Checkpoint-selection metric: warm-only Hit@K, so cold items never influence selection."""
    frame = evaluate(dataset, split_df, score_fn, batch_size, warm_only=True)
    if len(frame) == 0:
        return 0.0
    return float(frame["hit"].mean())


def evaluate(
    dataset: Dataset,
    split_df: pd.DataFrame,
    score_fn,
    batch_size: int,
    warm_only: bool = False,
    cold_only: bool = False,
) -> pd.DataFrame:
    """Per-example ranks. `cold_topk` = share of the top-K taken by cold items."""
    if warm_only and cold_only:
        raise ValueError("warm_only and cold_only are exclusive")
    columns = ["user_id", "target_movie", "rank", "hit", "ndcg", "mrr", "bucket", "cold_topk"]
    if len(split_df) == 0:
        return pd.DataFrame(columns=columns)
    n_full = len(dataset.full_to_movie)
    parts: list[pd.DataFrame] = []
    for start in range(0, len(split_df), batch_size):
        batch = split_df.iloc[start : start + batch_size]
        scores = score_fn(batch)
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores)
        if scores.shape != (len(batch), n_full):
            raise ValueError(f"score_fn must return [B, {n_full}], got {scores.shape}")
        mask = np.stack(
            [candidate_mask(dataset, batch.iloc[i], warm_only=warm_only) for i in range(len(batch))]
        )
        if cold_only:
            mask |= ~dataset.cold_mask
        targets = batch["target_full"].to_numpy()
        rank = ranks(scores, targets, mask)
        hit, ndcg, mrr = metrics_from_ranks(rank)
        masked = np.where(mask, -np.inf, scores.astype(np.float64))
        top = np.argpartition(-masked, TOP_K - 1, axis=1)[:, :TOP_K]
        parts.append(
            pd.DataFrame(
                {
                    "user_id": batch["user_id"].to_numpy(),
                    "target_movie": batch["target_movie"].to_numpy(),
                    "rank": rank,
                    "hit": hit,
                    "ndcg": ndcg,
                    "mrr": mrr,
                    "bucket": dataset.pop_buckets[targets],
                    "cold_topk": dataset.cold_mask[top].mean(axis=1),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
