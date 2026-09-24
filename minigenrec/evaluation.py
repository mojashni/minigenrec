"""Catalog evaluation. Masks come only from candidate_mask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minigenrec.config import TOP_K
from minigenrec.data import Dataset, candidate_mask
from minigenrec.metrics import metrics_from_ranks, ranks


def cold_topk_share(
    scores: np.ndarray,
    mask: np.ndarray,
    cold_mask: np.ndarray,
    k: int = TOP_K,
) -> np.ndarray:
    """Expected cold share in top-k, splitting boundary ties proportionally."""
    if k <= 0:
        raise ValueError("k must be positive")
    shares = np.empty(len(scores), dtype=np.float64)
    for i in range(len(scores)):
        valid = ~mask[i]
        n_take = min(k, int(valid.sum()))
        if n_take == 0:
            shares[i] = np.nan
            continue
        valid_scores = scores[i, valid]
        valid_cold = cold_mask[valid]
        threshold = np.partition(valid_scores, len(valid_scores) - n_take)[
            len(valid_scores) - n_take
        ]
        above = valid_scores > threshold
        tied = valid_scores == threshold
        remaining = n_take - int(above.sum())
        expected_cold = float(valid_cold[above].sum())
        expected_cold += remaining * float(valid_cold[tied].mean())
        shares[i] = expected_cold / n_take
    return shares


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
        cold_topk = cold_topk_share(scores.astype(np.float64), mask, dataset.cold_mask)
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
                    "cold_topk": cold_topk,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
