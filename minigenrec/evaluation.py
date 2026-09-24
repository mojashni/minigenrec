"""Catalog evaluation. Masks come only from candidate_mask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minigenrec.data import Dataset, candidate_mask
from minigenrec.metrics import metrics_from_ranks, ranks


def evaluate(
    dataset: Dataset,
    split_df: pd.DataFrame,
    score_fn,
    batch_size: int,
    warm_only: bool = False,
) -> pd.DataFrame:
    columns = ["user_id", "target_movie", "rank", "hit", "ndcg", "mrr", "bucket"]
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
        targets = batch["target_full"].to_numpy()
        rank = ranks(scores, targets, mask)
        hit, ndcg, mrr = metrics_from_ranks(rank)
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
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
