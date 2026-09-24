"""Ranking metrics. Ties are pessimistic so a constant scorer gets no free credit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from minigenrec.config import TOP_K


def _as_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return value.numpy()
    return np.asarray(value)


def ranks(scores, targets, mask) -> np.ndarray:
    """1-based rank of each target. `mask` True means excluded from the catalog."""
    scores = _as_array(scores)
    targets = _as_array(targets).astype(np.int64, copy=False)
    mask = _as_array(mask).astype(bool, copy=False)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, N]")
    batch, _n = scores.shape
    if targets.shape != (batch,):
        raise ValueError("targets must have shape [B]")
    if mask.shape != scores.shape:
        raise ValueError("mask must have shape [B, N]")
    if mask[np.arange(batch), targets].any():
        raise ValueError("target is masked")
    scores = np.array(scores, dtype=np.float64, copy=True)
    unmasked = ~mask
    if not np.isfinite(scores[unmasked]).all():
        raise ValueError("unmasked scores contain NaN or Inf")
    scores[mask] = -np.inf
    target_scores = scores[np.arange(batch), targets]
    greater = np.sum(scores > target_scores[:, None], axis=1)
    # Equality count includes the target itself; drop that one.
    tied = np.sum(scores == target_scores[:, None], axis=1) - 1
    return (1 + greater + tied).astype(np.int64)


def metrics_from_ranks(rank, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = np.asarray(rank)
    hit = (rank <= k).astype(np.float64)
    ndcg = np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)
    mrr = 1.0 / rank.astype(np.float64)
    return hit, ndcg, mrr


def user_level(per_example_df: pd.DataFrame) -> pd.DataFrame:
    return per_example_df.groupby("user_id")[["hit", "ndcg", "mrr"]].mean()


def bootstrap_ci(per_user_series: pd.Series, n: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(per_user_series, dtype=np.float64)
    if values.size == 0:
        raise ValueError("empty user series")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n, values.size))
    means = values[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(
    per_user_a: pd.Series,
    per_user_b: pd.Series,
    n: int,
    seed: int,
) -> tuple[float, float, float]:
    a = per_user_a.sort_index()
    b = per_user_b.sort_index()
    if not a.index.equals(b.index):
        raise ValueError("paired bootstrap user sets differ")
    va = a.to_numpy(dtype=np.float64)
    vb = b.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(va), size=(n, len(va)))
    diffs = va[draws].mean(axis=1) - vb[draws].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(va.mean() - vb.mean()), float(lo), float(hi)
