"""MovieLens 1M loading and the pre-registered split contract."""

from __future__ import annotations

import json
import hashlib
import pickle
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from minigenrec.config import (
    COLD_MIN_INTERACTIONS,
    COLD_MIN_USERS,
    COLD_RATIO_DEFAULT,
    COLD_RATIO_FALLBACK,
    COLD_SPLIT_SEED,
    DATA_DIR,
    ML1M_URL,
    POSITIVE_MIN_RATING,
    RESULTS_DIR,
)

# GroupLens omitted the space in "Associate, The (L'Associe)(1982)".
# Title must end on a non-space so the optional separator is not swallowed into it.
_TITLE_YEAR = re.compile(r"^(.*\S) ?\((\d{4})\)$")
_SPLIT_COLUMNS = [
    "user_id",
    "target_full",
    "target_movie",
    "target_rating",
    "target_pos",
    "hist_end",
]
DATASET_CACHE_VERSION = 2


def dataset_cache_fingerprint() -> str:
    """Fingerprint every setting that changes the split or catalog mappings."""
    payload = {
        "version": DATASET_CACHE_VERSION,
        "positive_min_rating": POSITIVE_MIN_RATING,
        "cold_split_seed": COLD_SPLIT_SEED,
        "cold_ratio_default": COLD_RATIO_DEFAULT,
        "cold_ratio_fallback": COLD_RATIO_FALLBACK,
        "cold_min_users": COLD_MIN_USERS,
        "cold_min_interactions": COLD_MIN_INTERACTIONS,
        "source": ML1M_URL,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_dataset_cache(path, fingerprint: str) -> Dataset | None:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    dataset = payload.get("dataset")
    return dataset if isinstance(dataset, Dataset) else None


def _write_dataset_cache(path, dataset: Dataset, fingerprint: str) -> None:
    with path.open("wb") as fh:
        pickle.dump(
            {"fingerprint": fingerprint, "dataset": dataset},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@dataclass
class Dataset:
    ratings: pd.DataFrame
    movies: pd.DataFrame | None
    movie_to_full: dict[int, int]
    full_to_movie: np.ndarray
    full_to_warm: np.ndarray
    warm_to_full: np.ndarray
    cold_mask: np.ndarray
    pop_counts: np.ndarray
    pop_deciles: np.ndarray
    pop_buckets: np.ndarray
    warm_items: dict[int, np.ndarray]
    warm_ratings: dict[int, np.ndarray]
    warm_pos: dict[int, np.ndarray]
    all_items_full: dict[int, np.ndarray]
    all_pos: dict[int, np.ndarray]
    train_cutoff: dict[int, int]
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cold_test: pd.DataFrame
    cold_ratio: float | None
    cold_attempts: list[dict]


def download_ml1m() -> None:
    dest = DATA_DIR / "ml-1m"
    if (dest / "ratings.dat").exists() and (dest / "movies.dat").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"
    if not zip_path.exists():
        req = urllib.request.Request(ML1M_URL, headers={"User-Agent": "minigenrec"})
        with urllib.request.urlopen(req, timeout=120) as resp, zip_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    if not (dest / "ratings.dat").exists():
        raise FileNotFoundError(dest / "ratings.dat")


def _read_dat(path, encoding: str, n_fields: int) -> list[list[str]]:
    rows = []
    with path.open(encoding=encoding) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("::", n_fields - 1)
            if len(parts) != n_fields:
                raise ValueError(f"{path.name}: expected {n_fields} fields, got {parts!r}")
            rows.append(parts)
    return rows


def load_ml1m() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = DATA_DIR / "ml-1m"
    raw = _read_dat(root / "ratings.dat", "utf-8", 4)
    ratings = pd.DataFrame(raw, columns=["user_id", "movie_id", "rating", "timestamp"])
    for col in ratings.columns:
        ratings[col] = ratings[col].astype(np.int64)

    raw_movies = _read_dat(root / "movies.dat", "latin-1", 3)
    movies = pd.DataFrame(raw_movies, columns=["movie_id", "title", "genres"])
    movies["movie_id"] = movies["movie_id"].astype(np.int64)
    parsed = movies["title"].str.extract(_TITLE_YEAR)
    if parsed[1].isna().any():
        bad = movies.loc[parsed[1].isna(), "title"].head(5).tolist()
        raise ValueError(f"titles missing trailing (YYYY): {bad}")
    movies["title"] = parsed[0]
    movies["year"] = parsed[1].astype(np.int64)
    movies["genres"] = movies["genres"].str.split("|")
    movies = movies[["movie_id", "title", "year", "genres"]]
    return ratings, movies


def popularity_deciles(counts: pd.Series) -> pd.Series:
    # rank(method="first") breaks count ties by row order; sort so that order is movie_id.
    counts = counts.sort_index()
    deciles = pd.qcut(counts.rank(method="first"), 10, labels=False)
    return deciles.astype(np.int64)


def decile_buckets(deciles: np.ndarray) -> np.ndarray:
    buckets = np.empty(len(deciles), dtype=object)
    buckets[deciles <= 3] = "tail"
    buckets[(deciles >= 4) & (deciles <= 7)] = "mid"
    buckets[deciles >= 8] = "head"
    return buckets


def select_cold_items(counts: pd.Series, ratio: float, seed: int) -> np.ndarray:
    deciles = popularity_deciles(counts)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for decile in range(10):
        ids = np.sort(deciles.index[deciles.to_numpy() == decile].to_numpy().astype(np.int64))
        n_take = round(ratio * len(ids))
        if n_take == 0:
            continue
        chosen.append(rng.choice(ids, size=n_take, replace=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen)


def _rating_counts(ratings: pd.DataFrame) -> pd.Series:
    counts = ratings.groupby("movie_id").size()
    counts.index = counts.index.astype(np.int64)
    return counts


def _empty_split() -> pd.DataFrame:
    return pd.DataFrame(columns=_SPLIT_COLUMNS)


def _example(user_id: int, target_full: int, target_rating: int, target_pos: int, warm_pos: np.ndarray, full_to_movie: np.ndarray) -> dict:
    return {
        "user_id": user_id,
        "target_full": int(target_full),
        "target_movie": int(full_to_movie[target_full]),
        "target_rating": int(target_rating),
        "target_pos": int(target_pos),
        "hist_end": int(np.searchsorted(warm_pos, target_pos, side="left")),
    }


def build_dataset(
    ratings: pd.DataFrame,
    cold_items: np.ndarray,
    movies: pd.DataFrame | None = None,
    cold_ratio: float | None = None,
) -> Dataset:
    """Split builder. `cold_items` are movie ids; the full-catalog mask is derived here."""
    ratings = ratings.loc[:, ["user_id", "movie_id", "rating", "timestamp"]].copy()
    for col in ratings.columns:
        ratings[col] = ratings[col].astype(np.int64)

    full_to_movie = np.sort(ratings["movie_id"].unique())
    movie_to_full = {int(movie): i for i, movie in enumerate(full_to_movie)}
    cold_ids = {int(movie) for movie in np.asarray(cold_items).reshape(-1)}
    cold_mask = np.array([int(movie) in cold_ids for movie in full_to_movie], dtype=bool)
    warm_to_full = np.flatnonzero(~cold_mask).astype(np.int64)
    full_to_warm = np.full(len(full_to_movie), -1, dtype=np.int64)
    full_to_warm[warm_to_full] = np.arange(len(warm_to_full), dtype=np.int64)

    counts = _rating_counts(ratings)
    deciles = popularity_deciles(counts)
    pop_counts = counts.reindex(full_to_movie).to_numpy(dtype=np.int64)
    pop_deciles = deciles.reindex(full_to_movie).to_numpy(dtype=np.int64)
    pop_buckets = decile_buckets(pop_deciles)

    # mergesort is stable: equal (timestamp, movie_id) keep their original order.
    ordered = ratings.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    movie_ids = ordered["movie_id"].to_numpy()
    full_ids = np.searchsorted(full_to_movie, movie_ids)
    if not np.array_equal(full_to_movie[full_ids], movie_ids):
        raise ValueError("rating movie_id missing from the catalog")
    ordered = ordered.assign(_full=full_ids)

    warm_items: dict[int, np.ndarray] = {}
    warm_ratings: dict[int, np.ndarray] = {}
    warm_pos: dict[int, np.ndarray] = {}
    all_items_full: dict[int, np.ndarray] = {}
    all_pos: dict[int, np.ndarray] = {}
    train_cutoff: dict[int, int] = {}
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    cold_rows: list[dict] = []

    for user_id, grp in ordered.groupby("user_id", sort=False):
        uid = int(user_id)
        items = grp["_full"].to_numpy(dtype=np.int64)
        rats = grp["rating"].to_numpy(dtype=np.int64)
        pos = np.arange(len(grp), dtype=np.int64)
        warm_at = np.flatnonzero(~cold_mask[items])
        w_items = items[warm_at]
        w_rats = rats[warm_at]
        w_pos = pos[warm_at]
        positive_at = np.flatnonzero(w_rats >= POSITIVE_MIN_RATING)
        if len(positive_at) < 3:
            continue
        warm_items[uid] = w_items
        warm_ratings[uid] = w_rats
        warm_pos[uid] = w_pos
        all_items_full[uid] = items
        all_pos[uid] = pos
        train_at = positive_at[:-2]
        val_at = int(positive_at[-2])
        test_at = int(positive_at[-1])
        cutoff = int(w_pos[train_at[-1]])
        train_cutoff[uid] = cutoff
        for idx in train_at:
            train_rows.append(_example(uid, w_items[idx], w_rats[idx], w_pos[idx], w_pos, full_to_movie))
        val_rows.append(_example(uid, w_items[val_at], w_rats[val_at], w_pos[val_at], w_pos, full_to_movie))
        test_rows.append(_example(uid, w_items[test_at], w_rats[test_at], w_pos[test_at], w_pos, full_to_movie))
        for j in np.flatnonzero(cold_mask[items]):
            if rats[j] >= POSITIVE_MIN_RATING and pos[j] > cutoff:
                cold_rows.append(_example(uid, items[j], rats[j], pos[j], w_pos, full_to_movie))

    train = pd.DataFrame(train_rows, columns=_SPLIT_COLUMNS) if train_rows else _empty_split()
    val = pd.DataFrame(val_rows, columns=_SPLIT_COLUMNS) if val_rows else _empty_split()
    test = pd.DataFrame(test_rows, columns=_SPLIT_COLUMNS) if test_rows else _empty_split()
    cold_test = pd.DataFrame(cold_rows, columns=_SPLIT_COLUMNS) if cold_rows else _empty_split()

    kept = set(train_cutoff)
    assert set(train["user_id"].astype(int)) == kept
    assert set(val["user_id"].astype(int)) == kept
    assert set(test["user_id"].astype(int)) == kept
    assert set(cold_test["user_id"].astype(int)) <= kept
    if len(train):
        assert not cold_mask[train["target_full"].to_numpy()].any()
        assert not cold_mask[val["target_full"].to_numpy()].any()
        assert not cold_mask[test["target_full"].to_numpy()].any()
    if len(cold_test):
        assert cold_mask[cold_test["target_full"].to_numpy()].all()
        cut = cold_test["user_id"].map(train_cutoff).to_numpy()
        assert np.all(cold_test["target_pos"].to_numpy() > cut)
    assert np.all(full_to_warm[cold_mask] == -1)
    assert np.array_equal(full_to_warm[warm_to_full], np.arange(len(warm_to_full)))

    return Dataset(
        ratings=ratings,
        movies=movies,
        movie_to_full=movie_to_full,
        full_to_movie=full_to_movie,
        full_to_warm=full_to_warm,
        warm_to_full=warm_to_full,
        cold_mask=cold_mask,
        pop_counts=pop_counts,
        pop_deciles=pop_deciles,
        pop_buckets=pop_buckets,
        warm_items=warm_items,
        warm_ratings=warm_ratings,
        warm_pos=warm_pos,
        all_items_full=all_items_full,
        all_pos=all_pos,
        train_cutoff=train_cutoff,
        train=train,
        val=val,
        test=test,
        cold_test=cold_test,
        cold_ratio=cold_ratio,
        cold_attempts=[],
    )


def choose_cold_ratio(
    ratings: pd.DataFrame,
    movies: pd.DataFrame | None = None,
) -> tuple[Dataset, dict]:
    """Pick 10% unless cold-test is below the pre-registered minimums, then 20%."""
    counts = _rating_counts(ratings)
    attempts: list[dict] = []
    chosen: Dataset | None = None
    for ratio in (COLD_RATIO_DEFAULT, COLD_RATIO_FALLBACK):
        cold_items = select_cold_items(counts, ratio, COLD_SPLIT_SEED)
        ds = build_dataset(ratings, cold_items, movies=movies, cold_ratio=ratio)
        n_users = int(ds.cold_test["user_id"].nunique()) if len(ds.cold_test) else 0
        n_interactions = int(len(ds.cold_test))
        attempts.append({"ratio": ratio, "n_users": n_users, "n_interactions": n_interactions})
        chosen = ds
        if n_users >= COLD_MIN_USERS and n_interactions >= COLD_MIN_INTERACTIONS:
            break
    assert chosen is not None
    chosen.cold_attempts = attempts
    return chosen, {"cold_ratio": chosen.cold_ratio, "attempts": attempts}


def data_stats(dataset: Dataset) -> dict:
    if len(dataset.cold_test):
        buckets = dataset.pop_buckets[dataset.cold_test["target_full"].to_numpy()]
        by_bucket = {name: int((buckets == name).sum()) for name in ("head", "mid", "tail")}
        cold_users = int(dataset.cold_test["user_id"].nunique())
    else:
        by_bucket = {"head": 0, "mid": 0, "tail": 0}
        cold_users = 0
    return {
        "n_users_kept": len(dataset.train_cutoff),
        "n_items": int(len(dataset.full_to_movie)),
        "n_warm": int((~dataset.cold_mask).sum()),
        "n_cold": int(dataset.cold_mask.sum()),
        "cold_ratio": dataset.cold_ratio,
        "attempts": dataset.cold_attempts,
        "n_train": int(len(dataset.train)),
        "n_val": int(len(dataset.val)),
        "n_test": int(len(dataset.test)),
        "n_cold_test": int(len(dataset.cold_test)),
        "cold_test_users": cold_users,
        "cold_test_by_bucket": by_bucket,
    }


def prepare() -> Dataset:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "dataset.pkl"
    fingerprint = dataset_cache_fingerprint()
    dataset = _load_dataset_cache(cache, fingerprint) if cache.exists() else None
    if dataset is None:
        download_ml1m()
        ratings, movies = load_ml1m()
        dataset, _stats = choose_cold_ratio(ratings, movies)
        _write_dataset_cache(cache, dataset, fingerprint)
    stats = data_stats(dataset)
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return dataset


def sample_train_subsets(
    train_df: pd.DataFrame,
    sizes: tuple[int, ...] | list[int],
    seed: int,
) -> dict[int, pd.DataFrame]:
    for n in sizes:
        if n > len(train_df):
            raise ValueError(f"requested {n} train rows, only {len(train_df)} available")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_df))
    return {int(n): train_df.iloc[order[:n]].reset_index(drop=True) for n in sizes}


def select_history(
    items: np.ndarray,
    ratings: np.ndarray,
    k: int,
    event_cost: np.ndarray | None = None,
    budget: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Most recent k warm events, oldest→newest. Budget drops whole oldest events."""
    items = np.asarray(items)
    ratings = np.asarray(ratings)
    if len(items) != len(ratings):
        raise ValueError("items and ratings must have the same length")
    if k <= 0:
        sel_items = items[:0]
        sel_ratings = ratings[:0]
    else:
        sel_items = items[-k:]
        sel_ratings = ratings[-k:]
    if event_cost is not None and budget is not None:
        cost = np.asarray(event_cost)
        # Whole events only: drop the oldest selected line until the sum fits.
        while len(sel_items) and cost[sel_items].sum() > budget:
            sel_items = sel_items[1:]
            sel_ratings = sel_ratings[1:]
    return np.asarray(sel_items).copy(), np.asarray(sel_ratings).copy()


def candidate_mask(dataset: Dataset, example_row, warm_only: bool = False) -> np.ndarray:
    user_id = int(example_row["user_id"])
    target_pos = int(example_row["target_pos"])
    target_full = int(example_row["target_full"])
    mask = np.zeros(len(dataset.full_to_movie), dtype=bool)
    all_items = dataset.all_items_full[user_id]
    all_pos = dataset.all_pos[user_id]
    mask[all_items[all_pos < target_pos]] = True
    if warm_only:
        mask |= dataset.cold_mask
    assert not mask[target_full]
    return mask


def main() -> None:
    prepare()
    print((RESULTS_DIR / "data_stats.json").read_text())


if __name__ == "__main__":
    # Running this file as __main__ would pickle Dataset under the wrong module.
    from minigenrec.data import main as entry

    entry()
