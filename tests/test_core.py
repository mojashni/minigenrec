"""Leakage, split, and metric contracts. Synthetic fixture, no download."""

import pickle

import numpy as np
import pandas as pd
import pytest

from minigenrec.config import (
    COLD_MIN_INTERACTIONS,
    COLD_MIN_USERS,
    COLD_RATIO_DEFAULT,
    COLD_RATIO_FALLBACK,
    DATA_DIR,
)
from minigenrec.data import (
    _load_dataset_cache,
    _write_dataset_cache,
    build_dataset,
    candidate_mask,
    choose_cold_ratio,
    dataset_cache_fingerprint,
    load_ml1m,
    popularity_deciles,
    sample_train_subsets,
    select_cold_items,
    select_history,
)
from minigenrec.metrics import metrics_from_ranks, paired_bootstrap, ranks

# Catalog movies. 40 and 50 are cold. Kept users: 1, 3, 6.
COLD_MOVIES = np.array([40, 50])


def synthetic_ratings() -> pd.DataFrame:
    rows = [
        # User 1: timestamp tie at t=100 must order by movie_id (10, 20, 30), not insert order.
        (1, 30, 5, 100),
        (1, 10, 3, 100),
        (1, 20, 5, 100),
        (1, 40, 5, 200),
        (1, 50, 2, 300),
        (1, 60, 4, 400),
        (1, 70, 5, 500),
        # User 2: two warm positives. Dropped, including their cold positive.
        (2, 10, 5, 100),
        (2, 20, 5, 200),
        (2, 40, 5, 300),
        # User 3: cold positive before train_cutoff.
        (3, 40, 5, 50),
        (3, 10, 5, 100),
        (3, 20, 5, 200),
        (3, 30, 5, 300),
        # User 4: three warm events but only two warm positives. Dropped.
        (4, 10, 5, 100),
        (4, 20, 2, 200),
        (4, 30, 5, 300),
        (4, 40, 5, 400),
        # User 5: filler so the catalog has 10 movies (qcut needs >= 10). Dropped.
        (5, 80, 1, 1),
        (5, 90, 1, 2),
        (5, 100, 1, 3),
        # User 6: cold positive sits between the two train targets, so it is before cutoff.
        (6, 10, 5, 100),
        (6, 40, 5, 200),
        (6, 20, 5, 300),
        (6, 30, 5, 400),
        (6, 60, 5, 500),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def synthetic_dataset():
    return build_dataset(synthetic_ratings(), COLD_MOVIES, cold_ratio=0.1)


def test_dataset_cache_requires_matching_fingerprint(tmp_path):
    path = tmp_path / "dataset.pkl"
    dataset = synthetic_dataset()
    fingerprint = dataset_cache_fingerprint()

    _write_dataset_cache(path, dataset, fingerprint)
    loaded = _load_dataset_cache(path, fingerprint)
    assert loaded is not None
    assert loaded.full_to_movie.tolist() == dataset.full_to_movie.tolist()
    assert _load_dataset_cache(path, "wrong-fingerprint") is None

    # Legacy bare Dataset caches must be rebuilt rather than silently reused.
    with path.open("wb") as fh:
        pickle.dump(dataset, fh)
    assert _load_dataset_cache(path, fingerprint) is None


def _movies(split: pd.DataFrame, user_id: int) -> list[int]:
    rows = split[split["user_id"] == user_id].sort_values("target_pos")
    return rows["target_movie"].astype(int).tolist()


def test_cold_items_absent_from_history_train_and_warm_index():
    ds = synthetic_dataset()
    assert set(ds.full_to_movie[ds.cold_mask].tolist()) == {40, 50}
    assert np.all(ds.full_to_warm[ds.cold_mask] == -1)
    warm_movies = set(ds.full_to_movie[ds.warm_to_full].tolist())
    assert 40 not in warm_movies and 50 not in warm_movies
    for items in ds.warm_items.values():
        assert not ds.cold_mask[items].any()
    for split in (ds.train, ds.val, ds.test):
        assert not ds.cold_mask[split["target_full"].to_numpy()].any()
    for uid, hend in zip(ds.train["user_id"], ds.train["hist_end"]):
        hist = ds.warm_items[int(uid)][: int(hend)]
        assert not ds.cold_mask[hist].any()


def test_cold_test_is_after_train_cutoff():
    ds = synthetic_dataset()
    assert _movies(ds.cold_test, 1) == [40]
    assert ds.cold_test[ds.cold_test["user_id"] == 6].empty
    assert ds.cold_test[ds.cold_test["user_id"] == 3].empty
    for row in ds.cold_test.itertuples(index=False):
        assert row.target_pos > ds.train_cutoff[int(row.user_id)]
    user1 = ds.cold_test[ds.cold_test["user_id"] == 1].iloc[0]
    assert int(user1.target_pos) == 3
    assert ds.train_cutoff[1] == 2


def test_history_has_no_future_and_hist_end_matches():
    ds = synthetic_dataset()
    # User 1 sorted positions: 10@0, 20@1, 30@2, 40@3, 50@4, 60@5, 70@6.
    assert ds.all_items_full[1].tolist() == [
        ds.movie_to_full[m] for m in (10, 20, 30, 40, 50, 60, 70)
    ]
    expected_hist = {20: 1, 30: 2, 60: 3, 70: 4, 40: 3}
    frames = pd.concat([ds.train, ds.val, ds.test, ds.cold_test], ignore_index=True)
    user1 = frames[frames["user_id"] == 1]
    for row in user1.itertuples(index=False):
        wpos = ds.warm_pos[1]
        assert row.hist_end == expected_hist[int(row.target_movie)]
        assert row.hist_end == int(np.searchsorted(wpos, row.target_pos, side="left"))
        assert np.all(wpos[: row.hist_end] < row.target_pos)
        hist_movies = ds.full_to_movie[ds.warm_items[1][: row.hist_end]]
        assert 40 not in hist_movies and 50 not in hist_movies
    # Val target is in the test history; the cold event between them is not.
    test_row = ds.test[ds.test["user_id"] == 6].iloc[0]
    hist = ds.full_to_movie[ds.warm_items[6][: int(test_row.hist_end)]]
    assert list(hist) == [10, 20, 30]


def test_timestamp_ties_order_by_movie_id():
    ds = synthetic_dataset()
    assert list(ds.full_to_movie[ds.all_items_full[1][:3]]) == [10, 20, 30]


def test_users_with_fewer_than_three_warm_positives_are_dropped():
    ds = synthetic_dataset()
    kept = set(ds.train_cutoff)
    assert kept == {1, 3, 6}
    for dropped in (2, 4, 5):
        assert dropped not in kept
        for split in (ds.train, ds.val, ds.test, ds.cold_test):
            assert split[split["user_id"] == dropped].empty
    assert _movies(ds.train, 1) == [20, 30]
    assert _movies(ds.val, 1) == [60]
    assert _movies(ds.test, 1) == [70]
    assert _movies(ds.train, 3) == [10]
    assert _movies(ds.val, 3) == [20]
    assert _movies(ds.test, 3) == [30]


def test_select_history_recent_k_budget_and_callers_agree():
    items = np.array([0, 1, 2, 3, 4])
    ratings = np.array([5, 4, 3, 5, 1])
    got_items, got_ratings = select_history(items, ratings, k=3)
    assert got_items.tolist() == [2, 3, 4]
    assert got_ratings.tolist() == [3, 5, 1]

    cost = np.array([3, 3, 5, 100, 4])
    trimmed_items, trimmed_ratings = select_history(items, ratings, k=4, event_cost=cost, budget=20)
    assert trimmed_items.tolist() == [4]
    assert trimmed_ratings.tolist() == [1]

    # A partial truncate of the oldest event would fit; the whole event is dropped instead.
    whole_items, whole_ratings = select_history(
        np.array([0, 1]), np.array([5, 4]), k=2, event_cost=np.array([50, 10]), budget=20
    )
    assert whole_items.tolist() == [1]
    assert whole_ratings.tolist() == [4]
    empty_items, empty_ratings = select_history(
        np.array([0, 1]), np.array([5, 4]), k=2, event_cost=np.array([50, 10]), budget=9
    )
    assert empty_items.tolist() == []
    assert empty_ratings.tolist() == []

    caller_a = select_history(items, ratings, k=3, event_cost=cost, budget=1000)
    caller_b = select_history(items, ratings, k=3, event_cost=cost, budget=1000)
    assert np.array_equal(caller_a[0], caller_b[0])
    assert np.array_equal(caller_a[1], caller_b[1])


def test_nested_train_subsets_are_deterministic():
    train = pd.DataFrame({"user_id": np.arange(100), "target_movie": np.arange(1000, 1100)})
    first = sample_train_subsets(train, (10, 30, 100), seed=0)
    second = sample_train_subsets(train, (10, 30, 100), seed=0)
    for n in (10, 30, 100):
        pd.testing.assert_frame_equal(first[n], second[n])
    pd.testing.assert_frame_equal(first[30].iloc[:10].reset_index(drop=True), first[10])
    pd.testing.assert_frame_equal(first[100].iloc[:30].reset_index(drop=True), first[30])
    with pytest.raises(ValueError):
        sample_train_subsets(train, (101,), seed=0)


def test_stratified_cold_selection_is_deterministic_per_decile():
    # Equal counts, index descending: decile membership depends on sorting by movie_id.
    counts = pd.Series(np.ones(20, dtype=np.int64), index=np.arange(19, -1, -1))
    ratio = 0.5
    cold = select_cold_items(counts, ratio, seed=0)
    assert np.array_equal(cold, select_cold_items(counts, ratio, seed=0))
    deciles = popularity_deciles(counts)
    for decile in range(10):
        members = set(deciles.index[deciles.to_numpy() == decile].astype(int).tolist())
        assert members == {2 * decile, 2 * decile + 1}
        assert len(set(cold.tolist()) & members) == round(ratio * len(members))
    assert len(cold) == len(set(cold.tolist()))


def test_index_mappings_round_trip():
    ds = synthetic_dataset()
    for movie, full in ds.movie_to_full.items():
        assert int(ds.full_to_movie[full]) == movie
    for full, movie in enumerate(ds.full_to_movie):
        assert ds.movie_to_full[int(movie)] == full
    for warm, full in enumerate(ds.warm_to_full):
        assert ds.full_to_warm[full] == warm
        assert not ds.cold_mask[full]
    assert list(ds.full_to_movie) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert ds.full_to_warm.tolist() == [0, 1, 2, -1, -1, 3, 4, 5, 6, 7]


def test_ranks_match_hand_computed_values_including_ties():
    scores = np.array(
        [
            [1.0, 3.0, 3.0, 2.0],
            [5.0, 4.0, 4.0, 0.0],
        ]
    )
    targets = np.array([1, 0])
    mask = np.zeros_like(scores, dtype=bool)
    rank = ranks(scores, targets, mask)
    # Row 0: target score 3 ties one other unmasked item → rank 2.
    # Row 1: unique best → rank 1.
    assert rank.tolist() == [2, 1]
    hit, ndcg, mrr = metrics_from_ranks(rank)
    assert hit.tolist() == [1.0, 1.0]
    assert ndcg[0] == pytest.approx(1.0 / np.log2(3.0))
    assert ndcg[1] == pytest.approx(1.0)
    assert mrr[0] == pytest.approx(0.5)
    assert mrr[1] == pytest.approx(1.0)

    constant = ranks(np.ones((1, 4)), np.array([2]), np.zeros((1, 4), dtype=bool))
    assert int(constant[0]) == 4

    masked_tie = np.array([[False, False, True, False]])
    assert int(ranks(scores[:1], np.array([1]), masked_tie)[0]) == 1

    wide = np.arange(12, dtype=np.float64)[None, :]
    wide_rank = ranks(wide, np.array([0]), np.zeros_like(wide, dtype=bool))
    assert int(wide_rank[0]) == 12
    hit12, ndcg12, mrr12 = metrics_from_ranks(wide_rank)
    assert hit12[0] == 0.0
    assert ndcg12[0] == 0.0
    assert mrr12[0] == pytest.approx(1.0 / 12.0)


def test_ranks_reject_nonfinite_unmasked_scores_and_masked_targets():
    ok = np.array([[1.0, np.nan]])
    ranks(ok, np.array([0]), np.array([[False, True]]))
    with pytest.raises(ValueError):
        ranks(np.array([[1.0, np.nan]]), np.array([0]), np.zeros((1, 2), dtype=bool))
    with pytest.raises(ValueError):
        ranks(np.array([[1.0, np.inf]]), np.array([0]), np.zeros((1, 2), dtype=bool))
    with pytest.raises(ValueError):
        ranks(np.array([[1.0, -np.inf]]), np.array([1]), np.zeros((1, 2), dtype=bool))
    with pytest.raises(ValueError):
        ranks(np.array([[1.0, 2.0]]), np.array([0]), np.array([[True, False]]))


def test_candidate_mask_excludes_earlier_items_and_keeps_target():
    ds = synthetic_dataset()
    test_row = ds.test[ds.test["user_id"] == 3].iloc[0]
    mask = candidate_mask(ds, test_row, warm_only=False)
    earlier = {10, 20, 40}
    for movie in ds.full_to_movie:
        full = ds.movie_to_full[int(movie)]
        assert bool(mask[full]) is (int(movie) in earlier)
    assert not mask[int(test_row.target_full)]

    warm_only = candidate_mask(ds, test_row, warm_only=True)
    assert not warm_only[int(test_row.target_full)]
    assert warm_only[ds.cold_mask].all()
    # Movie 50 is cold and unseen by user 3; only warm-only excludes it.
    assert not mask[ds.movie_to_full[50]]
    assert warm_only[ds.movie_to_full[50]]

    cold_row = ds.cold_test[ds.cold_test["user_id"] == 1].iloc[0]
    cold_mask = candidate_mask(ds, cold_row)
    assert not cold_mask[int(cold_row.target_full)]
    assert cold_mask[ds.movie_to_full[10]]
    assert cold_mask[ds.movie_to_full[20]]
    assert cold_mask[ds.movie_to_full[30]]


def test_paired_bootstrap_identical_users_and_mismatch():
    a = pd.Series([1.0, 2.0, 4.0], index=[10, 20, 30])
    b = pd.Series([4.0, 1.0, 2.0], index=[30, 10, 20])
    mean, lo, hi = paired_bootstrap(a, b, n=40, seed=0)
    assert mean == 0.0
    assert lo == 0.0 and hi == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap(a, pd.Series([1.0, 2.0], index=[10, 20]), n=10, seed=0)
    with pytest.raises(ValueError):
        paired_bootstrap(a, pd.Series([1.0, 2.0, 4.0], index=[10, 20, 99]), n=10, seed=0)


def test_cold_ratio_rule(monkeypatch):
    ratings = synthetic_ratings()
    ds, info = choose_cold_ratio(ratings)
    assert ds.cold_ratio == COLD_RATIO_FALLBACK
    assert [row["ratio"] for row in info["attempts"]] == [COLD_RATIO_DEFAULT, COLD_RATIO_FALLBACK]
    assert ds.cold_attempts == info["attempts"]
    first = info["attempts"][0]
    assert first["n_users"] < COLD_MIN_USERS or first["n_interactions"] < COLD_MIN_INTERACTIONS

    monkeypatch.setattr("minigenrec.data.COLD_MIN_USERS", 0)
    monkeypatch.setattr("minigenrec.data.COLD_MIN_INTERACTIONS", 0)
    kept, kept_info = choose_cold_ratio(ratings)
    assert kept.cold_ratio == COLD_RATIO_DEFAULT
    assert len(kept_info["attempts"]) == 1


def _assert_split_history(ds, split: pd.DataFrame) -> None:
    for uid, grp in split.groupby("user_id"):
        wpos = ds.warm_pos[int(uid)]
        assert len(wpos) == 0 or bool(np.all(np.diff(wpos) > 0))
        ends = grp["hist_end"].to_numpy()
        tpos = grp["target_pos"].to_numpy()
        assert np.all((ends >= 0) & (ends <= len(wpos)))
        kept = ends > 0
        # Last event actually included is strictly before the target.
        assert np.all(wpos[ends[kept] - 1] < tpos[kept])
        # Next warm event is not before the target, so nothing future slipped in.
        nxt = ends < len(wpos)
        assert np.all(wpos[ends[nxt]] >= tpos[nxt])
        hist_items = ds.warm_items[int(uid)]
        if len(hist_items):
            assert not ds.cold_mask[hist_items].any()


def _assert_real_invariants(ds) -> None:
    assert len(ds.full_to_movie) == 3706
    assert np.all(ds.full_to_warm[ds.cold_mask] == -1)
    assert not ds.cold_mask[ds.train["target_full"].to_numpy()].any()
    assert not ds.cold_mask[ds.val["target_full"].to_numpy()].any()
    assert not ds.cold_mask[ds.test["target_full"].to_numpy()].any()
    for items in ds.warm_items.values():
        if len(items):
            assert not ds.cold_mask[items].any()
    assert ds.cold_mask[ds.cold_test["target_full"].to_numpy()].all()
    cut = ds.cold_test["user_id"].map(ds.train_cutoff).to_numpy()
    assert np.all(ds.cold_test["target_pos"].to_numpy() > cut)
    for split in (ds.train, ds.val, ds.test, ds.cold_test):
        _assert_split_history(ds, split)


@pytest.mark.skipif(not (DATA_DIR / "ml-1m").exists(), reason="ml-1m not downloaded")
def test_real_ml1m_leakage_invariants():
    ratings, movies = load_ml1m()
    toy = movies.loc[movies["movie_id"] == 1].iloc[0]
    assert toy["title"] == "Toy Story"
    assert int(toy["year"]) == 1995
    assert not movies["title"].str.endswith(" ").any()
    associate = movies.loc[movies["title"].str.startswith("Associate, The")].iloc[0]
    assert associate["title"] == "Associate, The (L'Associe)"
    assert int(associate["year"]) == 1982
    ds, _info = choose_cold_ratio(ratings, movies)
    _assert_real_invariants(ds)
    rng = np.random.default_rng(0)
    for split in (ds.train, ds.val, ds.test, ds.cold_test):
        take = min(30, len(split))
        picked = rng.choice(len(split), size=take, replace=False)
        for i in picked:
            row = split.iloc[int(i)]
            mask = candidate_mask(ds, row, warm_only=False)
            assert not mask[int(row["target_full"])]
    for i in rng.choice(len(ds.test), size=20, replace=False):
        row = ds.test.iloc[int(i)]
        warm_only = candidate_mask(ds, row, warm_only=True)
        assert not warm_only[int(row["target_full"])]
        assert warm_only[ds.cold_mask].all()
