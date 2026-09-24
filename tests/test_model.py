"""Model invariants: cold gating, softplus λ, finite scores."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from minigenrec.batching import HistorySelector, collate_history, warm_ce_loss
from minigenrec.model import ItemEncoder, ScoreHead, softplus_inv
from minigenrec.sasrec import SASRecEncoder, SASRecRanker


def test_softplus_inv_gives_lambda_point_one():
    raw = softplus_inv(0.1)
    assert abs(raw - math.log(math.expm1(0.1))) < 1e-9
    assert abs(float(F.softplus(torch.tensor(raw))) - 0.1) < 1e-5


def test_cold_items_exclude_id_from_hybrid_embedding():
    n, d, td = 5, 8, 16
    cold = torch.tensor([False, False, True, False, True])
    enc = ItemEncoder(n, d, mode="hybrid", text_dim=td, cold_mask=cold, init_lambda=0.5, id_dropout=0.0)
    text = torch.randn(n, td)
    with torch.no_grad():
        # force large id so any leak is obvious
        enc.id_emb.weight.copy_(torch.randn(n, d) * 10)
        e = enc(text)
        text_only = F.normalize(enc.w_text(text), dim=-1)
    # cold rows equal pure text (m_i=0)
    assert torch.allclose(e[2], text_only[2], atol=1e-5)
    assert torch.allclose(e[4], text_only[4], atol=1e-5)
    assert not torch.allclose(e[0], text_only[0], atol=1e-3)


def test_cold_id_embedding_gets_no_gradient_via_hybrid():
    n, d, td = 4, 8, 16
    cold = torch.tensor([False, True, False, True])
    enc = ItemEncoder(n, d, mode="hybrid", text_dim=td, cold_mask=cold, init_lambda=0.5, id_dropout=0.0)
    text = torch.randn(n, td)
    e = enc(text)
    loss = e.sum()
    loss.backward()
    # warm ids may have grad; cold ids must be ~0 because m_i=0 blocks the path
    assert enc.id_emb.weight.grad is not None
    assert float(enc.id_emb.weight.grad[1].abs().sum()) == 0.0
    assert float(enc.id_emb.weight.grad[3].abs().sum()) == 0.0
    assert float(enc.id_emb.weight.grad[0].abs().sum()) > 0.0


def test_id_dropout_only_in_training_and_only_warm():
    n, d, td = 400, 8, 16
    cold = torch.zeros(n, dtype=torch.bool)
    cold[:100] = True
    enc = ItemEncoder(n, d, mode="hybrid", text_dim=td, cold_mask=cold, init_lambda=0.5, id_dropout=0.5)
    text = torch.randn(n, td)
    with torch.no_grad():
        enc.id_emb.weight.copy_(torch.randn(n, d) * 10)
        text_only = F.normalize(enc.w_text(text), dim=-1)
        torch.manual_seed(0)
        train_e = enc.train()(text)
        eval_e = enc.eval()(text)
    as_text = torch.isclose(train_e, text_only, atol=1e-5).all(dim=1)
    assert as_text[:100].all()  # cold always text-only
    dropped = as_text[100:].float().mean().item()
    assert 0.3 < dropped < 0.7  # ~half the warm items lose their ID part
    assert not torch.isclose(eval_e[100:], text_only[100:], atol=1e-5).all(dim=1).any()


def test_evaluate_cold_only_and_cold_intrusion():
    from minigenrec.evaluation import evaluate

    # 4 items: 0,1 warm; 2,3 cold. Scorer loves cold items.
    dataset = SimpleNamespace(
        full_to_movie=np.arange(4),
        cold_mask=np.array([False, False, True, True]),
        all_items_full={1: np.array([], dtype=np.int64)},
        all_pos={1: np.array([], dtype=np.int64)},
        pop_buckets=np.array(["head"] * 4, dtype=object),
    )
    rows = pd.DataFrame([{"user_id": 1, "target_full": 3, "target_movie": 3, "target_pos": 0}])
    scores = np.array([[5.0, 4.0, 9.0, 1.0]])

    import minigenrec.evaluation as ev

    old_k = ev.TOP_K
    ev.TOP_K = 2
    try:
        full = evaluate(dataset, rows, lambda b: scores, batch_size=8)
        cold_only = evaluate(dataset, rows, lambda b: scores, batch_size=8, cold_only=True)
    finally:
        ev.TOP_K = old_k
    assert int(full["rank"][0]) == 4  # behind 2, 0, 1
    assert int(cold_only["rank"][0]) == 2  # only competes with cold item 2
    assert float(full["cold_topk"][0]) == 0.5  # top-2 = {2, 0}


def test_scores_finite_and_target_unmasked_path():
    n, d = 10, 8
    enc = ItemEncoder(n, d, mode="id")
    head = ScoreHead(d, d)
    sas = SASRecEncoder(d, max_len=4)
    model = SASRecRanker(enc, head, sas)
    item_ids = torch.tensor([[1, 2, 3]])
    ratings = torch.tensor([[5, 4, 3]])
    pad = torch.tensor([[False, False, False]])
    logits = model(item_ids, ratings, pad)
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, n)


def test_score_head_does_not_hide_nonfinite_inputs():
    head = ScoreHead(2, 2)
    h = torch.tensor([[float("nan"), 0.0]])
    item_emb = torch.eye(2)
    scores = head(h, item_emb)
    assert not torch.isfinite(scores).all()


def test_training_loss_fails_fast_on_nonfinite_scores():
    logits = torch.tensor([[float("nan"), 0.0]])
    targets = torch.tensor([0])
    warm_to_full = torch.tensor([0, 1])
    full_to_warm = torch.tensor([0, 1])

    with pytest.raises(FloatingPointError, match="non-finite"):
        warm_ce_loss(logits, targets, warm_to_full, full_to_warm)


def test_marker_pool_uses_last_nonpad():
    # ScoreHead path not needed — just ensure lengths logic in SASRec
    sas = SASRecEncoder(8, max_len=5)
    x = torch.randn(2, 4, 8)
    ratings = torch.ones(2, 4, dtype=torch.long) * 5
    pad = torch.tensor([[False, False, False, True], [False, False, True, True]])
    h = sas(x, ratings, pad)
    assert h.shape == (2, 8)
    assert torch.isfinite(h).all()


def _variable_length_batch():
    dataset = SimpleNamespace(
        warm_items={
            1: np.array([1, 2], dtype=np.int64),
            2: np.array([3, 4, 5, 6], dtype=np.int64),
        },
        warm_ratings={
            1: np.array([4, 5], dtype=np.int64),
            2: np.array([5, 4, 3, 5], dtype=np.int64),
        },
    )
    batch = pd.DataFrame(
        [
            {"user_id": 1, "hist_end": 2, "target_full": 7},
            {"user_id": 2, "hist_end": 4, "target_full": 8},
        ]
    )
    return dataset, batch


def test_collate_history_right_pads():
    dataset, batch = _variable_length_batch()
    item_ids, ratings, pad_mask, _ = collate_history(dataset, batch, k=4)

    assert item_ids.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6]]
    assert ratings.tolist() == [[4, 5, 0, 0], [5, 4, 3, 5]]
    assert pad_mask.tolist() == [[False, False, True, True], [False, False, False, False]]


def test_sasrec_output_is_invariant_to_neighbor_history_length():
    dataset, batch = _variable_length_batch()
    mixed_ids, mixed_ratings, mixed_pad, _ = collate_history(dataset, batch, k=4)
    single_ids, single_ratings, single_pad, _ = collate_history(dataset, batch.iloc[:1], k=4)

    torch.manual_seed(0)
    dim = 8
    sas = SASRecEncoder(dim, max_len=4, dropout=0.0).eval()
    catalog = torch.randn(10, dim)
    with torch.no_grad():
        mixed_h = sas(catalog[mixed_ids], mixed_ratings, mixed_pad)[0]
        single_h = sas(catalog[single_ids], single_ratings, single_pad)[0]

    assert torch.allclose(mixed_h, single_h, atol=1e-6)


class _LenTok:
    """encode length == number of characters (deterministic stand-in for HF)."""

    def __init__(self):
        self.calls = 0

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        extra = 2 if add_special_tokens else 0
        return list(range(len(text) + extra))


def test_sasrec_and_llm_share_token_budget_events():
    from minigenrec.batching import history_for_row, prompts_from_history
    from types import SimpleNamespace

    # Long titles force budget drops when max_len is tiny.
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["A" * 40, "B" * 40, "C" * 40],
            "year": [1999, 2000, 2001],
            "genres": ["Drama", "Comedy", "Action"],
        }
    )
    dataset = SimpleNamespace(
        warm_items={1: np.array([0, 1, 2], dtype=np.int64)},
        warm_ratings={1: np.array([5, 4, 3], dtype=np.int64)},
        full_to_movie=np.array([10, 20, 30]),
        movies=movies,
    )
    row = pd.Series({"user_id": 1, "hist_end": 3, "target_full": 0})
    tok = _LenTok()
    selector = HistorySelector(dataset, tok, k=3, max_len=80)
    items, ratings = history_for_row(dataset, row, selector=selector)
    # Prompt with 3 long events exceeds 80; drops oldest until fit.
    assert len(items) < 3
    calls_after_first_selection = tok.calls
    # Selection is memoized and does not depend on the active title ablation.
    assert history_for_row(dataset, row, selector=selector)[0].tolist() == items.tolist()
    assert tok.calls == calls_after_first_selection
    collated, _, pad, _ = collate_history(
        dataset,
        pd.DataFrame([row]),
        k=3,
        selector=selector,
    )
    n = int((~pad[0]).sum())
    assert collated[0, :n].tolist() == items.tolist()
    for title_map in selector.title_maps.values():
        prompt = prompts_from_history(dataset, items, ratings, title_map)
        assert len(tok.encode(prompt, add_special_tokens=True)) <= 80


def test_long_history_budget_search_is_not_quadratic():
    from minigenrec.verbalize import build_prompt, fit_events_to_budget

    tok = _LenTok()
    lines = ["event-" + ("x" * 20)] * 2_000
    kept = fit_events_to_budget(lines, tok, max_len=512)

    assert len(tok.encode(build_prompt(kept), add_special_tokens=True)) <= 512
    # One full check plus a binary search and the assertion above.
    assert tok.calls < 20


def test_report_aggregates_seeds(tmp_path):
    import json as _json

    from report import seed_table, summary_markdown

    for seed, hit in ((0, 0.10), (1, 0.20)):
        run = tmp_path / f"sasrec_id_n100_seed{seed}"
        run.mkdir()
        seg = {"hit": {"mean": hit}}
        payload = {
            "segments": {
                "warm_test_full": {**seg, "cold_intrusion": 0.0},
                "warm_test_warm_only": seg,
                "cold_test": seg,
            }
        }
        (run / "metrics.json").write_text(_json.dumps(payload))
    df = seed_table(tmp_path, 100)
    assert len(df) == 2 and set(df["model"]) == {"sasrec_id"}
    md = summary_markdown(df)
    assert "| sasrec_id | 2 |" in md
    assert "0.1500 ± 0.0707" in md
    assert "—" in md  # old runs have no cold-only segment


def test_text_embedding_cache_key_fingerprints_content_and_order():
    from minigenrec.verbalize import cache_key

    common = ("model", "revision", "real")
    first = cache_key(*common, ["movie-a", "movie-b"])
    assert first == cache_key(*common, ["movie-a", "movie-b"])
    assert first != cache_key(*common, ["movie-b", "movie-a"])
    assert first != cache_key(*common, ["movie-a", "movie-c"])


def test_trainable_checkpoint_excludes_frozen_parameters():
    from minigenrec.train_llm import trainable_state_dict

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    state = trainable_state_dict(model)

    assert set(state) == {"1.weight", "1.bias"}

