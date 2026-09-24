"""Model invariants: cold gating, softplus λ, finite scores."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from minigenrec.model import ItemEncoder, ScoreHead, softplus_inv
from minigenrec.sasrec import SASRecEncoder, SASRecRanker


def test_softplus_inv_gives_lambda_point_one():
    raw = softplus_inv(0.1)
    assert abs(raw - math.log(math.expm1(0.1))) < 1e-9
    assert abs(float(F.softplus(torch.tensor(raw))) - 0.1) < 1e-5


def test_cold_items_exclude_id_from_hybrid_embedding():
    n, d, td = 5, 8, 16
    cold = torch.tensor([False, False, True, False, True])
    enc = ItemEncoder(n, d, mode="hybrid", text_dim=td, cold_mask=cold, init_lambda=0.5)
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
    enc = ItemEncoder(n, d, mode="hybrid", text_dim=td, cold_mask=cold, init_lambda=0.5)
    text = torch.randn(n, td)
    e = enc(text)
    loss = e.sum()
    loss.backward()
    # warm ids may have grad; cold ids must be ~0 because m_i=0 blocks the path
    assert enc.id_emb.weight.grad is not None
    assert float(enc.id_emb.weight.grad[1].abs().sum()) == 0.0
    assert float(enc.id_emb.weight.grad[3].abs().sum()) == 0.0
    assert float(enc.id_emb.weight.grad[0].abs().sum()) > 0.0


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


def test_marker_pool_uses_last_nonpad():
    # ScoreHead path not needed — just ensure lengths logic in SASRec
    sas = SASRecEncoder(8, max_len=5)
    x = torch.randn(2, 4, 8)
    ratings = torch.ones(2, 4, dtype=torch.long) * 5
    pad = torch.tensor([[False, False, False, True], [False, False, True, True]])
    h = sas(x, ratings, pad)
    assert h.shape == (2, 8)
    assert torch.isfinite(h).all()
