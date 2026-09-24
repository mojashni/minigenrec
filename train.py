"""Train and evaluate. Each run trains its own weights from scratch."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from minigenrec.batching import HistorySelector, collate_history, warm_ce_loss
from minigenrec.config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    COLD_MIN_INTERACTIONS,
    COLD_MIN_USERS,
    COLD_RATIO_DEFAULT,
    COLD_RATIO_FALLBACK,
    COLD_SPLIT_SEED,
    ID_DROPOUT,
    MAX_EVENTS_K,
    MAX_LEN,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SEEDS,
    N_TRAIN_SIZES,
    POSITIVE_MIN_RATING,
    RESULTS_DIR,
    TITLE_SHUFFLE_SEED,
    TOP_K,
    TRAIN_SUBSET_SEED,
)
from minigenrec.data import prepare, sample_train_subsets
from minigenrec.evaluation import evaluate, mean_hit
from minigenrec.metrics import bootstrap_ci, user_level
from minigenrec.model import ItemEncoder, ScoreHead
from minigenrec.sasrec import SASRecEncoder, SASRecRanker

BATCH_SIZE = 256
EVAL_BATCH = 512
DIM = 64
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # SASRec Mac: CPU; RunPod: CUDA
# MPS was flaky after process kills; don't use it for long trains.


def best_state_by_val(model, dataset, score_fn, batch_size: int = EVAL_BATCH):
    """Return (hit, state_dict copy) for current model on warm val."""
    hit = mean_hit(dataset, dataset.val, score_fn, batch_size)
    return hit, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class FrameDataset(TorchDataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> int:
        return idx


def popularity_scores(n_full: int, train_subset: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(n_full, dtype=np.float64)
    scores += np.bincount(train_subset["target_full"].to_numpy(), minlength=n_full)
    return scores


def _ci_block(per_user: pd.DataFrame) -> dict:
    block = {"n_users": int(len(per_user)), "n_rows": None}
    for name in ("hit", "ndcg", "mrr"):
        mean, lo, hi = bootstrap_ci(per_user[name], BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
        block[name] = {"mean": mean, "lo": lo, "hi": hi}
    return block


def summarize_segment(per_example: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if len(per_example) == 0:
        empty = {"mean": None, "lo": None, "hi": None}
        return {"n_users": 0, "n_rows": 0, "hit": empty, "ndcg": empty, "mrr": empty}, per_example
    per_user = user_level(per_example)
    summary = _ci_block(per_user)
    summary["n_rows"] = int(len(per_example))
    return summary, per_user


def write_results(run_name: str, payload: dict, per_user: dict) -> None:
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, users in per_user.items():
        users.reset_index().to_csv(run_dir / f"per_user_{name}.csv", index=False)
    print(run_dir / "metrics.json")


def base_payload(args, dataset) -> dict:
    return {
        "model": args.model,
        "items": getattr(args, "items", None),
        "n_train": args.n_train,
        "seeds": {
            "cold_split_seed": COLD_SPLIT_SEED,
            "train_subset_seed": TRAIN_SUBSET_SEED,
            "title_shuffle_seed": TITLE_SHUFFLE_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "model_seed": args.model_seed,
            "model_seeds": list(MODEL_SEEDS),
        },
        "config": {
            "positive_min_rating": POSITIVE_MIN_RATING,
            "cold_ratio_default": COLD_RATIO_DEFAULT,
            "cold_ratio_fallback": COLD_RATIO_FALLBACK,
            "cold_min_users": COLD_MIN_USERS,
            "cold_min_interactions": COLD_MIN_INTERACTIONS,
            "chosen_cold_ratio": dataset.cold_ratio,
            "n_train_sizes": list(N_TRAIN_SIZES),
            "max_events_k": MAX_EVENTS_K,
            "max_len": MAX_LEN,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "top_k": TOP_K,
            "id_dropout": ID_DROPOUT,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "device": DEVICE,
        },
    }


def eval_all(dataset, score_fn, batch_size: int = EVAL_BATCH) -> tuple[dict, dict]:
    segments = {
        "warm_test_full": evaluate(dataset, dataset.test, score_fn, batch_size, warm_only=False),
        "warm_test_warm_only": evaluate(dataset, dataset.test, score_fn, batch_size, warm_only=True),
        "cold_test": evaluate(dataset, dataset.cold_test, score_fn, batch_size, warm_only=False),
        # Target ranked among cold items only: rewards picking the right cold
        # movie, not boosting every cold item as a group.
        "cold_test_cold_only": evaluate(dataset, dataset.cold_test, score_fn, batch_size, cold_only=True),
    }
    payload_segments = {}
    per_user = {}
    for name, frame in segments.items():
        summary, users = summarize_segment(frame)
        payload_segments[name] = summary
        per_user[name] = users
    # Share of warm-test top-K slots taken by cold items; high = cold flooding.
    warm_full = segments["warm_test_full"]
    payload_segments["warm_test_full"]["cold_intrusion"] = (
        float(warm_full["cold_topk"].mean()) if len(warm_full) else None
    )
    cold = segments["cold_test"]
    by_bucket = {}
    for bucket in ("head", "mid", "tail"):
        summary, _ = summarize_segment(cold[cold["bucket"] == bucket])
        by_bucket[bucket] = summary
    payload_segments["cold_test"]["by_bucket"] = by_bucket
    return payload_segments, per_user


def run_popular(args, dataset, train_subset) -> None:
    scores = popularity_scores(len(dataset.full_to_movie), train_subset)

    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        return np.repeat(scores[None, :], len(batch), axis=0)

    payload = base_payload(args, dataset)
    payload["run_name"] = args.run_name or f"popular_n{args.n_train}_seed{args.model_seed}"
    segs, per_user = eval_all(dataset, score_fn)
    payload["segments"] = segs
    write_results(payload["run_name"], payload, per_user)


def build_sasrec(
    dataset,
    items_mode: str,
    seed: int,
    text_dim: int | None = None,
    max_events: int = MAX_EVENTS_K,
) -> SASRecRanker:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_items = len(dataset.full_to_movie)
    cold = torch.from_numpy(dataset.cold_mask)
    item_enc = ItemEncoder(n_items, DIM, mode=items_mode, text_dim=text_dim, cold_mask=cold)
    head = ScoreHead(DIM, DIM)
    sas = SASRecEncoder(DIM, max_len=max_events)
    return SASRecRanker(item_enc, head, sas)


def train_sasrec(args, dataset, train_subset, text_emb: torch.Tensor | None = None) -> SASRecRanker:
    from minigenrec.llm import load_tokenizer
    text_dim = None if text_emb is None else int(text_emb.shape[1])
    model_max_events = MAX_LEN if args.full_history else MAX_EVENTS_K
    model = build_sasrec(
        dataset,
        args.items,
        args.model_seed,
        text_dim=text_dim,
        max_events=model_max_events,
    ).to(DEVICE)
    warm_to_full = torch.from_numpy(dataset.warm_to_full).to(DEVICE)
    full_to_warm = torch.from_numpy(dataset.full_to_warm).to(DEVICE)
    text = None if text_emb is None else text_emb.to(DEVICE)
    # Same token-budget event list as LLM (even for --items id).
    tokenizer = load_tokenizer(MODEL_ID, MODEL_REVISION)
    selector = HistorySelector(dataset, tokenizer, full_history=args.full_history)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    frame_ds = FrameDataset(train_subset)

    def collate(idxs):
        batch = train_subset.iloc[list(idxs)]
        item_ids, ratings, pad_mask, targets = collate_history(
            dataset,
            batch,
            k=MAX_EVENTS_K,
            full_history=args.full_history,
            selector=selector,
        )
        return item_ids, ratings, pad_mask, targets

    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        return sasrec_score_fn(
            model, dataset, batch, args.full_history, text_emb=text_emb, selector=selector
        )

    loader = DataLoader(frame_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    model.train()
    t0 = time.time()
    best_hit, best_epoch, best_state = -1.0, 0, None
    for epoch in range(EPOCHS):
        total = 0.0
        n = 0
        model.train()
        for item_ids, ratings, pad_mask, targets in loader:
            item_ids = item_ids.to(DEVICE)
            ratings = ratings.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            targets = targets.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(item_ids, ratings, pad_mask, text_emb=text)
            loss = warm_ce_loss(logits, targets, warm_to_full, full_to_warm)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(targets)
            n += len(targets)
        hit, state = best_state_by_val(model, dataset, score_fn)
        if hit > best_hit:
            best_hit, best_epoch, best_state = hit, epoch + 1, state
        print(f"epoch {epoch+1}/{EPOCHS} loss={total/max(n,1):.4f} val_hit={hit:.4f} ({time.time()-t0:.0f}s)")
    assert best_state is not None
    model.load_state_dict(best_state)
    model._best_epoch = best_epoch  # type: ignore[attr-defined]
    model._best_val_hit = best_hit  # type: ignore[attr-defined]
    model._history_selector = selector  # type: ignore[attr-defined]
    print(f"restored epoch={best_epoch} val_hit={best_hit:.4f}")
    return model


@torch.no_grad()
def sasrec_score_fn(
    model: SASRecRanker,
    dataset,
    batch: pd.DataFrame,
    full_history: bool,
    text_emb: torch.Tensor | None = None,
    selector: HistorySelector | None = None,
) -> np.ndarray:
    model.eval()
    item_ids, ratings, pad_mask, _ = collate_history(
        dataset,
        batch,
        k=MAX_EVENTS_K,
        full_history=full_history,
        selector=selector,
    )
    text = None if text_emb is None else text_emb.to(DEVICE)
    logits = model(item_ids.to(DEVICE), ratings.to(DEVICE), pad_mask.to(DEVICE), text_emb=text)
    return logits.float().cpu().numpy()


def run_sasrec(args, dataset, train_subset) -> None:
    text_emb = None
    if args.items in ("text", "hybrid"):
        from minigenrec.text_emb import load_text_emb

        text_emb = load_text_emb(dataset, title_mode=args.titles)
    model = train_sasrec(args, dataset, train_subset, text_emb=text_emb)
    selector = model._history_selector  # type: ignore[attr-defined]
    payload = base_payload(args, dataset)
    payload["titles"] = args.titles
    payload["run_name"] = args.run_name or f"sasrec_{args.items}_n{args.n_train}_seed{args.model_seed}"
    payload["train"] = {
        "epochs": EPOCHS,
        "best_epoch": model._best_epoch,  # type: ignore[attr-defined]
        "best_val_hit": model._best_val_hit,  # type: ignore[attr-defined]
        "dim": DIM,
        "lr": LR,
        "batch_size": BATCH_SIZE,
    }
    run_dir = RESULTS_DIR / payload["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "items": args.items, "dim": DIM}, run_dir / "model.pt")

    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        return sasrec_score_fn(
            model, dataset, batch, args.full_history, text_emb=text_emb, selector=selector
        )

    segs, per_user = eval_all(dataset, score_fn)
    payload["segments"] = segs
    write_results(payload["run_name"], payload, per_user)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["popular", "sasrec", "frozen_llm", "genrec"])
    parser.add_argument("--items", default="id", choices=["id", "text", "hybrid"])
    parser.add_argument("--titles", default="real", choices=["real", "shuffled", "none"])
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-limit", type=int, default=None, help="cap eval rows per split (smoke)")
    args = parser.parse_args()

    dataset = prepare()
    if args.eval_limit:
        dataset.test = dataset.test.iloc[: args.eval_limit].copy()
        dataset.cold_test = dataset.cold_test.iloc[: args.eval_limit].copy()
        dataset.val = dataset.val.iloc[: args.eval_limit].copy()
    sizes = list(N_TRAIN_SIZES)
    if args.n_train not in sizes:
        sizes.append(args.n_train)
    train_subset = sample_train_subsets(dataset.train, sizes, TRAIN_SUBSET_SEED)[args.n_train]

    if args.model == "popular":
        run_popular(args, dataset, train_subset)
    elif args.model == "sasrec":
        run_sasrec(args, dataset, train_subset)
    elif args.model in ("frozen_llm", "genrec"):
        from minigenrec.train_llm import run_llm

        run_llm(args, dataset, train_subset)
    else:
        raise SystemExit(args.model)


if __name__ == "__main__":
    main()
