"""Frozen-LLM (cached h) and MiniGenRec (LoRA) training."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from minigenrec.batching import HistorySelector, history_for_row, prompts_from_history, warm_ce_loss
from minigenrec.config import MAX_EVENTS_K, MAX_LEN, MODEL_ID, MODEL_REVISION, RESULTS_DIR
from minigenrec.evaluation import mean_hit
from minigenrec.llm import LLMUserEncoder, load_backbone, load_tokenizer, tokenize_prompts
from minigenrec.model import ItemEncoder, ScoreHead
from minigenrec.text_emb import load_text_emb
from minigenrec.verbalize import build_title_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 64
LLM_LR = 2e-4
CACHE_BATCH = 16
HEAD_BATCH = 256
FROZEN_EPOCHS = 15
LORA_EPOCHS = 3
LORA_BATCH = 4
EVAL_BATCH_LLM = 8  # CausalLM/AutoModel prompt scoring OOMs at 512 on A100


def seed_everything(seed: int) -> None:
    """Seed model initialization and shuffled data loading for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU checkpoint containing only parameters that optimization can change."""
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    missing = trainable.difference(state)
    if missing:
        raise KeyError(f"trainable parameters missing from state_dict: {sorted(missing)[:5]}")
    return {name: state[name].detach().cpu().clone() for name in trainable}


def _title_map(dataset, mode: str) -> dict[int, str]:
    return build_title_map(dataset.movies, mode)


def prompts_for_batch(
    dataset,
    batch: pd.DataFrame,
    title_map: dict[int, str],
    tokenizer,
    selector: HistorySelector | None = None,
) -> list[str]:
    prompts = []
    for i in range(len(batch)):
        items, ratings = history_for_row(
            dataset,
            batch.iloc[i],
            k=MAX_EVENTS_K,
            selector=selector,
        )
        prompts.append(prompts_from_history(dataset, items, ratings, title_map))
    return prompts


@torch.no_grad()
def cache_user_hidden(
    dataset,
    frame: pd.DataFrame,
    title_map: dict[int, str],
    selector: HistorySelector,
    device: str = DEVICE,
) -> torch.Tensor:
    """One forward pass per example with frozen backbone. Returns [N, H] float32."""
    tokenizer = load_tokenizer(MODEL_ID, MODEL_REVISION)
    backbone = load_backbone(MODEL_ID, MODEL_REVISION, freeze=True)
    user = LLMUserEncoder(backbone, tokenizer, use_lora=False).to(device)
    user.eval()
    outs = []
    for start in range(0, len(frame), CACHE_BATCH):
        batch = frame.iloc[start : start + CACHE_BATCH]
        prompts = prompts_for_batch(dataset, batch, title_map, tokenizer, selector=selector)
        enc = tokenize_prompts(tokenizer, prompts, MAX_LEN)
        h = user(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        outs.append(h.float().cpu())
        if start % 64 == 0:
            print(f"  cached {min(start+CACHE_BATCH, len(frame))}/{len(frame)}")
    del user, backbone
    if device == "mps":
        torch.mps.empty_cache()
    return torch.cat(outs, dim=0)


class CachedRanker(torch.nn.Module):
    """Trainable head on cached user vectors + item encoder."""

    def __init__(self, hidden: int, n_items: int, text_dim: int, items_mode: str, cold_mask: torch.Tensor):
        super().__init__()
        self.item_encoder = ItemEncoder(n_items, DIM, mode=items_mode, text_dim=text_dim, cold_mask=cold_mask)
        self.score_head = ScoreHead(hidden, DIM)

    def forward(self, h: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        return self.score_head(h, self.item_encoder(text_emb))


def train_frozen(args, dataset, train_subset, text_emb: torch.Tensor) -> CachedRanker:
    seed_everything(args.model_seed)
    title_map = _title_map(dataset, args.titles)
    selector = HistorySelector(dataset, load_tokenizer(MODEL_ID, MODEL_REVISION))
    print("caching train user states...")
    h_train = cache_user_hidden(dataset, train_subset, title_map, selector)
    print("caching val user states...")
    h_val = cache_user_hidden(dataset, dataset.val, title_map, selector)
    targets = torch.from_numpy(train_subset["target_full"].to_numpy(dtype=np.int64, copy=True))
    model = CachedRanker(
        h_train.shape[1],
        len(dataset.full_to_movie),
        int(text_emb.shape[1]),
        args.items,
        torch.from_numpy(dataset.cold_mask),
    ).to(DEVICE)
    text = text_emb.to(DEVICE)
    warm_to_full = torch.from_numpy(dataset.warm_to_full).to(DEVICE)
    full_to_warm = torch.from_numpy(dataset.full_to_warm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LLM_LR, weight_decay=0.01)
    loader = DataLoader(TensorDataset(h_train, targets), batch_size=HEAD_BATCH, shuffle=True)
    t0 = time.time()
    best_hit, best_epoch, best_state = -1.0, 0, None
    val_key = list(
        zip(
            dataset.val["user_id"].to_numpy(),
            dataset.val["target_full"].to_numpy(),
            dataset.val["hist_end"].to_numpy(),
        )
    )
    key_to_i = {k: i for i, k in enumerate(val_key)}

    @torch.no_grad()
    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        model.eval()
        keys = list(zip(batch["user_id"].to_numpy(), batch["target_full"].to_numpy(), batch["hist_end"].to_numpy()))
        hb = h_val[[key_to_i[k] for k in keys]].to(DEVICE)
        return model(hb, text).float().cpu().numpy()

    for epoch in range(FROZEN_EPOCHS):
        model.train()
        total = 0.0
        n = 0
        for hb, tb in loader:
            hb, tb = hb.to(DEVICE), tb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = warm_ce_loss(model(hb, text), tb, warm_to_full, full_to_warm)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(tb)
            n += len(tb)
        hit = mean_hit(dataset, dataset.val, score_fn, EVAL_BATCH_LLM)
        if hit > best_hit:
            best_hit = hit
            best_epoch = epoch + 1
            best_state = trainable_state_dict(model)
        print(f"epoch {epoch+1}/{FROZEN_EPOCHS} loss={total/max(n,1):.4f} val_hit={hit:.4f} ({time.time()-t0:.0f}s)")
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    model._best_epoch = best_epoch  # type: ignore[attr-defined]
    model._best_val_hit = best_hit  # type: ignore[attr-defined]
    model._history_selector = selector  # type: ignore[attr-defined]
    print(f"restored epoch={best_epoch} val_hit={best_hit:.4f}")
    model._cached_title_map = title_map  # type: ignore[attr-defined]
    model._h_train = h_train  # type: ignore[attr-defined]
    return model


def train_lora(args, dataset, train_subset, text_emb: torch.Tensor):
    """Full LoRA path — intended for RunPod; works on Mac but slowly."""
    from minigenrec.llm import LLMRanker
    from train import FrameDataset

    seed_everything(args.model_seed)
    tokenizer = load_tokenizer(MODEL_ID, MODEL_REVISION)
    backbone = load_backbone(MODEL_ID, MODEL_REVISION, freeze=True)
    user = LLMUserEncoder(backbone, tokenizer, use_lora=True)
    cold = torch.from_numpy(dataset.cold_mask)
    item_enc = ItemEncoder(len(dataset.full_to_movie), DIM, mode=args.items, text_dim=int(text_emb.shape[1]), cold_mask=cold)
    head = ScoreHead(user.hidden_size, DIM)
    model = LLMRanker(user, item_enc, head).to(DEVICE)
    title_map = _title_map(dataset, args.titles)
    selector = HistorySelector(dataset, tokenizer)
    text = text_emb.to(DEVICE)
    warm_to_full = torch.from_numpy(dataset.warm_to_full).to(DEVICE)
    full_to_warm = torch.from_numpy(dataset.full_to_warm).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LLM_LR)
    frame_ds = FrameDataset(train_subset)

    def collate(idxs):
        batch = train_subset.iloc[list(idxs)]
        prompts = prompts_for_batch(dataset, batch, title_map, tokenizer, selector=selector)
        enc = tokenize_prompts(tokenizer, prompts, MAX_LEN)
        targets = torch.from_numpy(batch["target_full"].to_numpy(dtype=np.int64, copy=True))
        return enc["input_ids"], enc["attention_mask"], targets

    @torch.no_grad()
    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        model.eval()
        prompts = prompts_for_batch(dataset, batch, title_map, tokenizer, selector=selector)
        enc = tokenize_prompts(tokenizer, prompts, MAX_LEN)
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            text_emb=text,
        )
        return logits.float().cpu().numpy()

    loader = DataLoader(frame_ds, batch_size=LORA_BATCH, shuffle=True, collate_fn=collate)
    t0 = time.time()
    best_hit, best_epoch, best_state = -1.0, 0, None
    for epoch in range(LORA_EPOCHS):
        model.train()
        total = 0.0
        n = 0
        for input_ids, attn, targets in loader:
            input_ids, attn, targets = input_ids.to(DEVICE), attn.to(DEVICE), targets.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids, attn, text_emb=text)
            loss = warm_ce_loss(logits.float(), targets, warm_to_full, full_to_warm)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(targets)
            n += len(targets)
            if n % 64 == 0:
                print(f"  step n={n} loss={total/n:.4f}")
        hit = mean_hit(dataset, dataset.val, score_fn, EVAL_BATCH_LLM)
        if hit > best_hit:
            best_hit = hit
            best_epoch = epoch + 1
            best_state = trainable_state_dict(model)
        print(f"epoch {epoch+1}/{LORA_EPOCHS} loss={total/max(n,1):.4f} val_hit={hit:.4f} ({time.time()-t0:.0f}s)")
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    model._best_epoch = best_epoch  # type: ignore[attr-defined]
    model._best_val_hit = best_hit  # type: ignore[attr-defined]
    model._history_selector = selector  # type: ignore[attr-defined]
    print(f"restored epoch={best_epoch} val_hit={best_hit:.4f}")
    model._cached_title_map = title_map  # type: ignore[attr-defined]
    return model


def run_llm(args, dataset, train_subset) -> None:
    from train import base_payload, eval_all, write_results

    if args.items not in ("text", "hybrid"):
        raise SystemExit("LLM models need --items text|hybrid")
    text_emb = load_text_emb(dataset, title_mode=args.titles)
    use_lora = args.model == "genrec"
    if use_lora:
        model = train_lora(args, dataset, train_subset, text_emb)
    else:
        model = train_frozen(args, dataset, train_subset, text_emb)

    title_map = getattr(model, "_cached_title_map", _title_map(dataset, args.titles))
    selector = model._history_selector  # type: ignore[attr-defined]
    tokenizer = load_tokenizer(MODEL_ID, MODEL_REVISION)
    eval_user = None
    if not use_lora:
        backbone = load_backbone(MODEL_ID, MODEL_REVISION, freeze=True)
        eval_user = LLMUserEncoder(backbone, tokenizer, use_lora=False).to(DEVICE)
        eval_user.eval()

    @torch.no_grad()
    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        model.eval()
        if use_lora:
            prompts = prompts_for_batch(dataset, batch, title_map, tokenizer, selector=selector)
            enc = tokenize_prompts(tokenizer, prompts, MAX_LEN)
            logits = model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
                text_emb=text_emb.to(DEVICE),
            )
            return logits.float().cpu().numpy()
        prompts = prompts_for_batch(dataset, batch, title_map, tokenizer, selector=selector)
        enc = tokenize_prompts(tokenizer, prompts, MAX_LEN)
        assert eval_user is not None
        h = eval_user(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        logits = model(h.float(), text_emb.to(DEVICE))
        return logits.float().cpu().numpy()

    payload = base_payload(args, dataset)
    payload["items"] = args.items
    payload["titles"] = args.titles
    payload["run_name"] = args.run_name or f"{args.model}_{args.items}_n{args.n_train}_seed{args.model_seed}"
    payload["train"] = {
        "mode": "lora" if use_lora else "frozen_cached_h",
        "epochs": LORA_EPOCHS if use_lora else FROZEN_EPOCHS,
        "best_epoch": model._best_epoch,  # type: ignore[attr-defined]
        "best_val_hit": model._best_val_hit,  # type: ignore[attr-defined]
        "dim": DIM,
        "lr": LLM_LR,
    }
    segs, per_user = eval_all(dataset, score_fn, EVAL_BATCH_LLM)
    payload["segments"] = segs
    write_results(payload["run_name"], payload, per_user)
    saved_state = trainable_state_dict(model) if use_lora else model.state_dict()
    torch.save(
        {
            "state_dict": saved_state,
            "state_format": "trainable_only" if use_lora else "full",
            "items": args.items,
            "lora": use_lora,
        },
        RESULTS_DIR / payload["run_name"] / "model.pt",
    )
