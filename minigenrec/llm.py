"""LLM user encoder (frozen or LoRA) and frozen text embedding cache."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from minigenrec.config import MAX_LEN, MODEL_ID, MODEL_REVISION
from minigenrec.model import ItemEncoder, ScoreHead
from minigenrec.verbalize import MARKER, text_cache_path


def load_tokenizer(model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
    return AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)


def load_backbone(model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION, freeze: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    return model


def apply_lora(backbone, r: int = 8, alpha: int = 16):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(backbone, cfg)


class LLMUserEncoder(nn.Module):
    def __init__(self, backbone, tokenizer, use_lora: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if use_lora:
            self.backbone = apply_lora(backbone)
        else:
            self.backbone = backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.hidden_size = int(self.backbone.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]
        # pool at last non-pad token (= marker end when prompts are built correctly)
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class LLMRanker(nn.Module):
    def __init__(self, user_encoder: LLMUserEncoder, item_encoder: ItemEncoder, score_head: ScoreHead):
        super().__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.score_head = score_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.user_encoder(input_ids, attention_mask)
        catalog = self.item_encoder(text_emb)
        return self.score_head(h, catalog)


@torch.no_grad()
def encode_movie_texts(
    backbone,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    backbone.eval()
    backbone.to(device)
    vecs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = backbone(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"]
        lengths = mask.sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
        vecs.append(pooled.float().cpu())
    return torch.cat(vecs, dim=0)


def build_or_load_text_cache(
    movies_in_catalog_order: list[str],
    title_mode: str,
    model_id: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    device: str = "cpu",
) -> torch.Tensor:
    path = text_cache_path(model_id, revision, title_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(model_id, revision)
    backbone = load_backbone(model_id, revision, freeze=True)
    emb = encode_movie_texts(backbone, tokenizer, movies_in_catalog_order, device=device)
    torch.save(emb, path)
    del backbone
    return emb


def tokenize_prompts(tokenizer, prompts: list[str], max_len: int = MAX_LEN) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    # marker must survive truncation: we left-truncate by dropping events earlier
    return enc
