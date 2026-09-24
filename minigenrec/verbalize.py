"""Prompt and movie text. Title modes: real | shuffled | none."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from minigenrec.config import DATA_DIR, MAX_LEN, MODEL_ID, MODEL_REVISION, TITLE_SHUFFLE_SEED

MARKER = "\nNext movie:"
NONE_TITLE = "[TITLE]"


def movie_text_row(title: str, year: int, genres: list[str] | str, mode: str) -> str:
    if isinstance(genres, list):
        g = ", ".join(genres)
    else:
        g = str(genres).replace("|", ", ")
    if mode == "none":
        title = NONE_TITLE
    return f"{title} ({year}) | {g}"


def build_title_map(movies: pd.DataFrame, mode: str, seed: int = TITLE_SHUFFLE_SEED) -> dict[int, str]:
    titles = movies["title"].tolist()
    years = movies["year"].tolist()
    genres = movies["genres"].tolist()
    ids = movies["movie_id"].astype(int).tolist()
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        titles = list(rng.permutation(titles))
    elif mode not in ("real", "none"):
        raise ValueError(mode)
    return {
        mid: movie_text_row(titles[i], years[i], genres[i], mode)
        for i, mid in enumerate(ids)
    }


def event_line(movie_text: str, rating: int) -> str:
    return f"{movie_text} | rated {int(rating)}"


def build_prompt(event_lines: list[str]) -> str:
    body = "\n".join(event_lines)
    if body:
        return f"User recently rated:\n{body}{MARKER}"
    return f"User recently rated:{MARKER}"


def token_len(tokenizer, text: str) -> int:
    # Match tokenize_prompts exactly so the marker cannot be truncated by
    # special tokens added after the budget check.
    return len(tokenizer.encode(text, add_special_tokens=True))


def fit_events_to_budget(
    event_lines: list[str],
    tokenizer,
    max_len: int = MAX_LEN,
) -> list[str]:
    """Drop oldest whole events until prompt + marker fits. Never split an event."""
    lines = list(event_lines)
    if token_len(tokenizer, build_prompt(lines)) <= max_len:
        return lines

    # The old one-by-one loop repeatedly tokenized almost the same long prompt,
    # making --full-history quadratic. Find the smallest dropped prefix instead.
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi) // 2
        if token_len(tokenizer, build_prompt(lines[mid:])) <= max_len:
            hi = mid
        else:
            lo = mid + 1
    return lines[lo:]


def cache_key(
    model_id: str,
    revision: str | None,
    title_mode: str,
    texts: list[str],
    meta_version: str = "ml1m-v3-last-nonpad-max64-cpu-fp32",
) -> str:
    """Fingerprint the exact ordered encoder input and encoder contract."""
    texts_hash = hashlib.sha256(
        json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    raw = f"{model_id}|{revision or 'main'}|{title_mode}|{meta_version}|{texts_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def text_cache_path(model_id: str, revision: str | None, title_mode: str, texts: list[str]) -> Path:
    return DATA_DIR / "cache" / f"text_{cache_key(model_id, revision, title_mode, texts)}.pt"
