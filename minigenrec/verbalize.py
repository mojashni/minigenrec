"""Prompt and movie text. Title modes: real | shuffled | none."""

from __future__ import annotations

import hashlib
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
    return len(tokenizer.encode(text, add_special_tokens=False))


def event_costs(
    tokenizer,
    title_map: dict[int, str],
    movie_ids: np.ndarray,
    modes: tuple[str, ...] = ("real", "shuffled", "none"),
) -> np.ndarray:
    """Per-event token cost = max across title modes (keeps history identical)."""
    # Rebuild maps for shuffled/none once
    # Caller should pass movie_id -> texts for each mode; here we only have one map.
    # Use length of provided texts; for cross-mode max, caller merges.
    return np.array(
        [token_len(tokenizer, event_line(title_map[int(m)], 5)) for m in movie_ids],
        dtype=np.int64,
    )


def fit_events_to_budget(
    event_lines: list[str],
    tokenizer,
    max_len: int = MAX_LEN,
) -> list[str]:
    """Drop oldest whole events until prompt + marker fits. Never split an event."""
    lines = list(event_lines)
    while True:
        prompt = build_prompt(lines)
        if token_len(tokenizer, prompt) <= max_len:
            return lines
        if not lines:
            return lines
        lines = lines[1:]


def cache_key(model_id: str, revision: str | None, title_mode: str, meta_version: str = "ml1m-v1") -> str:
    raw = f"{model_id}|{revision or 'main'}|{title_mode}|{meta_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def text_cache_path(model_id: str, revision: str | None, title_mode: str) -> Path:
    return DATA_DIR / "cache" / f"text_{cache_key(model_id, revision, title_mode)}.pt"
