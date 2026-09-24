"""Load pinned text embeddings aligned to full-catalog order."""

from __future__ import annotations

import torch

from minigenrec.config import MODEL_ID, MODEL_REVISION
from minigenrec.data import Dataset
from minigenrec.llm import build_or_load_text_cache
from minigenrec.verbalize import movie_text_row


def catalog_texts(dataset: Dataset, title_mode: str = "real") -> list[str]:
    movies = dataset.movies.set_index("movie_id")
    texts = []
    for mid in dataset.full_to_movie:
        row = movies.loc[int(mid)]
        title = row["title"]
        if title_mode == "none":
            from minigenrec.verbalize import NONE_TITLE

            title = NONE_TITLE
        elif title_mode == "shuffled":
            # build once via build_title_map for consistency
            raise ValueError("use load_text_emb(..., title_mode='shuffled') which remaps")
        texts.append(movie_text_row(title, int(row["year"]), row["genres"], "real" if title_mode == "real" else title_mode))
    return texts


def load_text_emb(
    dataset: Dataset,
    title_mode: str = "real",
    device: str = "cpu",
) -> torch.Tensor:
    from minigenrec.verbalize import build_title_map

    if title_mode == "shuffled":
        title_map = build_title_map(dataset.movies, "shuffled")
        texts = [title_map[int(mid)] for mid in dataset.full_to_movie]
    elif title_mode == "none":
        title_map = build_title_map(dataset.movies, "none")
        texts = [title_map[int(mid)] for mid in dataset.full_to_movie]
    else:
        texts = catalog_texts(dataset, "real")
    return build_or_load_text_cache(texts, title_mode, MODEL_ID, MODEL_REVISION, device=device)
