"""Aggregate runs: mean±std across model seeds, paired bootstrap Δ."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from minigenrec.config import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, RESULTS_DIR
from minigenrec.metrics import paired_bootstrap

COLUMNS = {
    "warm Hit@10": ("warm_test_full", "hit"),
    "warm-only Hit@10": ("warm_test_warm_only", "hit"),
    "cold Hit@10": ("cold_test", "hit"),
    "cold-only Hit@10": ("cold_test_cold_only", "hit"),
    "cold intrusion": ("warm_test_full", "cold_intrusion"),
}


def _value(segments: dict, segment: str, key: str):
    block = segments.get(segment, {}).get(key)
    return block.get("mean") if isinstance(block, dict) else block


def seed_table(results_dir: Path, n_train: int) -> pd.DataFrame:
    rows = []
    for metrics in sorted(results_dir.glob(f"*_n{n_train}_seed*/metrics.json")):
        payload = json.loads(metrics.read_text())
        row = {"model": re.sub(r"_n\d+_seed\d+$", "", metrics.parent.name)}
        for col, (segment, key) in COLUMNS.items():
            row[col] = _value(payload["segments"], segment, key)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "no runs found\n"
    lines = ["| model | seeds | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 2)]
    for model, group in df.groupby("model", sort=True):
        cells = []
        for col in COLUMNS:
            vals = group[col].dropna().astype(float)
            if vals.empty:
                cells.append("—")
            elif len(vals) == 1:
                cells.append(f"{vals.iloc[0]:.4f}")
            else:
                cells.append(f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
        lines.append(f"| {model} | {len(group)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--a", help="run dir A for a paired bootstrap (e.g. genrec_hybrid_n100000_seed0)")
    parser.add_argument("--b", help="run dir B")
    parser.add_argument("--segment", default="cold_test")
    parser.add_argument("--metric", default="hit", choices=["hit", "ndcg", "mrr"])
    args = parser.parse_args()

    if not (args.a and args.b):
        print(summary_markdown(seed_table(RESULTS_DIR, args.n_train)), end="")
        return
    a = pd.read_csv(RESULTS_DIR / args.a / f"per_user_{args.segment}.csv").set_index("user_id")[args.metric]
    b = pd.read_csv(RESULTS_DIR / args.b / f"per_user_{args.segment}.csv").set_index("user_id")[args.metric]
    delta, lo, hi = paired_bootstrap(a, b, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    print(json.dumps({"segment": args.segment, "metric": args.metric, "delta": delta, "lo": lo, "hi": hi}, indent=2))


if __name__ == "__main__":
    main()
