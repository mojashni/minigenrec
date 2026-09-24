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


def averaged_user_metric(
    results_dir: Path,
    selector: str,
    segment: str,
    metric: str,
) -> tuple[pd.Series, list[str]]:
    """Load one run, or average matching ``<selector>_seed*`` runs per user."""
    exact = results_dir / selector / f"per_user_{segment}.csv"
    if exact.exists():
        paths = [exact]
    else:
        paths = [
            path
            for path in sorted(results_dir.glob(f"{selector}_seed*/per_user_{segment}.csv"))
            if re.fullmatch(rf"{re.escape(selector)}_seed\d+", path.parent.name)
        ]
    if not paths:
        raise FileNotFoundError(
            f"no per-user files for {selector!r}, segment {segment!r} in {results_dir}"
        )

    series = []
    for path in paths:
        frame = pd.read_csv(path)
        if "user_id" not in frame or metric not in frame:
            raise ValueError(f"{path} must contain user_id and {metric}")
        if frame["user_id"].duplicated().any():
            raise ValueError(f"duplicate user_id in {path}")
        series.append(frame.set_index("user_id")[metric].sort_index().rename(path.parent.name))

    expected = series[0].index
    for values in series[1:]:
        if not values.index.equals(expected):
            raise ValueError(f"seed user sets differ for {selector!r}")
    return pd.concat(series, axis=1).mean(axis=1), [path.parent.name for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument(
        "--a",
        help="run dir or pre-seed prefix A (e.g. genrec_hybrid_n100000)",
    )
    parser.add_argument("--b", help="run dir or pre-seed prefix B")
    parser.add_argument("--segment", default="cold_test")
    parser.add_argument("--metric", default="hit", choices=["hit", "ndcg", "mrr"])
    args = parser.parse_args()

    if not (args.a and args.b):
        print(summary_markdown(seed_table(RESULTS_DIR, args.n_train)), end="")
        return
    a, runs_a = averaged_user_metric(RESULTS_DIR, args.a, args.segment, args.metric)
    b, runs_b = averaged_user_metric(RESULTS_DIR, args.b, args.segment, args.metric)
    delta, lo, hi = paired_bootstrap(a, b, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    print(
        json.dumps(
            {
                "a": args.a,
                "b": args.b,
                "runs_a": runs_a,
                "runs_b": runs_b,
                "segment": args.segment,
                "metric": args.metric,
                "delta": delta,
                "lo": lo,
                "hi": hi,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
