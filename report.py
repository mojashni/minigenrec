"""Aggregate runs: mean±std across model seeds, paired bootstrap Δ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from minigenrec.config import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, RESULTS_DIR
from minigenrec.metrics import paired_bootstrap


def load_per_user(run_dir: Path, segment: str) -> pd.Series:
    path = run_dir / f"per_user_{segment}.csv"
    df = pd.read_csv(path)
    return df.set_index("user_id")["hit"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="run dir name A (e.g. genrec_hybrid_...)")
    parser.add_argument("--b", required=True, help="run dir name B (e.g. sasrec_hybrid_...)")
    parser.add_argument("--segment", default="cold_test")
    parser.add_argument("--metric", default="hit", choices=["hit", "ndcg", "mrr"])
    args = parser.parse_args()

    a_dir = RESULTS_DIR / args.a
    b_dir = RESULTS_DIR / args.b
    a = pd.read_csv(a_dir / f"per_user_{args.segment}.csv").set_index("user_id")[args.metric]
    b = pd.read_csv(b_dir / f"per_user_{args.segment}.csv").set_index("user_id")[args.metric]
    delta, lo, hi = paired_bootstrap(a, b, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    print(json.dumps({"segment": args.segment, "metric": args.metric, "delta": delta, "lo": lo, "hi": hi}, indent=2))


if __name__ == "__main__":
    main()
