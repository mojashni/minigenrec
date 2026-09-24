#!/usr/bin/env bash
# Main table: 5 models × 3 seeds on 100k. Skip finished runs.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
N=100000
SEEDS=(0 1 2)

run() {
  local name="$1"; shift
  local out="results/$name/metrics.json"
  if [[ -f "$out" ]]; then
    echo "skip $name"
    return 0
  fi
  echo "=== $name ==="
  "$PY" train.py "$@" --run-name "$name"
}

run popular_n${N}_seed0 --model popular --n-train "$N" --model-seed 0

for s in "${SEEDS[@]}"; do
  run "sasrec_id_n${N}_seed${s}" --model sasrec --items id --n-train "$N" --model-seed "$s"
  run "sasrec_hybrid_n${N}_seed${s}" --model sasrec --items hybrid --titles real --n-train "$N" --model-seed "$s"
  run "frozen_llm_hybrid_n${N}_seed${s}" --model frozen_llm --items hybrid --titles real --n-train "$N" --model-seed "$s"
  run "genrec_hybrid_n${N}_seed${s}" --model genrec --items hybrid --titles real --n-train "$N" --model-seed "$s"
done

mkdir -p results/bundle
tar -czf "results/bundle/main_table_$(date +%Y%m%d_%H%M).tar.gz" results/*_n${N}_seed*/metrics.json results/*_n${N}_seed*/per_user_*.csv results/data_stats.json results/model_revision.json results/choose_k.json 2>/dev/null || true
echo "done"
