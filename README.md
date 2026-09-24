# MiniGenRec

LLM-native movie ranking **inspired by** Netflix GenRec ([tech blog](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)). Not a full reimplementation (no Phase-1 corpus, no LM objective, no Netflix reward models).

## Research question

Does a text LLM (LoRA) beat a sequential recommender that sees **the same events and metadata** (SASRec-Hybrid) under low data and **synthetic zero-shot item cold-start**?

Primary comparison: **MiniGenRec-Hybrid − SASRec-Hybrid** (paired bootstrap on users).

## Result status

The original SASRec runs used left-padded batches with right-padding pooling logic. Those
numbers are invalid and intentionally excluded from the final comparison. All final rows
must be rerun with the current canonical-history and validation-checkpoint code.

The following single-seed runs are retained only as **pilot diagnostics**. They predate the
final history-selection/checkpoint protocol and must not be reported as final results:

| Pilot run (100k, seed 0) | Warm full Hit@10 | Warm-only Hit@10 | Cold Hit@10 |
|---|---:|---:|---:|
| Frozen-LLM-Hybrid | 0.116 | 0.116 | 0.002 |
| MiniGenRec-Hybrid | 0.137 | 0.241 | 0.181 |

The GenRec pilot's cold number is **not evidence of cold-start skill**: removing cold items
from the warm-test candidates lifts its warm Hit@10 from 0.137 to 0.241, i.e. cold items were
flooding the top-10 as a group. Cold items never appear in the training softmax and are the
only items without an ID component, so a model can boost them all at once. The protocol now
trains with ID dropout and reports cold-only ranking plus cold intrusion (below). The research
claim stays open until every model is rerun under this protocol across the registered seeds.
See [RUNPOD.md](RUNPOD.md).

## Setup

```bash
cd ~/Downloads/minigenrec
uv sync
.venv/bin/python -m minigenrec.data
.venv/bin/pytest -q
.venv/bin/python train.py --model popular --n-train 100000
.venv/bin/python train.py --model sasrec --items id --n-train 100000 --model-seed 0
.venv/bin/python train.py --model sasrec --items hybrid --titles real --n-train 100000 --model-seed 0
```

Main table (skip finished runs; writes `results/summary.md` with mean ± std over seeds):

```bash
bash run_all.sh
.venv/bin/python report.py                      # seed summary
.venv/bin/python report.py --a genrec_hybrid_n100000 --b sasrec_hybrid_n100000
```

For the prefix form above, per-user metrics are averaged across all matching seeds before
the paired bootstrap. Exact run names ending in `_seed0`, etc. still compare a single seed.

## Protocol (short)

- MovieLens 1M; positives = rating ≥ 4
- Stratified synthetic cold items; cold interactions removed from all histories
- Cold never train target / never in train softmax
- Cold-test only after each user's `train_cutoff`
- Shared history: most recent K=21 events, then the same 512-token budget for every model
- Checkpoint = best epoch on warm-only validation Hit@10 (cold items never used for selection)
- Hybrid items: ID part dropped with p=0.5 per warm item during training, so text-only items are scored on the same scale
- Nested subsets `10k ⊂ 30k ⊂ 100k`; only `model_seed` varies across 3 runs

Reported segments:

| Segment | Candidates | Reads as |
|---|---|---|
| warm Hit@10 | full catalog | main warm accuracy |
| warm-only Hit@10 | warm items only | warm accuracy without cold distractors |
| cold Hit@10 | full catalog | cold target must beat warm items too |
| cold-only Hit@10 | cold items only | picks the *right* cold movie, not just "some cold movie" |
| cold intrusion | — | share of warm-test top-10 taken by cold items; high = flooding |

A real cold-start win needs high cold **and** cold-only Hit@10 with low cold intrusion.

Full contract: [PLAN.md](PLAN.md).
