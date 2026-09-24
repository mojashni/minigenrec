# MiniGenRec

LLM-native movie ranking **inspired by** Netflix GenRec ([tech blog](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)). Not a full reimplementation (no Phase-1 corpus, no LM objective, no Netflix reward models).

## Research question

Does a text LLM (LoRA) beat a sequential recommender that sees **the same events and metadata** (SASRec-Hybrid) under low data and **synthetic zero-shot item cold-start**?

Primary comparison: **MiniGenRec-Hybrid − SASRec-Hybrid** (paired bootstrap on users).

## Results so far (Mac, 100k train)

| Model | Warm Hit@10 | Cold Hit@10 |
|-------|-------------|-------------|
| Popularity | 0.048 | 0.000 |
| SASRec-ID (3 seeds) | 0.163 ± 0.014 | 0.002 ± 0.001 |
| SASRec-Hybrid (3 seeds) | 0.140 ± 0.001 | **0.053 ± 0.008** |

Paired Δ cold Hit@10 (Hybrid − ID, seed 0): **+0.057** [0.046, 0.068].

Metadata helps cold-start a lot; ID-only is near chance on cold (as expected).

**Still needed on RunPod:** Frozen-LLM-Hybrid and MiniGenRec-Hybrid (3 seeds). See [RUNPOD.md](RUNPOD.md).

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

Main table (skip finished runs):

```bash
bash run_all.sh
```

## Protocol (short)

- MovieLens 1M; positives = rating ≥ 4
- Stratified synthetic cold items; cold interactions removed from all histories
- Cold never train target / never in train softmax
- Cold-test only after each user's `train_cutoff`
- Shared history of K=21 events for every model
- Nested subsets `10k ⊂ 30k ⊂ 100k`; only `model_seed` varies across 3 runs

Full contract: [PLAN.md](PLAN.md).
