# RunPod (A100) — main table

Budget: ~CAD 50 max. Expect ~3–6 GPU-hours on A100 80GB (~USD 5–10).

## One-time

1. Create RunPod account, add USD 20–30 credits.
2. Deploy **Pod**: A100 80GB, template **PyTorch 2.x**, volume ≥ 40GB.
3. SSH in (or Jupyter).

## On the pod

```bash
git clone <YOUR_GITHUB_URL> minigenrec && cd minigenrec
pip install -e .
# or: uv sync && source .venv/bin/activate

python -m minigenrec.data
# builds splits + data_stats.json (cold ratio rule runs here)

# optional: verify K / text cache (downloads Qwen once)
python - <<'PY'
from minigenrec.data import prepare
from minigenrec.text_emb import load_text_emb
ds = prepare()
emb = load_text_emb(ds, "real", device="cuda")
print(emb.shape)
PY

bash run_all.sh
```

`run_all.sh` skips any run that already has `results/<name>/metrics.json`.

## Before killing the pod

```bash
tar -czf bundle.tgz results/ data/cache/ results/bundle/ 2>/dev/null
# scp bundle.tgz down, or push adapters to HF Hub
```

Then **Stop/terminate** the pod so billing stops.

## Mac vs Pod

| Work | Where |
|------|--------|
| data, tests, Popularity, SASRec | Mac (done / in progress) |
| Frozen-LLM, GenRec LoRA, 3 seeds | RunPod |
| report.py + README tables | Mac after download |
