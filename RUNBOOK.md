# Runbook (CPU Cluster): GPT-2 124M streaming training + LoRA + vLLM + UI

## 0) Reality check (CPU economics)
GPT-2 124M is not huge, but training is still expensive on CPU.
Expect:
- low throughput (tokens/sec)
- you must cap steps and use small batch sizes
- use LoRA first; full fine-tune is slower

## 1) Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

## 2) Dataset streaming (no full download)
We use Hugging Face Datasets streaming:
- `load_dataset(..., streaming=True)` returns an iterable dataset.
- Data is downloaded progressively while you iterate. citeturn0search2turn0search14turn0search6

You can change datasets in `configs/*.yaml` by setting:
- dataset_name
- dataset_config (optional)
- split
- text_field

## 3) Training modes
### A) Continue pretraining (next-token)
This is domain-adaptive pretraining: teach the model your domain text.
Run:
```bash
python scripts/train.py --config configs/pretrain_lora_cpu.yaml
```

### B) Instruction fine-tune (SFT)
Teaches "follow instructions" behavior using instruction datasets.
Run:
```bash
python scripts/train.py --config configs/sft_lora_cpu.yaml
```

## 4) Outputs
Each run creates:
```
outputs/<timestamp>-<run_name>-<mode>/
  ckpt-step-<n>/
  metadata.json
```

`metadata.json` includes a dataset fingerprint to detect drift.

## 5) Dataset drift & retrain trigger
If upstream dataset changes and you want to retrain:
- Use `scripts/watch_dataset.py` which fingerprints the first N batches.
- If changed, it triggers training.

One-shot check + run:
```bash
python scripts/watch_dataset.py --config configs/pretrain_lora_cpu.yaml
```

Cron example (daily):
```bash
0 2 * * * cd /path/to/project && . .venv/bin/activate && python scripts/watch_dataset.py --config configs/pretrain_lora_cpu.yaml >> logs/watch.log 2>&1
```

## 6) Serving with vLLM (CPU)
vLLM supports CPU serving, but on **x86** there may be **no prebuilt wheels**, so you build from source. citeturn0search0turn0search12

### Option A: Use vLLM docs to install CPU build
Follow vLLM CPU install for your CPU type. citeturn0search0turn0search12

Then serve:
```bash
API_KEY=local-dev-key PORT=8000 bash scripts/serve_vllm.sh outputs/<run_id>/export
```

### Option B: If vLLM CPU build is not feasible
Fallback (still works): use Transformers `text-generation` pipeline in `scripts/serve_transformers.py`.
This is slower than vLLM but easier to run on CPU.

## 7) LoRA in vLLM
vLLM supports LoRA adapters and even runtime loading with config env vars. citeturn0search1turn0search4turn0search5

This project’s simplest production approach:
- merge LoRA -> export -> serve export

Merge:
```bash
python scripts/merge_lora.py --run_dir outputs/<run_id> --base_model gpt2
```

Serve export:
```bash
bash scripts/serve_vllm.sh outputs/<run_id>/export
```

## 8) UI
The Streamlit UI calls the server via OpenAI client.
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local-dev-key
export MODEL_NAME=gpt2
streamlit run scripts/ui_streamlit.py
```

## 9) Security basics (production minimum)
- vLLM: set `--api-key` (and don’t hardcode it)
- Put vLLM behind an API gateway with auth (OAuth/OIDC) in production
- Version every model artifact; do not overwrite
- Keep config + dataset fingerprint with every trained artifact
