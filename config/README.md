# GPT-2 124M (CPU-first) — Streaming Training + LoRA + vLLM Serving + UI

This is a **separate project** for CPU clusters. It does not touch your other repo.

What it supports on CPU:
- GPT-2 small (124M) **pretrained weights** OR **scratch init**
- **Streaming datasets** via Hugging Face Datasets (`streaming=True`) (no full dataset download)
- **Full fine-tune** and **LoRA** (CPU-compatible)

What is **not realistic on CPU**:
- **QLoRA** (4-bit quantized training) is designed around GPU quantization kernels; we keep a config file for it
  but it will not run on CPU in typical environments. Use it only when you have CUDA GPUs. citeturn0search3turn0search7

Serving:
- vLLM supports CPU serving, but on x86/AMD64 you may need to **build from source** (no prebuilt wheels). citeturn0search0turn0search12
- LoRA serving in vLLM is supported. citeturn0search1turn0search4turn0search5

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

### 1) Continue pretraining (CPU full fine-tune)
```bash
python scripts/train.py --config configs/pretrain_full_cpu.yaml
```

### 2) Continue pretraining (CPU LoRA)
```bash
python scripts/train.py --config configs/pretrain_lora_cpu.yaml
```

### 3) Export a merged model (for simplest serving)
```bash
python scripts/merge_lora.py --run_dir outputs/<run_id> --base_model gpt2
```

### 4) Serve (vLLM)
See `RUNBOOK.md` — CPU install is special. citeturn0search0turn0search12

### 5) UI
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local-dev-key
export MODEL_NAME=gpt2
streamlit run scripts/ui_streamlit.py
```
