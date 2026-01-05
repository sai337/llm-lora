from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from gpt2_124m_cpu.config import load_yaml
from gpt2_124m_cpu.data_stream import StreamConfig
from gpt2_124m_cpu.model import build_model
from gpt2_124m_cpu.lora_utils import apply_lora
from gpt2_124m_cpu.train_loop import train_streaming

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    task = cfg["run"]["task"]
    mode = cfg["run"]["mode"]
    run_name = cfg["run"].get("run_name", "run")

    device = cfg.get("compute", {}).get("device", "cpu")
    mixed_precision = cfg.get("compute", {}).get("mixed_precision", "no")

    if device == "cpu" and mode == "qlora":
        raise SystemExit("QLoRA mode is GPU/CUDA oriented and not supported on CPU in this project. Use LoRA or full fine-tune instead.")

    base_model = cfg["model"]["base_model"]
    init = cfg["model"].get("init", "pretrained")
    max_length = int(cfg["model"].get("max_length", 512))

    data_cfg = cfg["data"]
    sc = StreamConfig(
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg.get("dataset_config"),
        split=data_cfg.get("split", "train"),
        streaming=bool(data_cfg.get("streaming", True)),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 0) or 0),
        block_size=int(data_cfg.get("block_size", max_length)),
        text_field=data_cfg.get("text_field", "text"),
        pack_tokens=bool(data_cfg.get("pack_tokens", True)),
        prompt_field=data_cfg.get("prompt_field"),
        context_field=data_cfg.get("context_field"),
        response_field=data_cfg.get("response_field"),
        sft_template=cfg.get("sft", {}).get("template"),
    )

    tcfg = cfg["train"]
    out_root = Path(tcfg.get("output_dir", "outputs"))
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{run_name}-{mode}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = build_model(base_model=base_model, init=init, max_length=max_length, device=device)

    if mode == "lora":
        lcfg = cfg["lora"]
        model = apply_lora(model, r=int(lcfg["r"]), alpha=int(lcfg["alpha"]), dropout=float(lcfg["dropout"]), target_modules=list(lcfg["target_modules"]))

    train_streaming(
        model=model,
        tokenizer_name=base_model,
        stream_cfg=sc,
        task=task,
        out_dir=out_dir,
        max_steps=int(tcfg["max_steps"]),
        per_device_batch_size=int(tcfg["per_device_batch_size"]),
        grad_accum_steps=int(tcfg["grad_accum_steps"]),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        warmup_steps=int(tcfg.get("warmup_steps", 0)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 0.0)),
        log_every=int(tcfg.get("log_every", 50)),
        save_every=int(tcfg.get("save_every", 500)),
        seed=int(tcfg.get("seed", 123)),
        mixed_precision=mixed_precision,
    )

    print(f"Done. Run dir: {out_dir}")

if __name__ == "__main__":
    main()
