from __future__ import annotations

import time
import hashlib
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from .data_stream import StreamConfig, make_pretrain_iter, make_sft_iter, collate_batch

def fingerprint_stream(iterable, n_batches: int = 30) -> str:
    h = hashlib.sha256()
    for i, (x, _) in enumerate(iterable):
        h.update(x.numpy().tobytes())
        if i >= n_batches:
            break
    return h.hexdigest()

def save_checkpoint(accelerator: Accelerator, model, out_dir: Path, step: int):
    save_dir = out_dir / f"ckpt-step-{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_dir, safe_serialization=True)

def train_streaming(
    model,
    tokenizer_name: str,
    stream_cfg: StreamConfig,
    task: str,
    out_dir: Path,
    max_steps: int,
    per_device_batch_size: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_grad_norm: float,
    log_every: int,
    save_every: int,
    seed: int,
    mixed_precision: str,
):
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision=None if mixed_precision == "no" else mixed_precision)
    device = accelerator.device

    # dataset fingerprint
    it_fp = make_pretrain_iter(stream_cfg, tokenizer_name) if task == "pretrain" else make_sft_iter(stream_cfg, tokenizer_name)
    fp = fingerprint_stream(it_fp, n_batches=30)

    it = make_pretrain_iter(stream_cfg, tokenizer_name) if task == "pretrain" else make_sft_iter(stream_cfg, tokenizer_name)
    loader = DataLoader(it, batch_size=per_device_batch_size, collate_fn=collate_batch)

    # optimizer: AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()

    global_step = 0
    accum = 0.0
    pbar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    start = time.time()

    while global_step < max_steps:
        for (x, y) in loader:
            with accelerator.accumulate(model):
                out = model(input_ids=x, labels=y)
                loss = out.loss
                accelerator.backward(loss)

                if max_grad_norm and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            accum += float(loss.detach().cpu())
            global_step += 1
            pbar.update(1)

            if accelerator.is_local_main_process and (global_step % log_every == 0):
                avg = accum / log_every
                accum = 0.0
                elapsed = time.time() - start
                pbar.set_description(f"step={global_step} loss={avg:.4f} elapsed={elapsed:.0f}s")

            if accelerator.is_local_main_process and (global_step % save_every == 0):
                save_checkpoint(accelerator, model, out_dir, global_step)

            if global_step >= max_steps:
                break

    if accelerator.is_local_main_process:
        save_checkpoint(accelerator, model, out_dir, global_step)
        (out_dir / "metadata.json").write_text(
            __import__("json").dumps({
                "task": task,
                "tokenizer": tokenizer_name,
                "dataset_fingerprint": fp,
                "max_steps": max_steps,
                "per_device_batch_size": per_device_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "mixed_precision": mixed_precision,
            }, indent=2),
            encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    pbar.close()
