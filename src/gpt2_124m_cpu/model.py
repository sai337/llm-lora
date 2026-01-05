from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

def build_model(base_model: str, init: str, max_length: int, device: str):
    # Decoder-only causal LM (GPT-2)
    if init == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(base_model)
    elif init == "scratch":
        cfg = GPT2Config.from_pretrained(base_model)
        cfg.n_positions = max_length
        cfg.n_ctx = max_length
        model = GPT2LMHeadModel(cfg)  # random init
    else:
        raise ValueError("init must be pretrained or scratch")

    # enforce context length for attention mask sizing
    model.config.n_positions = max_length
    model.config.n_ctx = max_length

    model.to(device)
    return model
