from __future__ import annotations
from typing import List
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora(model, r: int, alpha: int, dropout: float, target_modules: List[str]):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, cfg)
    lora_model.print_trainable_parameters()
    return lora_model
