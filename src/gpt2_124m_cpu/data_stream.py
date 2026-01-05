from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Iterator

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .prompts import TEMPLATES

@dataclass
class StreamConfig:
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    streaming: bool
    shuffle_buffer: int
    block_size: int

    # pretrain text
    text_field: Optional[str] = "text"
    pack_tokens: bool = True

    # sft
    prompt_field: Optional[str] = None
    context_field: Optional[str] = None
    response_field: Optional[str] = None
    sft_template: Optional[str] = None

def _load_stream(sc: StreamConfig):
    if sc.dataset_config and sc.dataset_config != "null":
        ds = load_dataset(sc.dataset_name, sc.dataset_config, split=sc.split, streaming=sc.streaming)
    else:
        ds = load_dataset(sc.dataset_name, split=sc.split, streaming=sc.streaming)
    if sc.streaming and sc.shuffle_buffer and sc.shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=int(sc.shuffle_buffer), seed=123)
    return ds

def _pack_tokens(token_iter, block_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    buf: List[int] = []
    for tid in token_iter:
        buf.append(int(tid))
        if len(buf) >= block_size + 1:
            chunk = buf[: block_size + 1]
            buf = buf[block_size:]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y

def make_pretrain_iter(sc: StreamConfig, tokenizer_name: str):
    ds = _load_stream(sc)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token; use eos for padding

    def token_stream():
        for row in ds:
            text = row.get(sc.text_field) if sc.text_field else None
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for tid in ids:
                yield tid

    if sc.pack_tokens:
        yield from _pack_tokens(token_stream(), sc.block_size)
    else:
        for row in ds:
            text = row.get(sc.text_field)
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"][: sc.block_size + 1]
            if len(ids) < 2:
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            yield x, y

def make_sft_iter(sc: StreamConfig, tokenizer_name: str):
    if not sc.sft_template:
        raise ValueError("sft_template required for SFT")
    fmt = TEMPLATES[sc.sft_template]
    ds = _load_stream(sc)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tok.pad_token = tok.eos_token

    for row in ds:
        instr = row.get(sc.prompt_field) if sc.prompt_field else None
        resp = row.get(sc.response_field) if sc.response_field else None
        ctx = row.get(sc.context_field) if sc.context_field else None
        if not instr or not resp:
            continue
        full = fmt(str(instr), str(ctx) if ctx is not None else None, str(resp))
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=sc.block_size)["input_ids"]
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        yield x, y

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.full((len(xs), max_len), fill_value=0, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), fill_value=-100, dtype=torch.long)  # ignore index
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, : x.size(0)] = x
        y_pad[i, : y.size(0)] = y
    return x_pad, y_pad
