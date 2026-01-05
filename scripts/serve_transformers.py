from __future__ import annotations
# CPU fallback server if vLLM CPU build is painful.
# Provides a minimal OpenAI-compatible-ish endpoint using FastAPI.

import os
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "gpt2")
PORT = int(os.environ.get("PORT", "8000"))

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

app = FastAPI()

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/v1/chat/completions")
def chat(req: ChatRequest) -> Dict[str, Any]:
    prompt = "\n".join([m["content"] for m in req.messages])
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
