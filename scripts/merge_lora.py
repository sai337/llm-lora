from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_model", default="gpt2")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = sorted([p for p in run_dir.glob("ckpt-step-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        raise RuntimeError("No ckpt-step-* folders found.")
    last = ckpts[-1]

    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, last)
    merged = model.merge_and_unload()

    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(export_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.save_pretrained(export_dir)

    print(f"Exported merged model to: {export_dir}")

if __name__ == "__main__":
    main()
