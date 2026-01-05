from __future__ import annotations
import argparse, time, subprocess, json
from pathlib import Path
from gpt2_124m_cpu.config import load_yaml
from gpt2_124m_cpu.data_stream import StreamConfig, make_pretrain_iter, make_sft_iter
from gpt2_124m_cpu.train_loop import fingerprint_stream

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state_dir", default=".state")
    ap.add_argument("--every-minutes", type=int, default=0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    task = cfg["run"]["task"]
    base_model = cfg["model"]["base_model"]

    data_cfg = cfg["data"]
    sc = StreamConfig(
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg.get("dataset_config"),
        split=data_cfg.get("split", "train"),
        streaming=bool(data_cfg.get("streaming", True)),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 0) or 0),
        block_size=int(data_cfg.get("block_size", 512)),
        text_field=data_cfg.get("text_field", "text"),
        pack_tokens=bool(data_cfg.get("pack_tokens", True)),
        prompt_field=data_cfg.get("prompt_field"),
        context_field=data_cfg.get("context_field"),
        response_field=data_cfg.get("response_field"),
        sft_template=cfg.get("sft", {}).get("template"),
    )

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / (Path(args.config).stem + ".json")

    def compute_fp():
        it = make_pretrain_iter(sc, base_model) if task=="pretrain" else make_sft_iter(sc, base_model)
        return fingerprint_stream(it, n_batches=30)

    def maybe_train():
        fp = compute_fp()
        prev = None
        if state_file.exists():
            prev = json.loads(state_file.read_text(encoding="utf-8")).get("fingerprint")
        if prev != fp:
            print(f"Dataset fingerprint changed (or first run). prev={prev} new={fp}. Triggering training.")
            subprocess.check_call(["python", "scripts/train.py", "--config", args.config])
            state_file.write_text(json.dumps({"fingerprint": fp, "last_run": time.time()}, indent=2), encoding="utf-8")
        else:
            print(f"No dataset change. fingerprint={fp}")

    if args.every_minutes and args.every_minutes > 0:
        while True:
            maybe_train()
            time.sleep(args.every_minutes * 60)
    else:
        maybe_train()

if __name__ == "__main__":
    main()
