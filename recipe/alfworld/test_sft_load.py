"""End-to-end smoke test: load a converter parquet via verl's SFT pipeline.

Run from the verl repo root:
    python -m recipe.alfworld.test_sft_load \\
        --parquet /tmp/sft_smoke.parquet \\
        --tokenizer Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verl.trainer.sft_trainer import create_sft_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    data_cfg = OmegaConf.create({
        "train_files": args.parquet,
        "val_files": None,
        "messages_key": "messages",
        "tools_key": "tools",
        "enable_thinking_key": "enable_thinking",
        "enable_thinking_default": False,
        "pad_mode": "right",
        "max_length": args.max_length,
        "truncation": "right",
        "apply_chat_template_kwargs": {"enable_thinking": False},
        "ignore_input_ids_mismatch": True,  # Qwen3 chat template strips <think> from non-last turns
        "custom_cls": {"path": None, "name": None},
    })
    ds = create_sft_dataset(args.parquet, data_cfg, tokenizer, processor=None)
    print(f"dataset len: {len(ds)}")

    sample = ds[0]
    print("sample keys:", list(sample.keys()))
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("loss_mask shape:", sample["loss_mask"].shape, "sum:", int(sample["loss_mask"].sum().item()))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    print("\nbatch tensor shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

    # decode first record, mark assistant tokens
    ids = batch["input_ids"][0]
    lm = batch["loss_mask"][0]
    am = batch["attention_mask"][0]
    real_len = int(am.sum().item())
    text = tokenizer.decode(ids[:real_len])
    print("\nfirst record decoded (first 600 chars):")
    print(text[:600])
    print("\n... last 400 chars (incl assistant tail):")
    print(text[-400:])
    print(f"\nloss_mask: trainable={int(lm.sum().item())}/{real_len} non-pad tokens")
    print(f"loss_mask trainable fraction (non-pad): {lm.sum().item() / max(real_len, 1):.3f}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
