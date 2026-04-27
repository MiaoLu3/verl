"""Smoke test for PretokenizedSFTDataset.

Builds a tokens-included parquet from the smoke teacher dump, instantiates
PretokenizedSFTDataset directly, prints sample shapes and decodes the first
100 token IDs, then verifies loss_mask / attention_mask round-trip.

Run from the verl repo root:
    python -m recipe.alfworld.test_pretokenized_sft \\
        --teacher_dir /scratch/.../teacher_rollouts/smoke_290123_... \\
        --tokenizer Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
from transformers import AutoTokenizer

from recipe.alfworld.pretokenized_sft_dataset import PretokenizedSFTDataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max_length", type=int, default=16384,
                   help="Match RL rollout's --max_model_len 16384 (= prompt 4k + "
                        "response 12k). Anything shorter would tail-truncate "
                        "~28%% of teacher trajectories and risk dropping the "
                        "winning final action.")
    args = p.parse_args()

    tmp = tempfile.mkdtemp(prefix="pretok_sft_smoke_")
    parquet_path = os.path.join(tmp, "pretok.parquet")
    cmd = [
        sys.executable, "-m", "recipe.alfworld.jsonl_to_parquet",
        "--teacher_dir", args.teacher_dir,
        "--out", parquet_path,
        "--keep", "all",
        "--include_tokens", "true",
    ]
    print("[smoke] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"[smoke] parquet rows={len(df)} columns={list(df.columns)}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cfg = {"max_length": args.max_length, "truncation": "right", "pad_mode": "right"}
    ds = PretokenizedSFTDataset(parquet_path, tokenizer=tok, config=cfg)

    assert len(ds) == len(df), f"len mismatch: ds={len(ds)} df={len(df)}"
    sample = ds[0]
    print("[smoke] sample keys:", list(sample.keys()))
    for k, v in sample.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    pre_pad_len = int(df.iloc[0]["pretok_input_len"])
    expected_loss_ones = int(df.iloc[0]["response_loss_ones"])
    real_len = min(pre_pad_len, args.max_length)

    am_sum = int(sample["attention_mask"].sum().item())
    lm_sum = int(sample["loss_mask"].sum().item())

    print(f"[smoke] pre_pad_len={pre_pad_len} max_length={args.max_length} real_len={real_len}")
    print(f"[smoke] attention_mask.sum={am_sum} (expect {real_len})")
    print(f"[smoke] loss_mask.sum={lm_sum} (expect {expected_loss_ones} when no truncation)")

    decoded = tok.decode(sample["input_ids"][:100].tolist())
    print(f"[smoke] decoded first 100 tokens:\n  {decoded!r}")

    failures = []
    if am_sum != real_len:
        failures.append(f"attention_mask sum {am_sum} != real_len {real_len}")
    # loss_mask round-trips exactly when sample wasn't truncated
    if pre_pad_len <= args.max_length and lm_sum != expected_loss_ones:
        failures.append(f"loss_mask sum {lm_sum} != response_loss_ones {expected_loss_ones}")

    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print(f"PASS ({len(ds)} rows, sample0 loss_mask={lm_sum}/{real_len} attn_mask sum ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
