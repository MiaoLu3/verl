"""Convert teacher-rollout JSONL trajectories to a verl-SFT parquet.

Walks a teacher-rollout directory tree (``<root>/by_task_type/<task>/<id>__rollout_<n>.jsonl``)
and emits one parquet whose rows are ready to feed into
``verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`` — i.e. each
row has at minimum a ``messages`` column (list of ``{"role", "content"}``)
plus a few filterable metadata columns.

By default we keep only "good" rollouts (won=True). Override with
``--keep all`` or ``--keep losing`` if you want everything / only failures.

Usage:
    python -m recipe.alfworld.jsonl_to_parquet \\
        --teacher_dir /scratch/.../teacher_rollouts/qwen3_8b_rl_step570 \\
        --out /scratch/.../sft_data/qwen3_8b_rl_step570_won.parquet \\
        --keep won

Schema of the output parquet:
    messages           : list[dict]   ([{"role", "content"}, ...])
    gamefile_id        : str
    task_type          : str
    rollout_idx        : int
    won                : bool
    final_reward       : float
    num_turns          : int
    num_invalid_actions: int
    prompt_length      : int
    response_length    : int
    response_loss_ones : int           (= number of trainable tokens)
    model_path         : str
    ckpt_step          : int
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import pandas as pd


def load_record(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _passes_filter(rec: dict, keep: str) -> bool:
    if keep == "all":
        return True
    if keep == "won":
        return bool(rec.get("won"))
    if keep == "losing":
        return not bool(rec.get("won"))
    raise ValueError(f"unknown keep mode: {keep}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_dir", required=True,
                   help="Run dir written by eval_standalone --teacher_rollout_dir.")
    p.add_argument("--out", required=True, help="Output .parquet path.")
    p.add_argument("--keep", choices=["won", "all", "losing"], default="won",
                   help="Filter trajectories. Default: only won=True.")
    p.add_argument("--max_rows", type=int, default=-1,
                   help="If >0, limit output to first N rows (after filter, "
                        "deterministic by sorted file path).")
    args = p.parse_args()

    pattern = os.path.join(args.teacher_dir, "by_task_type", "*", "*.jsonl")
    files = sorted(glob(pattern))
    print(f"found {len(files)} jsonl files under {args.teacher_dir}/by_task_type/")
    if not files:
        sys.exit(f"no JSONL files matched {pattern}")

    rows: list[dict] = []
    n_skipped_bad = 0
    n_skipped_filter = 0
    for path in files:
        rec = load_record(path)
        if rec is None:
            n_skipped_bad += 1
            continue
        if not _passes_filter(rec, args.keep):
            n_skipped_filter += 1
            continue
        lengths = rec.get("lengths", {})
        rows.append({
            "messages": rec.get("messages", []),
            "gamefile_id": rec.get("gamefile_id", "unknown"),
            "task_type": rec.get("task_type", "unknown"),
            "rollout_idx": int(rec.get("rollout_idx", 0)),
            "won": bool(rec.get("won", False)),
            "final_reward": float(rec.get("final_reward", 0.0)),
            "num_turns": int(rec.get("num_turns", 0)),
            "num_invalid_actions": int(rec.get("num_invalid_actions", 0)),
            "prompt_length": int(lengths.get("prompt_length_final", 0)),
            "response_length": int(lengths.get("response_length_final", 0)),
            "response_loss_ones": int(lengths.get("loss_mask_1_count", 0)),
            "model_path": rec.get("model_path", ""),
            "ckpt_step": int(rec.get("ckpt_step", -1)),
        })
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    print(
        f"skipped: bad_files={n_skipped_bad} filtered_out={n_skipped_filter} "
        f"kept={len(rows)}"
    )
    if not rows:
        sys.exit("no rows passed filter — nothing to write")

    df = pd.DataFrame(rows)
    print("dataframe shape:", df.shape)
    print("dtypes:")
    print(df.dtypes)
    print("won breakdown:", df["won"].value_counts(dropna=False).to_dict())
    print("task_type breakdown:", df["task_type"].value_counts().to_dict())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    print(f"wrote {args.out} ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
