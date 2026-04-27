"""Convert teacher-rollout JSONL trajectories to a verl-SFT parquet.

Walks a teacher-rollout directory tree (``<root>/by_task_type/<task>/<id>__rollout_<n>.jsonl``)
and emits one parquet whose rows are ready to feed into
``verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`` -- i.e. each
row has at minimum a ``messages`` column (list of ``{"role", "content"}``)
plus a few filterable metadata columns.

By default we keep only "good" rollouts (won=True). Override with
``--keep all`` or ``--keep losing`` if you want everything / only failures.

Usage:
    python -m recipe.alfworld.jsonl_to_parquet \\
        --teacher_dir /scratch/.../teacher_rollouts/qwen3_8b_rl_step570 \\
        --out /scratch/.../sft_data/qwen3_8b_rl_step570_won.parquet \\
        --keep won \\
        --max_length_filter 8192 \\
        --enable_thinking false

Schema of the output parquet:
    messages           : list[dict]   ([{"role", "content"}, ...])
    enable_thinking    : bool         (per-row chat-template flag for Qwen3)
    tools              : list[dict]   (empty -- present for verl compatibility)
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
    pretok_input_len   : int           (length of pre-tokenized tokens.input_ids)
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


def _str_to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


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
    p.add_argument("--max_length_filter", type=int, default=-1,
                   help="If >0, drop trajectories whose pre-tokenized "
                        "tokens.input_ids length exceeds this (avoids OOM later).")
    p.add_argument("--enable_thinking", type=_str_to_bool, default=False,
                   help="Value to write into the per-row enable_thinking column. "
                        "Set False for no-think rollouts (default), True for "
                        "thinking-mode rollouts.")
    p.add_argument("--include_tokens", type=_str_to_bool, default=True,
                   help="Include pre-tokenized tokens_input_ids/tokens_loss_mask "
                        "columns (list[int]) for direct SFT consumption via "
                        "PretokenizedSFTDataset. Bloats parquet ~10x. "
                        "Pass --include_tokens false for messages-only output.")
    args = p.parse_args()

    pattern = os.path.join(args.teacher_dir, "by_task_type", "*", "*.jsonl")
    files = sorted(glob(pattern))
    print(f"found {len(files)} jsonl files under {args.teacher_dir}/by_task_type/")
    if not files:
        sys.exit(f"no JSONL files matched {pattern}")

    rows: list[dict] = []
    n_skipped_bad = 0
    n_skipped_filter = 0
    n_skipped_length = 0
    for path in files:
        rec = load_record(path)
        if rec is None:
            n_skipped_bad += 1
            continue
        if not _passes_filter(rec, args.keep):
            n_skipped_filter += 1
            continue
        pretok_len = len(rec.get("tokens", {}).get("input_ids", []) or [])
        if args.max_length_filter > 0 and pretok_len > args.max_length_filter:
            n_skipped_length += 1
            continue
        lengths = rec.get("lengths", {})
        row = {
            "messages": rec.get("messages", []),
            "tools": [],
            "enable_thinking": bool(args.enable_thinking),
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
            "pretok_input_len": pretok_len,
            "model_path": rec.get("model_path", ""),
            "ckpt_step": int(rec.get("ckpt_step", -1)),
        }
        if args.include_tokens:
            tok = rec.get("tokens", {}) or {}
            row["tokens_input_ids"] = [int(x) for x in (tok.get("input_ids") or [])]
            row["tokens_loss_mask"] = [int(x) for x in (tok.get("loss_mask") or [])]
            if len(row["tokens_input_ids"]) != len(row["tokens_loss_mask"]):
                print(f"[skip] {path}: tokens len mismatch input_ids={len(row['tokens_input_ids'])} "
                      f"loss_mask={len(row['tokens_loss_mask'])}", file=sys.stderr)
                continue
        rows.append(row)
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break

    print(
        f"skipped: bad_files={n_skipped_bad} filtered_out={n_skipped_filter} "
        f"too_long={n_skipped_length} kept={len(rows)}"
    )
    if not rows:
        sys.exit("no rows passed filter -- nothing to write")

    df = pd.DataFrame(rows)
    print("dataframe shape:", df.shape)
    print("won breakdown:", df["won"].value_counts(dropna=False).to_dict())
    print("task_type breakdown:", df["task_type"].value_counts().to_dict())
    print("pretok_input_len pct: p50={p50} p95={p95} max={mx}".format(
        p50=int(df["pretok_input_len"].quantile(0.5)),
        p95=int(df["pretok_input_len"].quantile(0.95)),
        mx=int(df["pretok_input_len"].max()),
    ))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)
    print(f"wrote {args.out} ({df.shape[0]} rows)")


if __name__ == "__main__":
    main()
