"""DAgger Phase 3: shrink the Phase-2 labeled parquet so its trainable-token
count matches a reference (e.g. the same-bucket direct-SFT parquet).

Why this exists
---------------
The user wants to compare DAgger ("A-trained student rolls on B; teacher
labels each visited state") against direct-SFT-on-B at *equal compute*.
"Equal compute" is operationalized as **equal trainable-token count**
(``sum(response_loss_ones)``).

What this script does
---------------------
1. Read the DAgger parquet (Phase-2 output) and the reference parquet.
2. Compute ``target = sum(reference.response_loss_ones)``.
3. Deterministically shuffle the DAgger rows (seeded), accumulate
   ``response_loss_ones``, and cut off at the first row that pushes the
   running sum past ``target`` (or just under, depending on
   ``--match_policy``).
4. Write the truncated parquet + manifest sidecar.

Output schema is identical to the Phase-2 parquet (subset of rows), so it
plugs straight into the existing SFT trainer via ``PretokenizedSFTDataset``.

Usage:
    python -m recipe.alfworld.dagger_match_token_budget \\
        --dagger_parquet  /scratch/.../sft_data/dagger/<phase2>.parquet \\
        --target_parquet  /scratch/.../sft_data/splits/S35x64_fwon8_prop_v1/B.parquet \\
        --out_parquet     /scratch/.../sft_data/dagger/<phase2>_matched_to_B.parquet \\
        --seed 42 --match_policy first_crossing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def _validate_columns(df: pd.DataFrame, name: str) -> None:
    needed = {"response_loss_ones"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"[match] {name} parquet missing columns: {missing}")


def _shuffle_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def _select_indices(
    cumsum: np.ndarray, target: int, policy: str
) -> int:
    """Return the count of rows to keep so cumsum's last entry crosses target.

    * first_crossing : smallest k such that cumsum[k-1] >= target
    * largest_under  : largest k such that cumsum[k-1] <= target
    """
    if policy == "first_crossing":
        # np.searchsorted with side='left' returns the first idx where
        # cumsum >= target. The corresponding count is idx + 1.
        idx = int(np.searchsorted(cumsum, target, side="left"))
        if idx >= len(cumsum):
            return len(cumsum)
        return idx + 1
    if policy == "largest_under":
        # Largest k with cumsum[k-1] <= target. searchsorted side='right' on
        # target gives the first idx where cumsum > target.
        idx = int(np.searchsorted(cumsum, target, side="right"))
        return idx  # cumsum[idx-1] <= target, cumsum[idx] > target
    raise ValueError(f"unknown match_policy: {policy}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dagger_parquet", required=True,
                   help="Phase-2 output parquet (one row per (gid, turn)).")
    p.add_argument("--target_parquet", required=True,
                   help="Reference parquet whose trainable-token sum is the target.")
    p.add_argument("--out_parquet", required=True,
                   help="Output parquet (subset of dagger_parquet rows).")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic shuffle seed (so the matched subset is "
                        "reproducible).")
    p.add_argument("--match_policy", choices=["first_crossing", "largest_under"],
                   default="first_crossing",
                   help="first_crossing: smallest subset whose token sum >= "
                        "target (default; tends to slightly overshoot). "
                        "largest_under: largest subset whose token sum <= target.")
    p.add_argument("--target_token_override", type=int, default=-1,
                   help="If >0, ignore --target_parquet's sum and use this exact "
                        "trainable-token target instead.")
    args = p.parse_args()

    print(f"[match] dagger_parquet = {args.dagger_parquet}", flush=True)
    print(f"[match] target_parquet = {args.target_parquet}", flush=True)
    print(f"[match] out_parquet    = {args.out_parquet}", flush=True)
    print(f"[match] seed           = {args.seed}", flush=True)
    print(f"[match] policy         = {args.match_policy}", flush=True)

    dagger = pd.read_parquet(args.dagger_parquet, engine="pyarrow")
    _validate_columns(dagger, "dagger")
    print(f"[match] dagger rows={len(dagger)} "
          f"trainable_tokens_total={int(dagger['response_loss_ones'].sum())}",
          flush=True)

    if args.target_token_override > 0:
        target = int(args.target_token_override)
        target_rows = -1
        print(f"[match] using target_token_override = {target}", flush=True)
    else:
        ref = pd.read_parquet(args.target_parquet, engine="pyarrow")
        _validate_columns(ref, "target")
        target = int(ref["response_loss_ones"].sum())
        target_rows = len(ref)
        print(f"[match] target rows={target_rows} trainable_tokens_total={target}",
              flush=True)

    if target <= 0:
        sys.exit(f"[match] target trainable tokens = {target}, nothing to match against")

    if int(dagger["response_loss_ones"].sum()) < target:
        print(
            f"[match] WARN: dagger total {int(dagger['response_loss_ones'].sum())} "
            f"< target {target} -- emitting ALL dagger rows.",
            flush=True,
        )
        kept = dagger.reset_index(drop=True).copy()
    else:
        idx = _shuffle_indices(len(dagger), args.seed)
        shuffled = dagger.iloc[idx].reset_index(drop=True)
        cumsum = shuffled["response_loss_ones"].cumsum().to_numpy()
        keep_n = _select_indices(cumsum, target, args.match_policy)
        kept = shuffled.iloc[:keep_n].reset_index(drop=True)

    kept_tokens = int(kept["response_loss_ones"].sum())
    delta = kept_tokens - target
    pct = (kept_tokens / target * 100.0) if target else 0.0
    print(
        f"[match] kept rows={len(kept)} trainable_tokens={kept_tokens} "
        f"target={target} delta={delta:+d} ({pct:.2f}% of target)",
        flush=True,
    )

    # task-type breakdown for sanity
    if "task_type" in kept.columns:
        print("[match] kept task_type breakdown:", flush=True)
        for t, c in sorted(kept["task_type"].value_counts().to_dict().items()):
            print(f"  {t:<45s} {int(c)}", flush=True)

    if "gamefile_id" in kept.columns:
        n_games = int(kept["gamefile_id"].nunique())
        print(f"[match] kept covers {n_games} unique gamefile_ids "
              f"(out of {dagger['gamefile_id'].nunique()} in dagger pool)",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)) or ".", exist_ok=True)
    kept.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    size_mb = os.path.getsize(args.out_parquet) / (1024 * 1024)
    print(f"[match] wrote {args.out_parquet} ({size_mb:.1f} MB)", flush=True)

    manifest = {
        "phase": "dagger_phase3_token_match",
        "dagger_parquet": os.path.abspath(args.dagger_parquet),
        "target_parquet": os.path.abspath(args.target_parquet),
        "target_token_override": int(args.target_token_override),
        "out_parquet": os.path.abspath(args.out_parquet),
        "seed": int(args.seed),
        "match_policy": args.match_policy,
        "n_dagger_rows_total": int(len(dagger)),
        "n_dagger_tokens_total": int(dagger["response_loss_ones"].sum()),
        "target_tokens": int(target),
        "n_target_rows": int(target_rows),
        "n_kept_rows": int(len(kept)),
        "kept_tokens": int(kept_tokens),
        "kept_token_delta": int(delta),
        "kept_token_pct_of_target": float(pct),
        "kept_unique_gids": int(kept["gamefile_id"].nunique()) if "gamefile_id" in kept.columns and len(kept) else 0,
        "kept_task_type": (
            {t: int(c) for t, c in kept["task_type"].value_counts().to_dict().items()}
            if "task_type" in kept.columns and len(kept) else {}
        ),
        "wrote_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    sidecar = args.out_parquet.replace(".parquet", ".manifest.json")
    with open(sidecar, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[match] manifest -> {sidecar}", flush=True)


if __name__ == "__main__":
    main()
