"""Partition the fully-won teacher trajectories into 5 task-balanced parts.

Default scheme is ``S5x512_fwon8_prop_v1``:

* Source: rows where ``won == True`` AND the game has all 8 rollouts (idx 0..7)
  winning (i.e. teacher was perfect on that game).
* Parts A, B, C, D each contain ``--part_size`` (default 512) games, with a
  per-task quota proportional to the pool's ``task_type`` distribution.
* Part E contains all remaining games (~380 with the default config).
* Within each task_type, games are shuffled deterministically by ``--seed``.
* Output: ``<out_dir>/{A,B,C,D,E}.parquet`` + ``manifest.json``.

The same ``--seed`` against any source parquet (with or without the
``tokens_*`` columns) produces the same per-game part assignment, so the
pre-tokenized and messages-only flows stay aligned.

Example:
    python -m recipe.alfworld.split_sft_data \\
        --source_parquet /scratch/.../sft_data/qwen3_8b_rl_step570_T0.4_won.parquet \\
        --out_dir       /scratch/.../sft_data/splits/S5x512_fwon8_prop_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


PART_NAMES_DEFAULT = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI",
]


def fully_won_games(df: pd.DataFrame) -> set[str]:
    """Return the set of gamefile_ids whose rollout_idx 0..7 ALL appear in df."""
    needed = {0, 1, 2, 3, 4, 5, 6, 7}
    out: set[str] = set()
    for gid, idxs in df.groupby("gamefile_id")["rollout_idx"].apply(set).items():
        if needed.issubset(idxs):
            out.add(str(gid))
    return out


def compute_quotas(pool_per_task: dict[str, int], part_size: int) -> dict[str, int]:
    """Per-task quota for one fixed-size part. Rounds, then absorbs the
    rounding delta into the most populous task so quotas sum exactly to
    ``part_size`` and stay non-negative.
    """
    total = sum(pool_per_task.values())
    raw = {tt: n / total * part_size for tt, n in pool_per_task.items()}
    quotas = {tt: int(round(v)) for tt, v in raw.items()}
    diff = part_size - sum(quotas.values())
    if diff != 0:
        biggest = max(pool_per_task.keys(), key=lambda k: pool_per_task[k])
        quotas[biggest] += diff
    assert sum(quotas.values()) == part_size, (
        f"quota sum {sum(quotas.values())} != target {part_size}"
    )
    assert all(v >= 0 for v in quotas.values()), f"negative quota: {quotas}"
    return quotas


def assign_parts(
    sub: pd.DataFrame,
    quotas: dict[str, int],
    part_names: list[str],
    n_fixed_parts: int,
    seed: int,
) -> dict[str, str]:
    """Return ``{gamefile_id: part_name}`` for every gid in ``sub``.

    Within each task_type, games are shuffled with a deterministic RNG seeded
    by ``seed``. Parts A..D draw their per-task quota in order; whatever is
    left lands in the final part.
    """
    rng = np.random.default_rng(seed)
    task_to_gids: dict[str, list[str]] = {}
    for tt in sorted(sub["task_type"].unique()):
        gids = sorted(sub[sub["task_type"] == tt]["gamefile_id"].unique().tolist())
        gids = list(map(str, gids))
        rng.shuffle(gids)
        task_to_gids[tt] = gids

    assignment: dict[str, str] = {}
    pos = {tt: 0 for tt in task_to_gids}
    for name in part_names[:n_fixed_parts]:
        for tt in sorted(quotas):
            n = quotas[tt]
            slice_ = task_to_gids[tt][pos[tt] : pos[tt] + n]
            assert len(slice_) == n, (
                f"part {name} task {tt}: pool exhausted "
                f"(want {n}, got {len(slice_)} from pos {pos[tt]} of {len(task_to_gids[tt])})"
            )
            for gid in slice_:
                assignment[gid] = name
            pos[tt] += n

    # Last part: everything that's left
    last = part_names[n_fixed_parts]
    for tt, gids in task_to_gids.items():
        for gid in gids[pos[tt] :]:
            assignment[gid] = last

    return assignment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_parquet", required=True,
                   help="Won-only teacher trajectory parquet (with or without tokens_* cols).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--part_size", type=int, default=512,
                   help="Size of each fixed part (default 512).")
    p.add_argument("--n_fixed_parts", type=int, default=4,
                   help="Number of fixed-size parts; the remainder forms one final part.")
    p.add_argument("--scheme", default="S5x512_fwon8_prop_v1",
                   help="Scheme label written into manifest.json for tracking.")
    args = p.parse_args()

    if args.n_fixed_parts + 1 > len(PART_NAMES_DEFAULT):
        sys.exit(f"too many parts: extend PART_NAMES_DEFAULT in {__file__}")
    part_names = PART_NAMES_DEFAULT[: args.n_fixed_parts + 1]

    print(f"[split] loading {args.source_parquet}", flush=True)
    df = pd.read_parquet(args.source_parquet, engine="pyarrow")
    print(f"[split]   rows={len(df)} cols={list(df.columns)[:6]}{'...' if len(df.columns) > 6 else ''}")

    fully = fully_won_games(df)
    print(f"[split] fully-won games (idx 0..7 all winning): "
          f"{len(fully)} / {df['gamefile_id'].nunique()} unique gids in source")

    sub = df[df["gamefile_id"].isin(fully)].copy()
    print(f"[split] fully-won trajectories in source: {len(sub)}")

    games_per_task = (
        sub.groupby("task_type")["gamefile_id"].nunique().to_dict()
    )
    total_games = sum(games_per_task.values())
    print(f"[split] pool by task_type:")
    for tt in sorted(games_per_task):
        print(f"  {tt:<35} {games_per_task[tt]}")
    print(f"[split]   total = {total_games}")

    quotas = compute_quotas(games_per_task, args.part_size)
    print(f"[split] per-fixed-part quota (sum={sum(quotas.values())} target={args.part_size}):")
    for tt in sorted(quotas):
        print(f"  {tt:<35} {quotas[tt]}")

    assignment = assign_parts(sub, quotas, part_names, args.n_fixed_parts, args.seed)
    assert set(assignment) == fully, "every fully-won gid must get a part"
    print(f"[split] assigned {len(assignment)} games to {len(part_names)} parts")

    os.makedirs(args.out_dir, exist_ok=True)
    sub2 = sub.copy()
    sub2["__part"] = sub2["gamefile_id"].map(assignment)

    print(f"[split] writing parquets to {args.out_dir}")
    part_stats = {}
    for name in part_names:
        part_df = sub2[sub2["__part"] == name].drop(columns=["__part"])
        out_path = os.path.join(args.out_dir, f"{name}.parquet")
        part_df.to_parquet(out_path, engine="pyarrow", index=False)
        n_games = part_df["gamefile_id"].nunique()
        n_traj = len(part_df)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        # task-type breakdown counted at game level (not trajectory level)
        gid_to_task = (
            part_df[["gamefile_id", "task_type"]].drop_duplicates()
            .set_index("gamefile_id")["task_type"]
        )
        bd_games = gid_to_task.value_counts().to_dict()
        bd_traj = part_df["task_type"].value_counts().to_dict()
        part_stats[name] = {
            "games": int(n_games),
            "trajectories": int(n_traj),
            "size_mb": round(size_mb, 2),
            "task_type_games": {k: int(v) for k, v in bd_games.items()},
            "task_type_trajectories": {k: int(v) for k, v in bd_traj.items()},
        }
        print(f"  {name}: games={n_games:>4}  traj={n_traj:>5}  size={size_mb:>6.1f}MB")
        print(f"     task_type (games): {dict(sorted(bd_games.items()))}")

    manifest = {
        "scheme": args.scheme,
        "source_parquet": os.path.abspath(args.source_parquet),
        "seed": args.seed,
        "part_size": args.part_size,
        "n_fixed_parts": args.n_fixed_parts,
        "part_names": part_names,
        "n_total_games_in_pool": total_games,
        "n_total_trajectories_in_pool": int(len(sub)),
        "pool_games_per_task": {k: int(v) for k, v in games_per_task.items()},
        "quotas_per_fixed_part": quotas,
        "part_stats": part_stats,
        "gamefile_assignment": {gid: assignment[gid] for gid in sorted(assignment)},
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[split] manifest: {manifest_path}")
    print(f"[split]   {len(manifest['gamefile_assignment'])} game→part entries")


if __name__ == "__main__":
    main()
