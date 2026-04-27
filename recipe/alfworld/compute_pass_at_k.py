"""Compute pass@1..pass@k from an alfworld eval dump.

Reads the per-episode JSONL files under ``<dump_dir>/step_*/by_task_type/<task>/<gid>__rollout_<n>.jsonl``,
groups by ``gamefile_id``, and computes pass@k for k=1..K using the standard
HumanEval estimator::

    pass@k = 1 - C(n - c, k) / C(n, k)

where n = total rollouts per game and c = number of successful rollouts.
The reported ``pass@k`` is the mean of per-game ``pass@k`` over all games
that have exactly ``--rollouts_per_game`` rollouts (games with fewer
rollouts are dropped with a warning).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def gather(dump_dir: str) -> dict[str, dict]:
    """Return {gid: {"task_type": tt, "outcomes": [bool, ...]}}."""
    by_gid: dict[str, dict] = {}
    n_files = 0
    for root, _, files in os.walk(dump_dir):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            n_files += 1
            with open(os.path.join(root, fn)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    gid = rec.get("gamefile_id")
                    if gid is None:
                        continue
                    won = bool(rec.get("won"))
                    tt = rec.get("task_type", "unknown")
                    g = by_gid.setdefault(gid, {"task_type": tt, "outcomes": []})
                    g["outcomes"].append(won)
    print(f"[pass@k] scanned {n_files} jsonl files, {len(by_gid)} unique games", flush=True)
    return by_gid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--out_json", default=None,
                   help="Where to write the summary. Default: <dump_dir>/pass_at_k.json")
    p.add_argument("--rollouts_per_game", type=int, default=8,
                   help="Expected rollouts per game; games with < this are dropped.")
    args = p.parse_args()

    if args.out_json is None:
        args.out_json = os.path.join(args.dump_dir, "pass_at_k.json")

    by_gid = gather(args.dump_dir)
    if not by_gid:
        print(f"[pass@k] no episodes found under {args.dump_dir}", file=sys.stderr)
        sys.exit(2)

    n = args.rollouts_per_game
    games = []
    skipped = 0
    for gid, g in by_gid.items():
        if len(g["outcomes"]) != n:
            skipped += 1
            continue
        c = sum(g["outcomes"])
        games.append({"gamefile_id": gid, "task_type": g["task_type"], "n": n, "c": c})
    if skipped:
        print(f"[pass@k] WARNING: skipped {skipped} games with != {n} rollouts", flush=True)

    n_games = len(games)
    if n_games == 0:
        print(f"[pass@k] no games with exactly n={n} rollouts", file=sys.stderr)
        sys.exit(3)

    overall: dict[str, float] = {}
    for k in range(1, n + 1):
        vals = [pass_at_k(g["n"], g["c"], k) for g in games]
        overall[f"pass@{k}"] = sum(vals) / n_games

    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    by_task_count: dict[str, int] = defaultdict(int)
    for k in range(1, n + 1):
        agg: dict[str, list[float]] = defaultdict(list)
        for g in games:
            agg[g["task_type"]].append(pass_at_k(g["n"], g["c"], k))
        for tt, vals in agg.items():
            by_task[tt][f"pass@{k}"] = sum(vals) / len(vals)
            by_task_count[tt] = len(vals)

    summary = {
        "dump_dir": args.dump_dir,
        "rollouts_per_game": n,
        "n_games": n_games,
        "n_skipped": skipped,
        "overall": overall,
        "by_task_type": {
            tt: {
                "n_games": by_task_count[tt],
                **by_task[tt],
            } for tt in sorted(by_task)
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[pass@k] {n_games} games × n={n} rollouts:")
    for k in range(1, n + 1):
        print(f"  pass@{k}: {overall[f'pass@{k}']:.4f}")
    print(f"[pass@k] summary written to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
