# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Async stress test for the Ray-actor ALFWorld env pool.

Runs N concurrent episodes across a pool of size P and verifies that none
of them error out (which would indicate parser-global corruption).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback

import ray

from recipe.alfworld.alfworld_env_pool import AlfWorldEnvPool


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config_tw.yaml",
)

POOL_SIZE = int(os.environ.get("ALF_POOL_SIZE", "4"))
N_EPISODES = int(os.environ.get("ALF_N_EPISODES", "20"))
MAX_STEPS_PER_EP = int(os.environ.get("ALF_MAX_STEPS", "5"))


async def _run_episode(pool: AlfWorldEnvPool, ep_id: int) -> dict:
    actor = await pool.acquire()
    t0 = time.time()
    try:
        obs, adm, info = await actor.reset.remote()
        done = False
        reward_sum = 0.0
        steps = 0
        for _ in range(MAX_STEPS_PER_EP):
            if not adm:
                break
            action = adm[0]  # pick first admissible (deterministic)
            obs, adm, r, done, info = await actor.step.remote(action)
            reward_sum += float(r)
            steps += 1
            if done:
                break
        return {
            "ep_id": ep_id,
            "ok": True,
            "steps": steps,
            "reward": reward_sum,
            "won": bool(info.get("won", False)),
            "wall_s": time.time() - t0,
        }
    except Exception as e:  # pragma: no cover - surfaced to stress report
        return {
            "ep_id": ep_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "wall_s": time.time() - t0,
        }
    finally:
        await pool.release(actor)


async def main() -> int:
    print(f"[cfg] CONFIG_PATH={CONFIG_PATH}")
    print(f"[cfg] POOL_SIZE={POOL_SIZE} N_EPISODES={N_EPISODES} "
          f"MAX_STEPS_PER_EP={MAX_STEPS_PER_EP}")
    print(f"[cfg] ALFWORLD_DATA={os.environ.get('ALFWORLD_DATA')}")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=16)
    print(f"[ray] available_resources={ray.available_resources()}")

    t_warm0 = time.time()
    pool = AlfWorldEnvPool(
        alf_config_path=CONFIG_PATH,
        pool_size=POOL_SIZE,
        seed_base=1000,
        is_train=True,
    )
    warmup_s = time.time() - t_warm0
    print(f"[pool] warmup: pool_size={POOL_SIZE} in {warmup_s:.1f}s "
          f"(available={pool.available})")

    try:
        t_run0 = time.time()
        tasks = [
            asyncio.create_task(_run_episode(pool, i)) for i in range(N_EPISODES)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        run_s = time.time() - t_run0

        n_ok = sum(1 for r in results if r["ok"])
        n_fail = N_EPISODES - n_ok
        wons = sum(1 for r in results if r.get("won"))
        mean_steps = sum(r.get("steps", 0) for r in results if r["ok"]) / max(n_ok, 1)

        print("\n=== RESULTS ===")
        print(f"wall-clock (episodes only): {run_s:.1f}s "
              f"({N_EPISODES} episodes / pool {POOL_SIZE})")
        print(f"successes: {n_ok}/{N_EPISODES}   wins: {wons}")
        print(f"avg steps/ep (ok only): {mean_steps:.2f}")
        if n_fail:
            print("--- first few failures ---")
            for r in results:
                if not r["ok"]:
                    print(f"  ep {r['ep_id']}: {r['error']}")
                    print(r["traceback"])
                    break  # just show the first
        print(f"\nwarmup={warmup_s:.1f}s, run={run_s:.1f}s, "
              f"total={warmup_s + run_s:.1f}s")
        return 0 if n_fail == 0 else 1
    finally:
        try:
            await pool.close_all()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
