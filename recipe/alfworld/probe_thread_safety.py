# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone probe: can 20 AlfWorldSingleEnv instances run concurrently via
asyncio.to_thread without TextWorld / alfworld exploding?

Run with:
    cd /scratch/m000069-pm05/miaolu/verl
    python -m recipe.alfworld.probe_thread_safety

Outputs: per-task outcome, concurrent wall time, sequential wall time, and
a THREAD_SAFE / THREAD_UNSAFE / INCONCLUSIVE verdict.
"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from pathlib import Path

# Silence tqdm progress bars from AlfredTWEnv init -- otherwise 20 concurrent
# inits interleave progress output into an unreadable wall of text.
os.environ.setdefault("TQDM_DISABLE", "1")

# --------------------------------------------------------------------------
# Resolve ALFWORLD_DATA BEFORE importing the wrapper (which imports alfworld,
# which reads ALFWORLD_DATA at load time for its game-file discovery).
# --------------------------------------------------------------------------
_DEFAULT_ALFWORLD_DATA = str(Path.home() / ".cache" / "alfworld")
if "ALFWORLD_DATA" not in os.environ:
    if Path(_DEFAULT_ALFWORLD_DATA).is_dir():
        os.environ["ALFWORLD_DATA"] = _DEFAULT_ALFWORLD_DATA
    else:
        raise RuntimeError(
            f"ALFWORLD_DATA env var is unset and default "
            f"{_DEFAULT_ALFWORLD_DATA} does not exist."
        )
print(f"[probe] ALFWORLD_DATA = {os.environ['ALFWORLD_DATA']}")

# Defer the wrapper import until after the env var is set.
from recipe.alfworld.alfworld_env_wrapper import AlfWorldSingleEnv  # noqa: E402

ALF_CONFIG_PATH = str(
    Path(__file__).parent / "config_tw.yaml"
)

NUM_ENVS = 20
STEPS_PER_ENV = 5
# AlfredTWEnv init scans ~8.8k game files (~15s per env). Running 20 of them
# sequentially for the speedup baseline would blow the <5min budget, so we
# only time N_SEQ_SAMPLE of them and extrapolate.
N_SEQ_SAMPLE = 5
# Hard ceiling on the concurrent pass. If TextWorld's shared-global state
# gets corrupted, some inits hang indefinitely in native code past the
# point where Python exception handling can save us. Give up after this
# many seconds and classify with what we have.
CONCURRENT_TIMEOUT_S = 180.0


def _run_one_sync(env_id: int) -> dict:
    """Sync helper used by both the async (via to_thread) and sequential paths."""
    result: dict = {
        "env_id": env_id,
        "reset_ok": False,
        "steps_done": 0,
        "final_won": False,
        "error": None,
        "error_type": None,
    }
    try:
        env = AlfWorldSingleEnv(
            alf_config_path=ALF_CONFIG_PATH,
            seed=1000 + env_id,
            history_length=2,
            is_train=True,
        )
        try:
            obs, adm, info = env.reset()
            result["reset_ok"] = True
            for _ in range(STEPS_PER_ENV):
                if not adm:
                    break
                action = adm[0]  # deterministic: any admissible command
                obs, adm, reward, done, info = env.step(action)
                result["steps_done"] += 1
                if done:
                    break
            result["final_won"] = bool(info.get("won", False))
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 - we want to classify everything
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["error_type"] = type(exc).__name__
        result["traceback"] = traceback.format_exc()
    return result


async def run_one(env_id: int) -> dict:
    r = await asyncio.to_thread(_run_one_sync, env_id)
    tag = "OK " if r.get("error") is None else "ERR"
    print(
        f"[probe] env {env_id:>2d} done [{tag}] steps={r.get('steps_done', 0)} "
        f"err={r.get('error')}",
        flush=True,
    )
    return r


def classify(results: list[dict], t_concurrent: float, t_sequential: float) -> str:
    n_ok = sum(1 for r in results if r.get("error") is None)
    n_err = len(results) - n_ok

    # Any exception == unsafe.
    if n_err > 0:
        return "THREAD_UNSAFE"

    # All succeeded. Speedup test: if concurrency doesn't help AT ALL
    # (concurrent wall time >= sequential), then TextWorld likely holds a
    # global C-level lock. That's still technically "safe" (no errors) but
    # useless for our pool.
    if t_concurrent <= 0 or t_sequential <= 0:
        return "INCONCLUSIVE"

    speedup = t_sequential / t_concurrent
    if speedup < 1.1:
        return "INCONCLUSIVE"  # no parallelism benefit: prefer Ray actors
    return "THREAD_SAFE"


async def main() -> None:
    print(
        f"[probe] launching {NUM_ENVS} concurrent envs via asyncio.to_thread, "
        f"{STEPS_PER_ENV} steps each"
    )

    # Concurrent pass. Use return_exceptions=True defensively: TextWorld's
    # C bindings can raise SystemError / crash in ways that bubble out of the
    # sync helper. Also hard-cap the whole pass with a timeout so we can
    # still classify if some inits hang in native code.
    t0 = time.time()
    tasks = [asyncio.create_task(run_one(i)) for i in range(NUM_ENVS)]
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=CONCURRENT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(
            f"[probe] concurrent pass hit {CONCURRENT_TIMEOUT_S:.0f}s hard timeout; "
            f"some tasks still running",
            flush=True,
        )
    t_concurrent = time.time() - t0

    results_raw: list = []
    for i, tk in enumerate(tasks):
        if tk.done() and not tk.cancelled():
            try:
                results_raw.append(tk.result())
            except BaseException as e:  # noqa: BLE001
                results_raw.append(e)
        else:
            tk.cancel()
            results_raw.append(
                RuntimeError(
                    f"task {i} did not finish within {CONCURRENT_TIMEOUT_S:.0f}s "
                    f"(TextWorld init/step hang)"
                )
            )
    # Normalize: anything that came back as a raw exception is turned into a
    # result dict.
    results: list[dict] = []
    for i, r in enumerate(results_raw):
        if isinstance(r, BaseException):
            results.append(
                {
                    "env_id": i,
                    "reset_ok": False,
                    "steps_done": 0,
                    "final_won": False,
                    "error": f"{type(r).__name__}: {r}",
                    "error_type": type(r).__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(r), r, r.__traceback__)
                    ),
                }
            )
        else:
            results.append(r)
    print(f"[probe] concurrent pass done in {t_concurrent:.2f}s")

    # Per-task summary.
    n_ok = 0
    n_err = 0
    err_types: dict[str, int] = {}
    printed_tb = False
    for r in results:
        if r.get("error") is None:
            n_ok += 1
        else:
            n_err += 1
            err_types[r["error_type"]] = err_types.get(r["error_type"], 0) + 1
            print(f"[probe] env {r['env_id']} ERROR: {r['error']}")
            if not printed_tb and r.get("traceback"):
                # Print ONE traceback (first failure) and keep counting.
                print(r["traceback"].rstrip())
                printed_tb = True

    print("[probe] per-task outcome:")
    for r in results:
        print(
            f"  env_id={r['env_id']:>2d} reset_ok={r['reset_ok']!s:<5} "
            f"steps={r['steps_done']} won={r['final_won']!s:<5} "
            f"err={r['error']}"
        )

    # Sequential pass for comparison (ONLY if all concurrent succeeded;
    # otherwise speedup is meaningless). We time a small sample of N_SEQ_SAMPLE
    # runs and extrapolate to NUM_ENVS -- each wrapper init scans ~8.8k game
    # files which is ~15s, and running 20 sequentially would blow the budget.
    t_sequential = -1.0
    t_seq_measured = -1.0
    if n_err == 0:
        print(
            f"[probe] running sequential pass ({N_SEQ_SAMPLE} runs, extrapolated) "
            f"for speedup comparison..."
        )
        t0 = time.time()
        seq_results = [_run_one_sync(100 + i) for i in range(N_SEQ_SAMPLE)]
        t_seq_measured = time.time() - t0
        n_seq_ok = sum(1 for r in seq_results if r.get("error") is None)
        # Linear extrapolation to NUM_ENVS.
        t_sequential = t_seq_measured * (NUM_ENVS / N_SEQ_SAMPLE)
        print(
            f"[probe] sequential pass: {N_SEQ_SAMPLE} runs in {t_seq_measured:.2f}s "
            f"(ok={n_seq_ok}/{N_SEQ_SAMPLE}); extrapolated to {NUM_ENVS} runs: "
            f"{t_sequential:.2f}s"
        )

    verdict = classify(results, t_concurrent, t_sequential)

    print("=" * 64)
    print(f"[probe] RESULT: {n_ok}/{NUM_ENVS} ok, {n_err} errored")
    if err_types:
        print(f"[probe] error types: {err_types}")
    print(f"[probe] concurrent wall time : {t_concurrent:.2f}s")
    print(f"[probe] sequential wall time : {t_sequential:.2f}s")
    if t_concurrent > 0 and t_sequential > 0:
        print(f"[probe] speedup              : {t_sequential / t_concurrent:.2f}x")
    print(f"[probe] VERDICT              : {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
