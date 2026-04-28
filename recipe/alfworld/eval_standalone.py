"""Standalone ALFWorld eval that REUSES upstream verl's AlfWorldAgentLoop +
AlfWorldEnvPool, but bypasses FSDP/ray_trainer/main_ppo to avoid the memory
spike we hit on Qwen3-32B under the full PPO wiring.

Architecture
------------
    StandaloneServerManager (wraps a single vllm AsyncLLM engine, exposes the
    `generate()` async interface that AgentLoopBase expects; concurrent
    callers are batched by vLLM's continuous batcher)
        |
        v
    AlfWorldAgentLoop (unmodified; from recipe.alfworld.alfworld_agent_loop)
        |
        v
    AlfWorldEnvPool ray-actor pool (unmodified; held via module-level cache
    inside AlfWorldAgentLoop)

We instantiate AlfWorldAgentLoop directly, skipping hydra/AgentLoopWorker, and
drive ``--concurrency`` episodes in parallel via an asyncio.Queue worker pool.
All state-machine / projection / trajectory-dumper logic is inherited verbatim.

Usage:
    python -m recipe.alfworld.eval_standalone \
        --model_path Qwen/Qwen3-32B \
        --tp 4 \
        --pool_size 8 --concurrency 8 \
        --alf_config_path /scratch/m000069-pm05/miaolu/verl/recipe/alfworld/config_tw.yaml \
        --dump_dir /scratch/m000069-pm05/miaolu/verl/trajectories/qwen3_32b_eval_YYYYMMDD
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any

from omegaconf import OmegaConf


TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def _task_type_from_gamefile(gamefile: str | None) -> str | None:
    if not gamefile:
        return None
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return None


# ----------------------------------------------------------------------
# StandaloneServerManager: wraps a local vllm AsyncLLM to match the subset of
# AsyncLLMServerManager.generate() that AlfWorldAgentLoop calls.
# ----------------------------------------------------------------------
class StandaloneServerManager:
    """Minimal stand-in for verl's AsyncLLMServerManager.

    AlfWorldAgentLoop calls ``await self.server_manager.generate(
    request_id=..., prompt_ids=..., sampling_params=...)`` and expects a
    ``TokenOutput`` back. We back this with a single in-process vllm
    ``AsyncLLM`` engine, which supports many concurrent ``generate(...)`` calls
    that vLLM's continuous batcher fuses into a single batch. No locking
    needed: each call submits an independent request keyed by ``request_id``.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 4,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = True,
        dtype: str = "bfloat16",
    ):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        print(
            f"[StandaloneServerManager] Loading vllm AsyncLLM model={model_path} "
            f"tp={tensor_parallel_size} max_model_len={max_model_len} "
            f"gpu_util={gpu_memory_utilization} enforce_eager={enforce_eager} dtype={dtype}",
            flush=True,
        )
        t0 = time.time()
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            trust_remote_code=True,
            disable_log_stats=True,
            enable_log_requests=False,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)
        print(
            f"[StandaloneServerManager] vllm AsyncLLM loaded in {time.time() - t0:.1f}s",
            flush=True,
        )
        # Per-call-suffix counter so concurrent calls sharing the same
        # AlfWorldAgentLoop request_id (sticky session) still submit unique
        # vllm engine request ids.
        self._req_seq = 0
        self._req_seq_lock = asyncio.Lock()

    async def _next_engine_request_id(self, base: str) -> str:
        async with self._req_seq_lock:
            self._req_seq += 1
            n = self._req_seq
        return f"{base}-{n}"

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        **_kwargs,
    ):
        from vllm import SamplingParams as VLSP
        from vllm import TokensPrompt

        from verl.workers.rollout.replica import TokenOutput

        sp_kwargs = dict(
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            repetition_penalty=float(sampling_params.get("repetition_penalty", 1.0)),
            max_tokens=int(sampling_params.get("max_tokens", 4096)),
        )
        logprobs_flag = sampling_params.get("logprobs", False)
        if logprobs_flag:
            sp_kwargs["logprobs"] = 0  # 0 = only the selected token's logprob
        vllm_sp = VLSP(**sp_kwargs)

        prompt = TokensPrompt(prompt_token_ids=list(prompt_ids))

        # Multiple concurrent assistant turns may share the same agent-loop
        # request_id; vllm requires unique engine request ids per inflight
        # request, so suffix with a monotonic counter.
        engine_request_id = await self._next_engine_request_id(str(request_id))

        final_output = None
        async for out in self.engine.generate(
            prompt=prompt,
            sampling_params=vllm_sp,
            request_id=engine_request_id,
        ):
            final_output = out
        if final_output is None:
            raise RuntimeError(f"vllm AsyncLLM yielded no outputs for request {engine_request_id}")
        gen = final_output.outputs[0]

        log_probs_list = None
        if logprobs_flag and getattr(gen, "logprobs", None):
            try:
                log_probs_list = [next(iter(d.values())).logprob for d in gen.logprobs]
            except Exception:
                log_probs_list = None

        return TokenOutput(
            token_ids=list(gen.token_ids),
            log_probs=log_probs_list,
            num_preempted=None,
            stop_reason=getattr(gen, "finish_reason", None),
            extra_fields={},
        )


# ----------------------------------------------------------------------
# Minimal OmegaConf stub that AgentLoopBase.__init__ walks through.
# ----------------------------------------------------------------------
def _build_minimal_config(
    model_path: str,
    prompt_length: int,
    response_length: int,
    max_assistant_turns: int,
    max_user_turns: int,
):
    """Build the smallest OmegaConf dict that AgentLoopBase.__init__ will read.

    Touched fields (see verl/experimental/agent_loop/agent_loop.py:297-316):
    * config.actor_rollout_ref.rollout (whole block for `_get_rollout_and_model_config`)
    * config.data.apply_chat_template_kwargs (empty dict is the default)
    """
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": int(prompt_length),
                    "response_length": int(response_length),
                    "multi_turn": {
                        "enable": True,
                        "max_assistant_turns": int(max_assistant_turns),
                        "max_user_turns": int(max_user_turns),
                    },
                    # AgentLoopBase doesn't hit these, but AlfWorldAgentLoop
                    # keeps self.rollout_config as an OmegaConf node and may
                    # .get() trace fields via the rollout_trace decorator.
                    "trace": {
                        "project_name": None,
                        "experiment_name": None,
                        "backend": None,
                        "token2text": False,
                        "max_samples_per_step_per_worker": None,
                    },
                    "calculate_log_probs": False,
                },
                "model": {
                    "path": str(model_path),
                },
            },
            "data": {
                # Empty dict keeps the default Qwen3 chat template behaviour.
                "apply_chat_template_kwargs": {},
            },
        }
    )
    return cfg


# ----------------------------------------------------------------------
# Main async eval driver
# ----------------------------------------------------------------------
async def _run_eval(args):
    import ray
    from transformers import AutoTokenizer

    # Pre-initialize ray with a sane CPU count so AlfWorldEnvPool's
    # `ray.init(ignore_reinit_error=True)` fallback doesn't spawn with defaults.
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=int(args.num_cpus),
            include_dashboard=False,
            log_to_driver=True,
        )

    # Build verl trainer-config stub used by AgentLoopBase.
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    trainer_cfg = _build_minimal_config(
        model_path=args.model_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        max_assistant_turns=args.max_assistant_turns,
        max_user_turns=args.max_user_turns,
    )

    # vllm engine wrapper.
    server_manager = StandaloneServerManager(
        model_path=args.model_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=bool(args.enforce_eager),
        dtype=args.dtype,
    )

    # Tokenizer for decoding and chat template. trust_remote_code isn't
    # required for Qwen3 but keep it on for forward compat.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Dump dir (the AlfWorldAgentLoop picks this up from env via os.environ).
    if args.dump_dir:
        os.environ["ALFWORLD_TRAJ_DUMP_DIR"] = args.dump_dir
        os.makedirs(args.dump_dir, exist_ok=True)
        print(f"[eval] ALFWORLD_TRAJ_DUMP_DIR={args.dump_dir}", flush=True)

    # Teacher rollout mode: set TEACHER_ROLLOUT_RUN_DIR + per-trajectory
    # metadata env vars BEFORE we import AlfWorldAgentLoop (so its dumper
    # picks them up at __init__). Also auto-extract ckpt_step from the
    # model_path when not explicitly provided.
    if args.teacher_rollout_dir:
        os.makedirs(args.teacher_rollout_dir, exist_ok=True)
        os.environ["TEACHER_ROLLOUT_RUN_DIR"] = args.teacher_rollout_dir
        os.environ["TEACHER_ROLLOUT_MODEL_PATH"] = args.model_path
        if args.ckpt_step >= 0:
            ckpt_step = args.ckpt_step
        else:
            m = re.search(r"global_step_(\d+)", args.model_path)
            ckpt_step = int(m.group(1)) if m else -1
        os.environ["TEACHER_ROLLOUT_CKPT_STEP"] = str(ckpt_step)
        print(
            f"[eval] TEACHER_ROLLOUT_RUN_DIR={args.teacher_rollout_dir} "
            f"ckpt_step={ckpt_step}",
            flush=True,
        )

    # Import AFTER env var is set so the loop reads it on __init__.
    from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop
    from recipe.alfworld.alfworld_dataset import AlfWorldDataset

    loop = AlfWorldAgentLoop(
        trainer_config=DictConfigWrap(config=trainer_cfg),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=AlfWorldDataset,  # class, not instance — unused for text-only
        data_config=DictConfigWrap(config=trainer_cfg.data),
        alf_config_path=args.alf_config_path,
        pool_size=args.pool_size,
        seed_base=args.seed_base,
        history_length=args.history_length,
        is_train=False,
        max_steps=args.max_steps,
    )

    # Build dataset purely for enumerating gamefiles/extra_info.
    ds = AlfWorldDataset(
        config=OmegaConf.create({"alfworld": {"split": args.split}}),
        tokenizer=tokenizer,
        split=args.split,
        alf_config_path=args.alf_config_path,
    )
    print(
        f"[eval] split={args.split} path={ds.split_path} "
        f"num_gamefiles={len(ds)}",
        flush=True,
    )
    if args.max_samples > 0:
        num_unique = min(args.max_samples, len(ds))
    else:
        num_unique = len(ds)

    # Optional gamefile-id whitelist (used by DAgger Phase 1 to roll the
    # student only on a specific bucket like S35x64 part B).
    from recipe.alfworld.alfworld_agent_loop import _extract_gamefile_id as _gid
    if args.gamefile_filter_json:
        with open(args.gamefile_filter_json) as _f:
            _payload = json.load(_f)
        # Accept either {"gamefile_ids": [...]} or a flat list.
        whitelist = set(_payload["gamefile_ids"] if isinstance(_payload, dict) else _payload)
        unique_indices = [
            i for i in range(num_unique)
            if _gid(ds[i]["extra_info"].get("gamefile", "")) in whitelist
        ]
        present = {_gid(ds[i]["extra_info"].get("gamefile", "")) for i in unique_indices}
        missing = whitelist - present
        print(
            f"[eval] gamefile_filter_json={args.gamefile_filter_json} "
            f"whitelist={len(whitelist)} matched={len(unique_indices)} "
            f"missing={len(missing)}",
            flush=True,
        )
        if missing:
            print(f"[eval]   missing gids (first 5): {sorted(missing)[:5]}", flush=True)
        if not unique_indices:
            raise RuntimeError(
                f"gamefile_filter_json {args.gamefile_filter_json} matched 0 games "
                f"in split {args.split} -- check the split argument."
            )
    else:
        unique_indices = list(range(num_unique))

    rollouts_per_game = max(1, int(args.rollouts_per_game))
    num_episodes = len(unique_indices) * rollouts_per_game
    print(
        f"[eval] num_unique_games={len(unique_indices)} rollouts_per_game={rollouts_per_game} "
        f"total_episodes={num_episodes}",
        flush=True,
    )

    # Sampling params for eval — matches T8.5 val_kwargs flow.
    sampling_params = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        logprobs=False,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens_per_turn,
    )
    print(f"[eval] sampling_params={sampling_params}", flush=True)

    # In teacher rollout mode, write run-level metadata.json once at the
    # very start so even partial runs are diagnosable.
    if args.teacher_rollout_dir:
        meta_path = os.path.join(args.teacher_rollout_dir, "metadata.json")
        meta_payload = {
            "run_dir": args.teacher_rollout_dir,
            "model_path": args.model_path,
            "ckpt_step": int(os.environ.get("TEACHER_ROLLOUT_CKPT_STEP", "-1")),
            "split": args.split,
            "num_unique_games": int(num_unique),
            "rollouts_per_game": int(rollouts_per_game),
            "total_episodes_planned": int(num_episodes),
            "sampling_params": sampling_params,
            "tp": int(args.tp),
            "concurrency": int(getattr(args, "concurrency", None) or args.pool_size),
            "pool_size": int(args.pool_size),
            "seed_base": int(args.seed_base),
            "max_steps": int(args.max_steps),
            "max_assistant_turns": int(args.max_assistant_turns),
            "max_user_turns": int(args.max_user_turns),
            "max_tokens_per_turn": int(args.max_tokens_per_turn),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "args": {k: v for k, v in vars(args).items()},
        }
        try:
            with open(meta_path, "w") as f:
                json.dump(meta_payload, f, indent=2, default=str)
            print(f"[eval] wrote metadata: {meta_path}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[eval] metadata write FAILED: {e}", flush=True)

    # Run episodes concurrently against the shared AsyncLLM engine. Each
    # worker pulls episode indices from a shared queue, runs one full
    # AlfWorldAgentLoop episode (acquiring an env actor from the ray pool
    # internally), and appends the result. vLLM's continuous batching fuses
    # the per-turn requests across workers; the env-actor pool serializes
    # access to the underlying alfworld games. For teacher rollout we push
    # rollouts_per_game copies of each index — the agent_loop dumper picks
    # the rollout_idx from filesystem state so we don't need to track it
    # here.
    results: list[dict] = []
    results_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()

    # Skip-resume support (teacher rollout only): scan existing
    # by_task_type/<task>/<gid>__rollout_<n>.jsonl files, count per gamefile_id,
    # and only enqueue the games that still need more rollouts. For non-teacher
    # mode (RL training, val) we don't dedup -- those legacy paths are still
    # request_id-keyed and re-running is the expected behavior.
    if args.teacher_rollout_dir:
        existing_per_gid: dict[str, int] = {}
        bt_root = os.path.join(args.teacher_rollout_dir, "by_task_type")
        if os.path.isdir(bt_root):
            for task_dir in os.listdir(bt_root):
                tdir = os.path.join(bt_root, task_dir)
                if not os.path.isdir(tdir):
                    continue
                for fname in os.listdir(tdir):
                    if not fname.endswith(".jsonl"):
                        continue
                    # e.g. "T20190907_174746_989712__rollout_3.jsonl"
                    if "__rollout_" not in fname:
                        continue
                    gid = fname.split("__rollout_", 1)[0]
                    existing_per_gid[gid] = existing_per_gid.get(gid, 0) + 1

        n_already_done = sum(min(c, rollouts_per_game) for c in existing_per_gid.values())
        n_remaining = 0
        for i in unique_indices:
            gid = _gid(
                ds[i]["extra_info"].get("gamefile", "") if "gamefile" in ds[i]["extra_info"] else ""
            )
            need = max(0, rollouts_per_game - existing_per_gid.get(gid, 0))
            for _ in range(need):
                queue.put_nowait(i)
                n_remaining += 1
        print(
            f"[eval] resume scan: existing_per_gid={len(existing_per_gid)} games, "
            f"already_done≈{n_already_done}, queueing {n_remaining} remaining episodes "
            f"(was planning {num_episodes})",
            flush=True,
        )
        num_episodes = n_remaining
    else:
        for _ in range(rollouts_per_game):
            for i in unique_indices:
                queue.put_nowait(i)

    concurrency = max(1, int(getattr(args, "concurrency", None) or args.pool_size))
    for _ in range(concurrency):
        queue.put_nowait(None)  # sentinel per worker
    print(
        f"[eval] launching {concurrency} concurrent workers over {num_episodes} episodes",
        flush=True,
    )

    if num_episodes == 0:
        print("[eval] nothing to do — all rollouts already exist. Exiting.", flush=True)
        return {"num_episodes": 0, "wins": 0, "success_rate": 0.0, "results": []}

    t_start = time.time()

    async def _worker(worker_id: int):
        while True:
            i = await queue.get()
            if i is None:
                queue.task_done()
                return
            row = ds[i]
            row_kwargs = {
                "raw_prompt": row["raw_prompt"],
                "extra_info": row["extra_info"],
                "index": row["index"],
                "tools_kwargs": row["tools_kwargs"],
                "interaction_kwargs": row["interaction_kwargs"],
                "agent_name": row["agent_name"],
            }
            ep_t0 = time.time()
            try:
                output = await loop.run(sampling_params, **row_kwargs)
                won = bool(output.extra_fields.get("won", False))
                num_turns = int(output.num_turns)
                num_invalid = int(output.extra_fields.get("num_invalid_actions", 0))
                env_steps = int(output.extra_fields.get("env_steps", 0))
                gamefile = output.extra_fields.get("gamefile") or row["extra_info"]["gamefile"]
            except Exception as e:
                print(
                    f"[eval] FAIL episode idx={i} worker={worker_id} "
                    f"gamefile={row['extra_info']['gamefile']}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                won = False
                num_turns = 0
                num_invalid = 0
                env_steps = 0
                gamefile = row["extra_info"]["gamefile"]

            elapsed = time.time() - ep_t0
            async with results_lock:
                results.append(
                    {
                        "gamefile": gamefile,
                        "task_type": _task_type_from_gamefile(gamefile),
                        "won": won,
                        "num_turns": num_turns,
                        "num_invalid_actions": num_invalid,
                        "env_steps": env_steps,
                    }
                )
                done_so_far = len(results)
                cur_wins = sum(1 for r in results if r["won"])
            total_elapsed = time.time() - t_start
            print(
                f"[eval] [{done_so_far}/{num_episodes}] (idx={i} worker={worker_id}) "
                f"won={won} turns={num_turns} invalid={num_invalid} "
                f"ep_time={elapsed:.1f}s "
                f"running_sr={cur_wins}/{done_so_far}={cur_wins/done_so_far:.3f} "
                f"total={total_elapsed:.1f}s",
                flush=True,
            )
            queue.task_done()

    workers = [asyncio.create_task(_worker(w)) for w in range(concurrency)]
    await asyncio.gather(*workers)

    # ----------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------
    total = len(results)
    wins = sum(1 for r in results if r["won"])
    total_steps = sum(r["env_steps"] for r in results)
    total_invalid = sum(r["num_invalid_actions"] for r in results)
    sr = wins / total if total else 0.0
    mean_steps = total_steps / total if total else 0.0

    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})
    for r in results:
        t = r["task_type"] or "unknown"
        by_type[t]["total"] += 1
        if r["won"]:
            by_type[t]["wins"] += 1

    print("=" * 70, flush=True)
    print(
        f"[eval] model={args.model_path} split={args.split} "
        f"wins={wins}/{total} success_rate={sr:.4f}",
        flush=True,
    )
    print(
        f"[eval] mean_env_steps={mean_steps:.2f} total_invalid_actions={total_invalid}",
        flush=True,
    )
    for t in sorted(by_type):
        s = by_type[t]
        subrate = s["wins"] / s["total"] if s["total"] else 0.0
        print(
            f"[eval]   task={t:45s} wins={s['wins']:3d}/{s['total']:<3d} "
            f"sr={subrate:.3f}",
            flush=True,
        )
    print("=" * 70, flush=True)

    # Dump summary JSON next to the per-episode JSONL dumper output.
    summary = {
        "model_path": args.model_path,
        "split": args.split,
        "num_episodes": total,
        "wins": wins,
        "success_rate": sr,
        "mean_env_steps": mean_steps,
        "total_invalid_actions": total_invalid,
        "by_task_type": {
            t: {
                "wins": v["wins"],
                "total": v["total"],
                "success_rate": v["wins"] / v["total"] if v["total"] else 0.0,
            }
            for t, v in by_type.items()
        },
        "sampling_params": sampling_params,
        "args": {k: v for k, v in vars(args).items()},
        "results": results,
    }
    if args.summary_path:
        summary_target = args.summary_path
    elif args.teacher_rollout_dir:
        summary_target = os.path.join(args.teacher_rollout_dir, "summary.json")
    elif args.dump_dir:
        summary_target = os.path.join(args.dump_dir, "summary.json")
    else:
        summary_target = ""
    if summary_target:
        try:
            with open(summary_target, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[eval] summary written to {summary_target}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[eval] summary write FAILED: {e}", flush=True)

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bfloat16")

    p.add_argument(
        "--alf_config_path",
        type=str,
        default="/scratch/m000069-pm05/miaolu/verl/recipe/alfworld/config_tw.yaml",
    )
    p.add_argument("--pool_size", type=int, default=8)
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Number of concurrent agent_loop episodes against the shared "
             "vllm AsyncLLM engine. 0 = match --pool_size.",
    )
    p.add_argument("--seed_base", type=int, default=1042)
    p.add_argument("--history_length", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=50)

    p.add_argument("--split", type=str, default="valid_seen",
                   choices=["valid_seen", "valid_unseen", "train"])
    p.add_argument("--max_samples", type=int, default=-1)

    p.add_argument("--prompt_length", type=int, default=4096)
    p.add_argument("--response_length", type=int, default=12288)
    p.add_argument("--max_assistant_turns", type=int, default=50)
    p.add_argument("--max_user_turns", type=int, default=50)
    p.add_argument("--max_tokens_per_turn", type=int, default=4096)

    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)

    p.add_argument("--dump_dir", type=str, default="")
    p.add_argument("--summary_path", type=str, default="")
    p.add_argument("--num_cpus", type=int, default=8)

    # Teacher-rollout mode: when --teacher_rollout_dir is set, the
    # AlfWorldAgentLoop dumper switches to the by_task_type/<task>/<gamefile_id>__rollout_<idx>.jsonl
    # layout (instead of step_<N>/<request_id>.jsonl) and includes
    # tokens.{input_ids,attention_mask,position_ids,loss_mask} +
    # gamefile_id/task_type/rollout_idx/sampling_params/model_path/ckpt_step.
    # The trajectories are then trivially convertible to a verl-SFT parquet
    # via tools/jsonl_to_parquet.py.
    p.add_argument(
        "--teacher_rollout_dir",
        type=str,
        default="",
        help="If set, treat this as a teacher rollout: dump trajectories under "
             "<teacher_rollout_dir>/by_task_type/<task_type>/<id>__rollout_<n>.jsonl "
             "with full tokens fields, and write metadata.json + summary.json at "
             "the run root.",
    )
    p.add_argument(
        "--ckpt_step",
        type=int,
        default=-1,
        help="Optional explicit checkpoint step number for metadata. If "
             "--model_path looks like .../global_step_<N>/actor/huggingface, the "
             "step is auto-extracted when this flag is left at -1.",
    )
    p.add_argument(
        "--rollouts_per_game",
        type=int,
        default=1,
        help="Number of independent passes over the dataset (each with a fresh "
             "seed_base offset). Each pass writes a new rollout_idx per game. "
             "Used for teacher rollout multi-sampling.",
    )
    p.add_argument(
        "--gamefile_filter_json",
        type=str,
        default="",
        help="Optional path to a JSON file with a gamefile_id whitelist "
             "(either {\"gamefile_ids\": [...]} or a flat list of timestamp "
             "ids). Only games whose extracted gid is in the whitelist are "
             "rolled. Used by DAgger Phase 1 to roll the student on a specific "
             "bucket (e.g. S35x64 part B's 64 games).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    for k, v in sorted(vars(args).items()):
        print(f"[eval] arg.{k} = {v}", flush=True)
    return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
