"""Standalone ALFWorld eval that REUSES upstream verl's AlfWorldAgentLoop +
AlfWorldEnvPool, but bypasses FSDP/ray_trainer/main_ppo to avoid the memory
spike we hit on Qwen3-32B under the full PPO wiring.

Architecture
------------
    StandaloneServerManager (wraps a single vllm.LLM engine, exposes the
    `generate()` async interface that AgentLoopBase expects)
        |
        v
    AlfWorldAgentLoop (unmodified; from recipe.alfworld.alfworld_agent_loop)
        |
        v
    AlfWorldEnvPool ray-actor pool (unmodified; held via module-level cache
    inside AlfWorldAgentLoop)

We instantiate AlfWorldAgentLoop directly, skipping hydra/AgentLoopWorker, and
drive one episode per valid_seen gamefile. All state-machine / projection /
trajectory-dumper logic is inherited verbatim.

Usage:
    python -m recipe.alfworld.eval_standalone \
        --model_path Qwen/Qwen3-32B \
        --tp 4 \
        --pool_size 8 \
        --alf_config_path /scratch/m000069-pm05/miaolu/verl/recipe/alfworld/config_tw.yaml \
        --dump_dir /scratch/m000069-pm05/miaolu/verl/trajectories/qwen3_32b_eval_YYYYMMDD
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
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
# StandaloneServerManager: wraps a local vllm.LLM to match the subset of
# AsyncLLMServerManager.generate() that AlfWorldAgentLoop calls.
# ----------------------------------------------------------------------
class StandaloneServerManager:
    """Minimal stand-in for verl's AsyncLLMServerManager.

    AlfWorldAgentLoop calls ``await self.server_manager.generate(
    request_id=..., prompt_ids=..., sampling_params=...)`` and expects a
    ``TokenOutput`` back. We do exactly that with a single in-process vllm.LLM.

    vllm.LLM.generate() is blocking; we punt it to a thread so the asyncio
    loop can keep running (pool acquire/release + tokenize also happen on
    executor threads). With pool_size=1 episodes are serialized through the
    single vllm engine anyway, which is what we want for peak memory safety.
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
        from vllm import LLM

        print(
            f"[StandaloneServerManager] Loading vllm.LLM model={model_path} "
            f"tp={tensor_parallel_size} max_model_len={max_model_len} "
            f"gpu_util={gpu_memory_utilization} enforce_eager={enforce_eager} dtype={dtype}",
            flush=True,
        )
        t0 = time.time()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dtype=dtype,
            trust_remote_code=True,
        )
        print(
            f"[StandaloneServerManager] vllm.LLM loaded in {time.time() - t0:.1f}s",
            flush=True,
        )
        self._generate_lock = asyncio.Lock()

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

        loop = asyncio.get_event_loop()

        def _run():
            # vllm >=0.9 removed the top-level ``prompt_token_ids`` kwarg;
            # pass token ids via ``TokensPrompt`` in the ``prompts`` list.
            from vllm import TokensPrompt
            return self.llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=list(prompt_ids))],
                sampling_params=vllm_sp,
                use_tqdm=False,
            )

        # Serialize vllm.generate calls — vllm.LLM's synchronous batched API is
        # not reentrant; concurrent launches from multiple asyncio tasks would
        # race over engine state. Our pool_size is the only real concurrency
        # knob and we target pool_size=1 for 32B; serialize explicitly anyway.
        async with self._generate_lock:
            outputs = await loop.run_in_executor(None, _run)

        out = outputs[0]
        gen = out.outputs[0]

        log_probs_list = None
        if logprobs_flag and getattr(gen, "logprobs", None):
            # vllm logprobs list[dict[token_id -> Logprob]]. We requested
            # logprobs=0 so each dict has a single entry; extract its .logprob.
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
        num = min(args.max_samples, len(ds))
    else:
        num = len(ds)
    print(f"[eval] evaluating {num} episodes", flush=True)

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

    # Run each episode sequentially. With pool_size=1 and a single vllm engine
    # this is the natural pipeline; bumping pool_size makes env warmup parallel
    # but vllm is still serialized by StandaloneServerManager._generate_lock.
    results = []
    t_start = time.time()
    for i in range(num):
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
                f"[eval] FAIL episode {i+1}/{num} gamefile={row['extra_info']['gamefile']}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            won = False
            num_turns = 0
            num_invalid = 0
            env_steps = 0
            gamefile = row["extra_info"]["gamefile"]

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
        elapsed = time.time() - ep_t0
        total_elapsed = time.time() - t_start
        cur_wins = sum(1 for r in results if r["won"])
        print(
            f"[eval] [{i+1}/{num}] won={won} turns={num_turns} "
            f"invalid={num_invalid} ep_time={elapsed:.1f}s "
            f"running_sr={cur_wins}/{len(results)}={cur_wins/len(results):.3f} "
            f"total={total_elapsed:.1f}s",
            flush=True,
        )

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
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[eval] summary written to {args.summary_path}", flush=True)
    elif args.dump_dir:
        p = os.path.join(args.dump_dir, "summary.json")
        with open(p, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[eval] summary written to {p}", flush=True)

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
    return p.parse_args()


def main():
    args = parse_args()
    for k, v in sorted(vars(args).items()):
        print(f"[eval] arg.{k} = {v}", flush=True)
    return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
