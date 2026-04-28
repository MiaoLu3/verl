"""DAgger Phase 2: query the teacher at every state the student visited.

Walks a Phase-1 student-rollout dump (``<root>/by_task_type/<task>/<gid>__rollout_<n>.jsonl``),
and for each *student assistant turn k* of every trajectory, reconstructs
the student's exact decision-time context by slicing the rollout's
``tokens.input_ids`` at the k-th ``0 -> 1`` transition in ``tokens.loss_mask``.
That prefix is the byte-identical token tape the student conditioned on at
turn k -- including every prior assistant turn's full ``<think>...</think>``
block. Re-rendering ``messages`` through ``apply_chat_template`` would
strip historical thinking (Qwen3 chat-template behaviour); this script
explicitly does NOT do that.

We then submit the prefix to the teacher (Qwen3-8B RL ckpt). The teacher's
response replaces the student's k-th action. We emit one *single-turn-row*
SFT example per (gid, turn_k):

    tokens_input_ids  : prefix (verbatim from rollout tape) + teacher_response
                        (+ trailing <|im_end|> if vLLM didn't include it)
    tokens_loss_mask  : 0s for the prefix, 1s for the teacher's response
    messages          : sanity-only re-parse of the prefix + teacher turn,
                        from the decoded tape (NOT re-templated). <think>
                        blocks in historical turns are preserved.
    gamefile_id, task_type, rollout_idx, turn_idx, student_won, ...

Output schema is a strict superset of what ``PretokenizedSFTDataset``
requires (``tokens_input_ids`` / ``tokens_loss_mask``), so it slots into
the existing SFT trainer pipeline.

Phase 3 (token-budget matching) is a separate pass; this script labels
every state the student visited.

Important: student and teacher MUST share the same tokenizer (Qwen3-0.6B
and Qwen3-8B both use vocab=151643 with identical special-token ids).
This script does NOT re-tokenize the prefix; it feeds the rollout's exact
token ids to the teacher engine.

Usage:
    python -m recipe.alfworld.dagger_teacher_label \\
        --student_run_dir /scratch/.../student_rollouts/dagger_qwen3_0.6b_S35x64_A_1r_16ep_on_B_<jobid>_<ts> \\
        --teacher_model   /scratch/.../checkpoints/merged_hf/qwen3_8b_rl_step570 \\
        --out_parquet     /scratch/.../sft_data/dagger/qwen3_0.6b_A_on_B_step570_T0.4.parquet \\
        --tp 4 --concurrency 32 --temperature 0.4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from glob import glob
from typing import Any

import pandas as pd


# Same regex used in alfworld_agent_loop's dumper -- splits decoded chat-
# template output back into [{"role", "content"}] messages.
_IM_SPLIT_RE = re.compile(
    r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL,
)


def _parse_chat_messages(decoded: str) -> list[dict]:
    out: list[dict] = []
    for m in _IM_SPLIT_RE.finditer(decoded):
        out.append({"role": m.group(1), "content": m.group(2).rstrip()})
    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _str_to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def _assistant_turn_starts(loss_mask: list[int]) -> list[int]:
    """Indices i where ``loss_mask[i]==1 and loss_mask[i-1]==0`` -- the start
    of each assistant-generated span. The slice ``input_ids[:start_k]`` is
    exactly the prompt the student conditioned on at turn k.
    """
    starts: list[int] = []
    for i in range(1, len(loss_mask)):
        if loss_mask[i] == 1 and loss_mask[i - 1] == 0:
            starts.append(i)
    return starts


# ----------------------------------------------------------------------
# vLLM teacher engine (mirrors eval_standalone.StandaloneServerManager,
# stripped down to what this labeler needs)
# ----------------------------------------------------------------------
class TeacherEngine:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        dtype: str,
    ):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        print(
            f"[teacher] loading vllm AsyncLLM model={model_path} tp={tensor_parallel_size} "
            f"max_model_len={max_model_len} gpu_util={gpu_memory_utilization} "
            f"enforce_eager={enforce_eager} dtype={dtype}",
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
        print(f"[teacher] vllm AsyncLLM loaded in {time.time() - t0:.1f}s", flush=True)
        self._req_seq = 0
        self._req_seq_lock = asyncio.Lock()

    async def _next_req_id(self, base: str) -> str:
        async with self._req_seq_lock:
            self._req_seq += 1
            n = self._req_seq
        return f"{base}-{n}"

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        stop_token_ids: list[int] | None = None,
    ) -> tuple[list[int], str | None]:
        from vllm import SamplingParams as VLSP
        from vllm import TokensPrompt

        sp_kwargs = dict(
            temperature=float(sampling_params.get("temperature", 0.4)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            repetition_penalty=float(sampling_params.get("repetition_penalty", 1.0)),
            max_tokens=int(sampling_params.get("max_tokens", 4096)),
        )
        if stop_token_ids:
            sp_kwargs["stop_token_ids"] = list(stop_token_ids)
        vllm_sp = VLSP(**sp_kwargs)

        prompt = TokensPrompt(prompt_token_ids=list(prompt_ids))
        rid = await self._next_req_id(request_id)

        final_output = None
        async for out in self.engine.generate(
            prompt=prompt, sampling_params=vllm_sp, request_id=rid,
        ):
            final_output = out
        if final_output is None:
            raise RuntimeError(f"vllm AsyncLLM yielded no outputs for request {rid}")
        gen = final_output.outputs[0]
        return list(gen.token_ids), getattr(gen, "finish_reason", None)


# ----------------------------------------------------------------------
# Build state batch from Phase-1 student dump
# ----------------------------------------------------------------------
def collect_states(student_run_dir: str, keep: str) -> list[dict]:
    """Return one (state) dict per (trajectory, assistant_turn_k).

    Each dict carries the byte-faithful prompt_ids the student saw at turn k
    (sliced from the rollout's tokens.input_ids), plus metadata.
    """
    pattern = os.path.join(student_run_dir, "by_task_type", "*", "*.jsonl")
    files = sorted(glob(pattern))
    if not files:
        sys.exit(f"[teacher] no JSONL files matched {pattern}")
    print(f"[teacher] found {len(files)} student trajectory jsonls", flush=True)

    states: list[dict] = []
    n_skipped_filter = 0
    n_skipped_bad = 0
    for path in files:
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"[teacher] skip {path}: {type(e).__name__}: {e}", file=sys.stderr)
            n_skipped_bad += 1
            continue
        if keep == "won" and not rec.get("won", False):
            n_skipped_filter += 1
            continue
        if keep == "losing" and rec.get("won", False):
            n_skipped_filter += 1
            continue

        ids = rec.get("tokens", {}).get("input_ids") or []
        mask = rec.get("tokens", {}).get("loss_mask") or []
        if not ids or not mask or len(ids) != len(mask):
            print(
                f"[teacher] skip {path}: missing or mismatched tokens "
                f"(ids={len(ids)} mask={len(mask)})",
                file=sys.stderr,
            )
            n_skipped_bad += 1
            continue

        starts = _assistant_turn_starts(mask)
        if not starts:
            n_skipped_bad += 1
            continue

        gid = rec.get("gamefile_id", "unknown")
        task_type = rec.get("task_type", "unknown")
        rollout_idx = int(rec.get("rollout_idx", 0))
        won = bool(rec.get("won", False))
        # Cache the int-converted full tape ONCE per traj.
        ids_int = [int(x) for x in ids]
        for k, s in enumerate(starts):
            prompt_ids = list(ids_int[:s])
            states.append({
                "gamefile_id": gid,
                "task_type": task_type,
                "rollout_idx": rollout_idx,
                "turn_idx": int(k),
                "student_won": won,
                "prompt_ids": prompt_ids,
                "src_path": path,
            })

    print(
        f"[teacher] collected {len(states)} (gid, turn) states "
        f"(skipped_filter={n_skipped_filter}, skipped_bad={n_skipped_bad}, "
        f"filter={keep})",
        flush=True,
    )
    return states


# ----------------------------------------------------------------------
# Main async driver
# ----------------------------------------------------------------------
async def _run(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        sys.exit("[teacher] could not resolve <|im_end|> token id from teacher tokenizer")
    print(f"[teacher] tokenizer im_end_id={im_end_id}", flush=True)

    states = collect_states(args.student_run_dir, args.keep)
    if args.max_states > 0:
        states = states[: args.max_states]
        print(f"[teacher] truncated to first {len(states)} states (--max_states)",
              flush=True)

    # Drop states whose prefix would not leave room for a full teacher
    # response under max_model_len.
    rendered: list[dict] = []
    n_too_long = 0
    for s in states:
        plen = len(s["prompt_ids"])
        if plen + args.max_tokens_per_turn > args.max_model_len:
            n_too_long += 1
            continue
        rendered.append(s)
    print(
        f"[teacher] kept {len(rendered)} states "
        f"(dropped {n_too_long} too-long for max_model_len={args.max_model_len} "
        f"with max_tokens={args.max_tokens_per_turn})",
        flush=True,
    )
    if not rendered:
        sys.exit("[teacher] no states to label after length filter")

    teacher = TeacherEngine(
        model_path=args.teacher_model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=bool(args.enforce_eager),
        dtype=args.dtype,
    )

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens_per_turn,
        "repetition_penalty": 1.0,
    }
    print(f"[teacher] sampling_params={sampling_params}", flush=True)

    queue: asyncio.Queue = asyncio.Queue()
    for s in rendered:
        queue.put_nowait(s)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    results: list[dict] = []
    results_lock = asyncio.Lock()
    t_start = time.time()
    n_done = 0
    n_total = len(rendered)

    async def _worker(wid: int):
        nonlocal n_done
        while True:
            s = await queue.get()
            if s is None:
                queue.task_done()
                return
            t0 = time.time()
            prompt_ids = s["prompt_ids"]
            try:
                resp_ids, finish = await teacher.generate(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=f"{s['gamefile_id']}__r{s['rollout_idx']}__t{s['turn_idx']}",
                    stop_token_ids=[im_end_id],
                )
            except Exception as e:
                print(
                    f"[teacher] FAIL gid={s['gamefile_id']} t={s['turn_idx']} "
                    f"worker={wid}: {type(e).__name__}: {e}",
                    flush=True,
                )
                queue.task_done()
                continue

            # Mirror the rollout convention: every assistant turn ends with
            # <|im_end|> as a model-emitted (loss=1) token. vLLM's behaviour
            # depends on version: stop_token_ids may or may not include the
            # stop token itself. Append iff missing.
            if not resp_ids or resp_ids[-1] != im_end_id:
                resp_ids = list(resp_ids) + [im_end_id]
            else:
                resp_ids = list(resp_ids)

            tokens_input_ids = list(prompt_ids) + resp_ids
            tokens_loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

            # Sanity messages reconstruction (decode-time, NOT re-templated;
            # _parse_chat_messages preserves <think> in historical turns
            # because it just splits on <|im_start|>...<|im_end|>).
            try:
                full_decoded = tokenizer.decode(tokens_input_ids, skip_special_tokens=False)
                messages = _parse_chat_messages(full_decoded)
            except Exception as e:
                full_decoded = f"<decode failed: {e}>"
                messages = []

            row = {
                "messages": messages,
                "tools": [],
                "enable_thinking": True,  # teacher rolls with thinking
                "gamefile_id": s["gamefile_id"],
                "task_type": s["task_type"],
                "rollout_idx": int(s["rollout_idx"]),
                "turn_idx": int(s["turn_idx"]),
                "student_won": bool(s["student_won"]),
                "won": True,  # treating teacher action as gold
                "prompt_length": int(len(prompt_ids)),
                "response_length": int(len(resp_ids)),
                "response_loss_ones": int(sum(tokens_loss_mask)),
                "pretok_input_len": int(len(tokens_input_ids)),
                "tokens_input_ids": [int(x) for x in tokens_input_ids],
                "tokens_loss_mask": [int(x) for x in tokens_loss_mask],
                "model_path": args.teacher_model,
                "ckpt_step": int(args.ckpt_step),
                "finish_reason": finish or "",
                "src_path": s["src_path"],
            }

            elapsed = time.time() - t0
            async with results_lock:
                results.append(row)
                n_done += 1
                cur_done = n_done
            if cur_done == 1 or cur_done % max(1, n_total // 20) == 0 or cur_done == n_total:
                total_elapsed = time.time() - t_start
                rate = cur_done / max(1e-6, total_elapsed)
                print(
                    f"[teacher] [{cur_done}/{n_total}] gid={s['gamefile_id']} "
                    f"t={s['turn_idx']} prompt_len={len(prompt_ids)} "
                    f"resp_len={len(resp_ids)} ep={elapsed:.1f}s "
                    f"rate={rate:.2f}/s total={total_elapsed:.1f}s",
                    flush=True,
                )
            queue.task_done()

    workers = [asyncio.create_task(_worker(w)) for w in range(args.concurrency)]
    await asyncio.gather(*workers)

    # ----------------------------------------------------------------
    # Write parquet
    # ----------------------------------------------------------------
    df = pd.DataFrame(results)
    df = df.sort_values(by=["gamefile_id", "rollout_idx", "turn_idx"]).reset_index(drop=True)
    print(f"[teacher] writing {len(df)} rows -> {args.out_parquet}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)) or ".", exist_ok=True)
    df.to_parquet(args.out_parquet, engine="pyarrow", index=False)
    size_mb = os.path.getsize(args.out_parquet) / (1024 * 1024)
    total_loss_ones = int(df["response_loss_ones"].sum()) if len(df) else 0
    print(
        f"[teacher] wrote {args.out_parquet} ({size_mb:.1f} MB)  "
        f"rows={len(df)}  trainable_tokens={total_loss_ones}",
        flush=True,
    )

    manifest = {
        "phase": "dagger_phase2",
        "student_run_dir": os.path.abspath(args.student_run_dir),
        "teacher_model": args.teacher_model,
        "ckpt_step": int(args.ckpt_step),
        "out_parquet": os.path.abspath(args.out_parquet),
        "n_rows": int(len(df)),
        "n_unique_games": int(df["gamefile_id"].nunique()) if len(df) else 0,
        "trainable_tokens_total": int(total_loss_ones),
        "trainable_tokens_per_row_mean": float(df["response_loss_ones"].mean()) if len(df) else 0.0,
        "by_task_type": {
            t: int(c) for t, c in df["task_type"].value_counts().to_dict().items()
        } if len(df) else {},
        "sampling_params": sampling_params,
        "max_model_len": int(args.max_model_len),
        "max_tokens_per_turn": int(args.max_tokens_per_turn),
        "keep_filter": args.keep,
        "prefix_source": "rollout_tokens_input_ids_byte_faithful",
        "args": {k: v for k, v in vars(args).items()},
        "wall_time_s": time.time() - t_start,
    }
    sidecar = args.out_parquet.replace(".parquet", ".manifest.json")
    with open(sidecar, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[teacher] manifest -> {sidecar}", flush=True)

    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_run_dir", required=True,
                   help="Phase-1 dump root: <root>/by_task_type/<task>/<gid>__rollout_<n>.jsonl.")
    p.add_argument("--teacher_model", required=True,
                   help="Path to the teacher HF ckpt (e.g. merged_hf/qwen3_8b_rl_step570). "
                        "Must share Qwen3's tokenizer with the student rollout.")
    p.add_argument("--out_parquet", required=True,
                   help="Output SFT parquet path.")
    p.add_argument("--ckpt_step", type=int, default=570,
                   help="Teacher ckpt step (metadata only).")

    p.add_argument("--keep", choices=["all", "won", "losing"], default="all",
                   help="Filter student trajectories by outcome before extracting "
                        "states. DAgger should default to 'all' -- the whole "
                        "point is to label states from failed rollouts too.")
    p.add_argument("--max_states", type=int, default=-1,
                   help="If >0, cap total (gid, turn) states (post-sort). For smoke tests.")

    # vLLM engine
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--concurrency", type=int, default=32)

    # Sampling
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens_per_turn", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    for k, v in sorted(vars(args).items()):
        print(f"[teacher] arg.{k} = {v}", flush=True)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
