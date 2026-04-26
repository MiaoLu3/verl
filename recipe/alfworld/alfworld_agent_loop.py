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
"""Multi-turn ALFWorld agent loop for upstream verl.

Token-in-token-out contract
---------------------------
We maintain three running buffers across all turns of a single episode:

* ``agent_data.prompt_ids``     - the running token sequence fed to vLLM. On
  T==0 it's the tokenized initial user message (via apply_chat_template with
  add_generation_prompt=True). After each generate call we append the
  assistant's token ids; after each env step we append the env-observation
  delta (tokenized as a user turn with remove_system_prompt=True).
* ``agent_data.response_mask``  - parallel to the part of ``prompt_ids``
  AFTER the initial prompt. 1 for assistant-generated tokens (trainable), 0
  for env-observation tokens spliced in between turns. This mirrors
  ``ToolAgentLoop`` where tool responses get mask=0.
* At finalization, the first ``len(prompt_ids) - len(response_mask)`` tokens
  of ``prompt_ids`` are the "prompt" and the tail matching ``response_mask``
  becomes ``response_ids`` - identical bookkeeping to ``tool_agent_loop.py``
  lines 191-214.

Critically, we NEVER re-render the full conversation from messages. Each env
turn is tokenized as a delta via
``apply_chat_template([{"role":"user","content":...}], ..., remove_system_prompt=True)``
which strips the BOS/system prefix that chat templates prepend on every call.
See ``tool_agent_loop.py:373-378`` for the same pattern on tool responses.

Sticky-session request_id
-------------------------
``agent_data.request_id`` is set once on episode start and reused across every
``server_manager.generate`` call, enabling vLLM prefix-cache reuse. Mirrors
``tool_agent_loop.py`` (it passes ``agent_data.request_id`` unchanged to every
generate call at line 236).
"""
from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .alfworld_projection import alfworld_projection

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_IM_SPLIT_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)(?:<\|im_end\|>|\Z)",
    re.DOTALL,
)


def _parse_chat_messages(decoded: str) -> list[dict]:
    """Split a Qwen chat-template-decoded string into ``[{role, content}]``.

    Finds every ``<|im_start|>ROLE\n...<|im_end|>`` span. The last span may
    lack ``<|im_end|>`` when the model hit the max-tokens cap mid-response;
    in that case we still emit the partial content so downstream SFT logic
    can see exactly what the model produced.
    """
    out: list[dict] = []
    for m in _IM_SPLIT_RE.finditer(decoded):
        role = m.group(1)
        content = m.group(2).rstrip()
        out.append({"role": role, "content": content})
    return out


_GAMEFILE_ID_RE = re.compile(r"trial_(T\d{8}_\d{6}_\d{6})")
_TASK_TYPE_RE = re.compile(
    r"json_2\.1\.1/(?:train|valid_seen|valid_unseen)/([A-Za-z_]+?)-"
)


def _extract_gamefile_id(gamefile: str) -> str:
    """Pull the unique trial timestamp out of an alfworld gamefile path.

    Example: ``.../trial_T20190907_174746_989712/game.tw-pddl`` →
    ``T20190907_174746_989712``. Returns ``"unknown"`` if no match.
    """
    if not gamefile:
        return "unknown"
    m = _GAMEFILE_ID_RE.search(gamefile)
    return m.group(1) if m else "unknown"


def _extract_task_type(gamefile: str) -> str:
    """Pull the task type (e.g. ``pick_and_place_simple``) from an alfworld
    gamefile path. Returns ``"unknown"`` if no match.
    """
    if not gamefile:
        return "unknown"
    m = _TASK_TYPE_RE.search(gamefile)
    return m.group(1) if m else "unknown"


def _next_rollout_idx(out_dir: str, gamefile_id: str) -> int:
    """Best-effort sequential rollout index: count existing
    ``<gamefile_id>__rollout_*.jsonl`` files in ``out_dir`` and return that
    count. Race-condition-prone under heavy concurrent dispatch; collisions
    must be retried at the caller via os.O_EXCL or similar.
    """
    if not os.path.isdir(out_dir):
        return 0
    prefix = f"{gamefile_id}__rollout_"
    return sum(
        1
        for f in os.listdir(out_dir)
        if f.startswith(prefix) and f.endswith(".jsonl")
    )


class _AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class _AlfWorldAgentData:
    """Per-episode state. Analogue of ``AgentData`` in ``tool_agent_loop.py``
    but specialised to the ALFWorld (env-interaction) flow - no tool calls,
    no images/videos.
    """

    def __init__(self, request_id: str, gamefile: str | None):
        self.request_id = request_id
        self.gamefile = gamefile

        # Running token buffers (see module docstring).
        self.prompt_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] | None = []

        # Turn counters.
        self.assistant_turns: int = 0
        self.user_turns: int = 0  # env-observation turns spliced in
        self.env_steps: int = 0

        # Last generate() output and last decoded assistant text (for parsing).
        self.last_response_ids: list[int] = []
        self.last_decoded_response: str = ""

        # Current env state.
        self.obs: str = ""
        self.admissible: list[str] = []
        self.info: dict[str, Any] = {}
        self.done: bool = False
        self.reward_accum: float = 0.0
        self.turn_scores: list[float] = []
        self.num_invalid_actions: int = 0

        # Actor handle (held for the duration of the episode).
        self.actor = None

        # Metrics + extra fields.
        self.metrics: dict[str, Any] = {}
        self.extra_fields: dict[str, Any] = {}

        # Per-turn trajectory records (populated only when the local JSONL
        # dumper is enabled; see ``ALFWORLD_TRAJ_DUMP_DIR``).
        self.trajectory_turns: list[dict] = []


# ---------------------------------------------------------------------------
# Pool singleton per worker
# ---------------------------------------------------------------------------
#
# The env pool is expensive to spin up (~15 s per actor for AlfredTWEnv's
# gamefile scan, times pool_size). We want exactly ONE pool per worker
# process, lazily created on first run() call. A module-level dict keyed by
# alf_config_path lets multiple configs coexist without re-warming the
# matching pool.

_POOL_CACHE: dict[tuple, "AlfWorldEnvPool"] = {}
_POOL_LOCK = None  # asyncio.Lock set up lazily


async def _get_or_create_pool(
    alf_config_path: str,
    pool_size: int,
    seed_base: int,
    is_train: bool,
    history_length: int,
):
    """Return the per-process pool for the given config tuple, warming it once.

    The key includes ``seed_base`` because verl-agent's ``make_envs`` creates
    separate train vs eval env sets with distinct seeds (``config.env.seed``
    vs ``config.env.seed + 1000``). Seeding selects which TextWorld gamefiles
    each actor ultimately rolls out on, so pools with different ``seed_base``
    are NOT interchangeable — dropping it from the key would let an eval
    invocation silently reuse a train-seeded pool.
    """
    import asyncio

    from .alfworld_env_pool import AlfWorldEnvPool

    global _POOL_LOCK
    if _POOL_LOCK is None:
        _POOL_LOCK = asyncio.Lock()

    async with _POOL_LOCK:
        key = (alf_config_path, pool_size, seed_base, is_train, history_length)
        if key in _POOL_CACHE:
            return _POOL_CACHE[key]

        loop = asyncio.get_event_loop()
        pool = await loop.run_in_executor(
            None,
            lambda: AlfWorldEnvPool(
                alf_config_path=alf_config_path,
                pool_size=pool_size,
                seed_base=seed_base,
                is_train=is_train,
                history_length=history_length,
            ),
        )
        _POOL_CACHE[key] = pool
        return pool


# NOTE: Do NOT decorate with ``@register("alfworld")``. The decorator (see
# agent_loop.py:429-434) overwrites the registry entry with
# ``{"_target_": fqdn}``, dropping any YAML-loaded kwargs (history_length,
# pool_size, etc.). Since we supply the full agent-loop config via the
# ``alfworld_agent_loop.yaml`` registry file, relying on the decorator loses
# those kwargs — causing ``self.default_history_length`` to silently fall
# back to the ``__init__`` default (2) even when the YAML sets 0.
# Registration happens purely through the YAML registry file.
class AlfWorldAgentLoop(AgentLoopBase):
    """Multi-turn ALFWorld agent loop with token-in-token-out bookkeeping.

    Expects the following to be resolvable from ``self.rollout_config``:

    * ``rollout_config.multi_turn.max_assistant_turns`` - hard cap on
      generate calls per episode (falls back to 50).
    * ``rollout_config.multi_turn.max_user_turns`` - hard cap on env turns
      (falls back to 50).
    * optional ``rollout_config.alfworld.*`` block with per-recipe kwargs:
        - ``alf_config_path``  : path to TextWorld yaml (default ``config_tw.yaml``)
        - ``pool_size``        : ray-actor pool size (default 8)
        - ``history_length``   : memory history length (default 2)
        - ``is_train``         : whether to use train split (default True)
        - ``seed_base``        : rng seed base for actors (default 12345)
        - ``max_steps``        : optional cap on env steps (default 50)

    These can also be supplied per-sample via ``kwargs["extra_info"]`` which
    overrides the rollout_config defaults - that's how per-split evaluation
    should hand the loop the path to ``valid_seen`` etc.
    """

    def __init__(
        self,
        *args,
        alf_config_path: str | None = None,
        pool_size: int = 8,
        history_length: int = 2,
        is_train: bool = True,
        seed_base: int = 12345,
        max_steps: int = 50,
        name: str | None = None,  # absorbed from YAML registry entry
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = int(mt.max_assistant_turns or 50)
        self.max_user_turns = int(mt.max_user_turns or 50)

        # AlfWorld-specific kwargs come from the agent_loop YAML registry via
        # hydra.utils.instantiate (AgentLoopWorker._generate, line 631-639).
        # RolloutConfig is a strict dataclass, so alfworld kwargs CANNOT live
        # at rollout_config.alfworld.*; they must be passed as explicit kwargs
        # here and declared in the yaml registry entry for "alfworld".
        default_cfg = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config_tw.yaml"
        )
        self.default_alf_config_path = alf_config_path or default_cfg
        self.default_pool_size = int(pool_size)
        self.default_history_length = int(history_length)
        self.default_is_train = bool(is_train)
        self.default_seed_base = int(seed_base)
        self.default_max_steps = int(max_steps)

        # Opt-in local JSONL trajectory dumper. When ALFWORLD_TRAJ_DUMP_DIR is
        # set, each terminated episode writes a single <request_id>.jsonl file
        # into it with per-turn observation/action/reward records. Zero
        # overhead when the env var is unset (dump_dir stays None and the
        # bookkeeping conditionals short-circuit).
        self.dump_dir = os.environ.get("ALFWORLD_TRAJ_DUMP_DIR")
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_user_content(
        self,
        obs: str,
        admissible: list[str],
        step: int,
        memory_text: str,
        memory_count: int,
        task: str,
    ) -> str:
        """Render the ALFWorld template for a single env turn."""
        from .prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS_V2

        reformatted_admissible = "\n ".join(
            f"'{s}'" for s in admissible if s != "help"
        )
        if step <= 0 or memory_count <= 0:
            # V2 drops the "You are an expert agent operating in the ALFRED
            # Embodied Environment." opener; every other part of NO_HIS is
            # preserved (obs + admissible + <think>/<action> instructions).
            return ALFWORLD_TEMPLATE_NO_HIS_V2.format(
                current_observation=obs,
                admissible_actions=reformatted_admissible,
            )
        return ALFWORLD_TEMPLATE.format(
            task_description=task,
            step_count=step,
            history_length=memory_count,
            action_history=memory_text,
            current_step=step + 1,
            current_observation=obs,
            admissible_actions=reformatted_admissible,
        )

    async def _tokenize_user_turn(self, content: str, is_first: bool) -> list[int]:
        """Tokenize a single user turn.

        * ``is_first=True``  : tokenize with apply_chat_template, keep system
          prefix, add_generation_prompt=True. This is the initial prompt of
          the episode.
        * ``is_first=False`` : tokenize the delta only (``remove_system_prompt=True``)
          so we don't double up the system prefix each turn.
        """
        messages = [{"role": "user", "content": content}]
        ids = await self.apply_chat_template(
            messages,
            remove_system_prompt=(not is_first),
        )
        return list(ids)

    # ------------------------------------------------------------------
    # State-machine entry point
    # ------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        alf_config_path = extra_info.get(
            "alf_config_path", self.default_alf_config_path
        )
        pool_size = int(extra_info.get("pool_size", self.default_pool_size))
        history_length = int(
            extra_info.get("history_length", self.default_history_length)
        )
        is_train = bool(extra_info.get("is_train", self.default_is_train))
        seed_base = int(extra_info.get("seed_base", self.default_seed_base))
        max_steps = int(extra_info.get("max_steps", self.default_max_steps))
        global_step = int(extra_info.get("global_step", -1))
        validate = bool(extra_info.get("validate", False))

        pool = await _get_or_create_pool(
            alf_config_path=alf_config_path,
            pool_size=pool_size,
            seed_base=seed_base,
            is_train=is_train,
            history_length=history_length,
        )

        agent_data = _AlfWorldAgentData(
            request_id=uuid4().hex,
            gamefile=extra_info.get("gamefile"),
        )

        # Track decoded history locally for the prompt template rendering
        # (the env wrapper has an internal SimpleMemory but it lives inside
        # the Ray actor; since we do a reset per episode its history mirrors
        # ours 1:1). Re-rendering history text here means the ray actor can
        # stay stateless from the loop's point of view - we don't need to
        # call render_prompt.remote() on every turn.
        history_records: list[tuple[str, str]] = []  # (pre_obs, action)
        task: str = ""

        try:
            agent_data.actor = await pool.acquire()

            state = _AgentState.PENDING
            while state != _AgentState.TERMINATED:
                if state == _AgentState.PENDING:
                    state = await self._handle_pending(
                        agent_data, sampling_params,
                    )
                    # After PENDING, `task` is set from initial obs.
                    task = agent_data.extra_fields.get("task", "")
                elif state == _AgentState.GENERATING:
                    state = await self._handle_generating(
                        agent_data, sampling_params,
                    )
                elif state == _AgentState.INTERACTING:
                    state = await self._handle_interacting(
                        agent_data, sampling_params,
                        history_records, history_length, task, max_steps,
                    )
                else:  # pragma: no cover - defensive
                    logger.error("invalid state %s", state)
                    state = _AgentState.TERMINATED

        finally:
            # Release the actor back to the pool (fire-and-forget errors).
            try:
                if agent_data.actor is not None:
                    await pool.release(agent_data.actor)
            except Exception as e:  # pragma: no cover
                logger.warning("failed to release actor: %s", e)

        # ------------------------------------------------------------------
        # Finalize output. Layout mirrors tool_agent_loop.py:191-214:
        #   prompt_ids = agent_data.prompt_ids[:len - len(response_mask)]
        #   response_ids = agent_data.prompt_ids[-len(response_mask):]
        # ------------------------------------------------------------------
        response_mask = agent_data.response_mask
        total = len(agent_data.prompt_ids)
        mask_len = len(response_mask)
        if mask_len == 0:
            # Degenerate case - nothing generated. Produce an empty response.
            prompt_ids_final = agent_data.prompt_ids
            response_ids_final: list[int] = []
        else:
            prompt_ids_final = agent_data.prompt_ids[: total - mask_len]
            response_ids_final = agent_data.prompt_ids[total - mask_len:]

        won = bool(agent_data.info.get("won", False))
        reward_score = float(won)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids_final,
            response_ids=response_ids_final[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None
            ),
            num_turns=agent_data.assistant_turns + agent_data.user_turns + 1,
            reward_score=reward_score,
            metrics=AgentLoopMetrics(
                generate_sequences=float(agent_data.metrics.get("generate_sequences", 0.0)),
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=int(agent_data.metrics.get("num_preempted", -1)),
            ),
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "gamefile": agent_data.gamefile or agent_data.info.get("extra.gamefile"),
                "won": won,
                "reward_accum": agent_data.reward_accum,
                "num_invalid_actions": agent_data.num_invalid_actions,
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": [],
                "env_steps": agent_data.env_steps,
                "task_type": agent_data.info.get("task_type"),
            }
        )

        if self.dump_dir or os.environ.get("TEACHER_ROLLOUT_RUN_DIR"):
            self._dump_trajectory(
                agent_data=agent_data,
                prompt_ids_final=prompt_ids_final,
                response_ids_final=response_ids_final,
                response_mask_final=response_mask,
                won=won,
                reward_score=reward_score,
                global_step=global_step,
                validate=validate,
                sampling_params=sampling_params,
            )

        return output

    # ------------------------------------------------------------------
    # Trajectory dumper
    # ------------------------------------------------------------------

    def _dump_trajectory(
        self,
        agent_data: _AlfWorldAgentData,
        prompt_ids_final: list[int],
        response_ids_final: list[int],
        response_mask_final: list[int],
        won: bool,
        reward_score: float,
        global_step: int = -1,
        validate: bool = False,
        sampling_params: Optional[dict] = None,
    ) -> None:
        """Write a single JSONL record for this episode.

        Two layouts:

        - **Legacy (RL training, val-only):** triggered by ``ALFWORLD_TRAJ_DUMP_DIR``.
          File path = ``{dump_dir}/step_{global_step}[_val]/<request_id>.jsonl``.
          ``step_unknown[_val]`` fallback when no step is passed.
        - **Teacher rollout:** triggered by ``TEACHER_ROLLOUT_RUN_DIR``. File path
          = ``{run_dir}/by_task_type/{task_type}/<gamefile_id>__rollout_<idx>.jsonl``.
          ``rollout_idx`` is the count of pre-existing files for this gamefile in
          the same directory (best-effort; concurrent dispatch can race, in which
          case the second writer simply overwrites — we accept that for SFT use).

        Both layouts emit the same enriched record (gamefile_id / task_type /
        rollout_idx / messages / tokens.{input_ids,attention_mask,position_ids,
        loss_mask} / lengths / sampling_params / model_path / ckpt_step). The
        ``tokens.loss_mask`` is full-length (= initial-prompt zeros + agent_loop
        ``response_mask``) so downstream SFT trainers can re-tokenize from
        ``messages`` and compare against this ground truth.
        """
        mask_1 = sum(1 for m in response_mask_final if m == 1)
        mask_0 = len(response_mask_final) - mask_1

        # Decode the full cumulative token buffer (initial prompt + every
        # assistant turn + every env-obs delta, in turn order) back to text.
        # This is exactly what the model "saw" as context on the final turn,
        # including chat-template markup. Useful for SFT curation and
        # verl-agent prompt-format byte-diff.
        full_ids = list(prompt_ids_final) + list(response_ids_final)
        try:
            full_decoded = self.tokenizer.decode(
                full_ids, skip_special_tokens=False
            )
        except Exception as e:  # pragma: no cover
            full_decoded = f"<decode failed: {e}>"

        # Parse the decoded chat-template string back into a list of
        # ``{"role", "content"}`` messages by splitting on Qwen's
        # ``<|im_start|>role\n...<|im_end|>`` markers. Makes downstream SFT
        # curation trivial -- no re-templating required.
        messages = _parse_chat_messages(full_decoded)

        gamefile = (
            agent_data.gamefile or agent_data.info.get("extra.gamefile") or ""
        )
        gamefile_id = _extract_gamefile_id(gamefile)
        task_type = _extract_task_type(gamefile)

        # Build full-length loss_mask: initial-prompt portion is non-trainable
        # (zeros), then the agent_loop response_mask carries forward (1 for
        # assistant-generated tokens, 0 for spliced env observations).
        prompt_len = len(prompt_ids_final)
        loss_mask_full = [0] * prompt_len + list(response_mask_final)
        # Length-align with input_ids in case response_mask was clipped earlier.
        if len(loss_mask_full) < len(full_ids):
            loss_mask_full = loss_mask_full + [0] * (len(full_ids) - len(loss_mask_full))
        elif len(loss_mask_full) > len(full_ids):
            loss_mask_full = loss_mask_full[: len(full_ids)]
        attention_mask = [1] * len(full_ids)
        position_ids = list(range(len(full_ids)))

        # Resolve target directory. Both modes use the same leaf layout:
        #   ``<root>/by_task_type/<task_type>/<gid>__rollout_<idx>.jsonl``
        # so downstream tooling (jsonl_to_parquet, traj_viewer) can treat
        # them uniformly. They differ only in where ``<root>`` is rooted:
        #
        # * Teacher rollout: ``<TEACHER_ROLLOUT_RUN_DIR>``
        # * RL training / val: ``<ALFWORLD_TRAJ_DUMP_DIR>/step_<N>[_val]``
        #   (``step_unknown[_val]`` fallback when no step is passed).
        teacher_dir = os.environ.get("TEACHER_ROLLOUT_RUN_DIR")
        if teacher_dir:
            root = teacher_dir
        else:
            if global_step >= 0:
                subdir = f"step_{global_step}_val" if validate else f"step_{global_step}"
            else:
                subdir = "step_unknown_val" if validate else "step_unknown"
            root = os.path.join(self.dump_dir, subdir)

        out_dir = os.path.join(root, "by_task_type", task_type)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:  # pragma: no cover
            logger.warning("alfworld trajectory mkdir failed for %s: %s", out_dir, e)
            return

        # Race-safe rollout_idx allocation: try O_EXCL create starting at
        # the count of existing files, increment on collision. Multiple
        # concurrent workers pinned to the same gamefile (which happens
        # whenever rollout_n > 1 in RL training, or rollouts_per_game > 1
        # in teacher rollout) will each grab a unique idx.
        rollout_idx = _next_rollout_idx(out_dir, gamefile_id)
        path = None
        for _try_idx in range(rollout_idx, rollout_idx + 100):
            cand = os.path.join(out_dir, f"{gamefile_id}__rollout_{_try_idx}.jsonl")
            try:
                fd = os.open(
                    cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
                )
                os.close(fd)
                rollout_idx = _try_idx
                path = cand
                break
            except FileExistsError:
                continue
        if path is None:
            logger.warning(
                "alfworld trajectory dump: could not claim rollout_idx in 100 tries "
                "for %s; skipping write", gamefile_id
            )
            return
            rollout_idx = 0
            fname = f"{agent_data.request_id}.jsonl"
            path = os.path.join(out_dir, fname)

        record = {
            "request_id": agent_data.request_id,
            "gamefile": gamefile,
            "gamefile_id": gamefile_id,
            "task_type": task_type,
            "task_goal": agent_data.extra_fields.get("task", ""),
            "rollout_idx": int(rollout_idx),
            "won": bool(won),
            "final_reward": float(reward_score),
            "global_step": int(global_step),
            "validate": bool(validate),
            "num_turns": len(agent_data.trajectory_turns),
            "num_invalid_actions": int(agent_data.num_invalid_actions),
            "lengths": {
                "prompt_length_final": prompt_len,
                "response_length_final": len(response_ids_final),
                "loss_mask_1_count": int(mask_1),
                "loss_mask_0_count": int(mask_0),
            },
            "turns": agent_data.trajectory_turns,
            "messages": messages,
            "full_decoded_sequence": full_decoded,
            "tokens": {
                "input_ids": list(full_ids),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask_full,
            },
            "model_path": os.environ.get("TEACHER_ROLLOUT_MODEL_PATH", ""),
            "ckpt_step": int(os.environ.get("TEACHER_ROLLOUT_CKPT_STEP", "-1")),
            "sampling_params": dict(sampling_params or {}),
        }

        try:
            with open(path, "w") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:  # pragma: no cover - dumper must never crash run
            logger.warning("alfworld trajectory dump failed for %s: %s", path, e)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _handle_pending(
        self,
        agent_data: _AlfWorldAgentData,
        sampling_params: dict[str, Any],
    ) -> _AgentState:
        """Reset the env and build the initial prompt.

        If ``agent_data.gamefile`` is non-empty (set from ``extra_info.gamefile``
        by ``run()``), the env is pinned to that exact gamefile (bypass the
        actor's shuffled_cycle). Otherwise the actor's iterator picks the next
        game.
        """
        pin = agent_data.gamefile if agent_data.gamefile else None
        obs, adm, info = await agent_data.actor.reset.remote(gamefile=pin)
        agent_data.obs = obs
        agent_data.admissible = list(adm)
        agent_data.info = dict(info)
        # Pull task string out of the first obs.
        task_marker = "Your task is to: "
        task_idx = obs.find(task_marker)
        task = obs[task_idx + len(task_marker):].strip() if task_idx != -1 else ""
        agent_data.extra_fields["task"] = task
        if info.get("extra.gamefile"):
            agent_data.gamefile = info["extra.gamefile"]

        # Render & tokenize initial user message with FULL chat template
        # (keeps system prefix / BOS).
        user_content = self._render_user_content(
            obs=obs,
            admissible=agent_data.admissible,
            step=0,
            memory_text="",
            memory_count=0,
            task=task,
        )
        initial_ids = await self._tokenize_user_turn(user_content, is_first=True)
        agent_data.prompt_ids = initial_ids
        return _AgentState.GENERATING

    async def _handle_generating(
        self,
        agent_data: _AlfWorldAgentData,
        sampling_params: dict[str, Any],
    ) -> _AgentState:
        """Generate one assistant turn and append its tokens to buffers."""
        # No prompt_length guard: cumulative prompt_ids has no upstream cap.
        # `prompt_length` is only the post-rollout padding target for the
        # INITIAL prompt (output.prompt_ids), not the running context.
        # vLLM context is bounded by `max_model_len` (engine arg) instead.
        # Mirrors tool_agent_loop.py which has no such guard.

        with simple_timer("generate_sequences", agent_data.metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
            )

        # num_preempted accumulation (mirror tool_agent_loop.py:243-247).
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = (
                out.num_preempted if out.num_preempted is not None else -1
            )
        else:
            agent_data.metrics["num_preempted"] += (
                out.num_preempted if out.num_preempted is not None else 0
            )

        resp_ids = list(out.token_ids)
        agent_data.last_response_ids = resp_ids
        agent_data.assistant_turns += 1

        # Append assistant tokens to prompt buffer + set mask=1 for them.
        # Mirrors tool_agent_loop.py:258-260.
        agent_data.prompt_ids += resp_ids
        agent_data.response_mask += [1] * len(resp_ids)
        # Length-invariant guard: logprobs must be parallel to response_mask
        # once the episode finalizes. If ANY turn returns no log_probs, drop
        # the running buffer entirely (we'd rather emit None than a
        # mis-aligned partial list). Upstream tool_agent_loop.py:261-262 has
        # the same latent defect; we intentionally diverge here to avoid
        # silent length skew between log_probs and response_mask. TODO:
        # upstream this guard to tool_agent_loop.py once approved.
        if agent_data.response_logprobs is not None:
            if out.log_probs:
                agent_data.response_logprobs += list(out.log_probs)
            else:
                agent_data.response_logprobs = None

        # Decode once for projection parsing. Run off-thread to keep the
        # event loop responsive (fast-tokenizer decode is typically fast but
        # the Rust guard can still spike CPU).
        agent_data.last_decoded_response = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(resp_ids, skip_special_tokens=True),
        )

        # Termination after generation (response budget / turn caps).
        if len(agent_data.response_mask) >= self.response_length:
            return _AgentState.TERMINATED
        if agent_data.assistant_turns >= self.max_assistant_turns:
            return _AgentState.TERMINATED
        if agent_data.user_turns >= self.max_user_turns:
            return _AgentState.TERMINATED

        return _AgentState.INTERACTING

    async def _handle_interacting(
        self,
        agent_data: _AlfWorldAgentData,
        sampling_params: dict[str, Any],
        history_records: list[tuple[str, str]],
        history_length: int,
        task: str,
        max_steps: int,
    ) -> _AgentState:
        """Project the assistant text to an action, step the env, and
        tokenize the new observation as a user-turn delta."""
        # Snapshot pre-step state so the dumper records what the MODEL saw
        # before taking this action (observation + admissible come from the
        # prior reset/step, the raw_response from the generate we just ran).
        pre_obs = agent_data.obs
        pre_admissible = list(agent_data.admissible)
        raw_response_text = agent_data.last_decoded_response
        turn_idx = agent_data.env_steps  # 0-indexed step we're about to take

        action, valid = alfworld_projection(
            agent_data.last_decoded_response, agent_data.admissible
        )
        if not valid:
            agent_data.num_invalid_actions += 1

        obs, adm, reward, done, info = await agent_data.actor.step.remote(action)

        history_records.append((pre_obs, action))
        agent_data.obs = obs
        agent_data.admissible = list(adm)
        agent_data.info = dict(info)
        agent_data.done = bool(done)
        agent_data.reward_accum += float(reward)
        agent_data.turn_scores.append(float(reward))
        agent_data.env_steps += 1

        if self.dump_dir:
            agent_data.trajectory_turns.append(
                {
                    "turn_idx": turn_idx,
                    "observation": pre_obs,
                    "admissible": pre_admissible,
                    "raw_response": raw_response_text,
                    "parsed_action": action,
                    "valid": bool(valid),
                    "reward": float(reward),
                    "done": bool(done),
                }
            )

        if done or agent_data.env_steps >= max_steps:
            return _AgentState.TERMINATED

        # Build memory text from the tail of history_records.
        # Guard against ``history_length=0``: ``list[-0:]`` returns the FULL
        # list (since -0 == 0), not an empty slice — explicit check is
        # required to get the "no memory summary" behaviour.
        recent = history_records[-history_length:] if history_length > 0 else []
        valid_len = len(recent)
        start_idx = len(history_records) - valid_len
        lines: list[str] = []
        for j, (po, a) in enumerate(recent):
            step_num = start_idx + j + 1
            lines.append(
                f"[Observation {step_num}: '{po}', Action {step_num}: '{a}']"
            )
        memory_text = "\n".join(lines)

        user_content = self._render_user_content(
            obs=obs,
            admissible=agent_data.admissible,
            step=agent_data.env_steps,
            memory_text=memory_text,
            memory_count=valid_len,
            task=task,
        )
        env_ids = await self._tokenize_user_turn(user_content, is_first=False)

        # Response-budget guard before committing the env-obs delta. Mirrors
        # tool_agent_loop.py:380-381. No prompt_length guard — see
        # _handle_generating.
        if len(agent_data.response_mask) + len(env_ids) >= self.response_length:
            return _AgentState.TERMINATED

        agent_data.prompt_ids += env_ids
        agent_data.response_mask += [0] * len(env_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(env_ids)
        agent_data.user_turns += 1
        return _AgentState.GENERATING
