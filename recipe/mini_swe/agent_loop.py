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
"""Mini-swe-agent rollout loop for verl.

M4 of the mini-swe-agent x verl integration. Drives a multi-turn rollout
against a long-lived :class:`~recipe.mini_swe.environments.apptainer.ApptainerEnvironment`:

1. Build an apptainer instance from the per-sample SIF image, ``git reset
   --hard`` to the base commit, and optionally run an install script.
2. Render the vendored ``swebench_v2.yaml`` prompt templates (system +
   instance) via Jinja2 ``StrictUndefined``.
3. Tokenize the initial prompt once with the full chat template, then drive
   a loop of (a) ``server_manager.generate`` to produce the next assistant
   turn, (b) parse bash-command blocks or a ``### MINI_SWE_AGENT_FINAL_OUTPUT
   ###`` marker out of the decoded assistant text, (c) run each bash command
   through the env (capped at 8 per turn) and tokenize the concatenated
   output as a user-turn delta (``remove_system_prompt=True``).
4. On final output, invoke :func:`~recipe.mini_swe.reward.score_patch` to
   compute the scalar reward.
5. Assemble an :class:`AgentLoopOutput` with prompt / response split on the
   first assistant token, left-truncated to ``max_prompt_length`` and
   right-truncated to ``max_response_length``.

Reference points in the upstream verl code (cross-checked line numbers):

* ``AgentLoopBase.__init__``        verl/experimental/agent_loop/agent_loop.py:297-316
* ``AgentLoopBase.apply_chat_template``  verl/experimental/agent_loop/agent_loop.py:339-405
* ``AgentLoopBase.run``             verl/experimental/agent_loop/agent_loop.py:407-418
* ``AsyncLLMServerManager.generate`` verl/experimental/agent_loop/agent_loop.py:143-176
* ``AgentLoopOutput``               verl/experimental/agent_loop/agent_loop.py:188-243
* ``AgentLoopMetrics``              verl/experimental/agent_loop/agent_loop.py:179-185
* ``TokenOutput``                   verl/workers/rollout/replica.py:39-51

The prompt-response splitting mirrors ``AlfWorldAgentLoop.run`` (lines
411-420) but is extended with a mandatory left-truncation of the prompt to
``max_prompt_length`` because SWE-bench instance prompts plus long bash
transcripts routinely exceed the hard prompt budget.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yaml

# NOTE on imports: ``verl`` transitively depends on ``ray``, which is only
# available on machines with the full training stack installed. The pure
# helpers in this module (``_render_initial_messages``, ``_parse_actions``,
# ``_parse_final``, ``_format_observation``, ``_split_prompt_response``,
# ``_truncate``, ``_load_prompt_config``) must be importable without ray,
# so verl / apptainer imports are performed lazily inside the class body.

from .environments.apptainer import ApptainerEnvironment
from .reward import score_patch

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
    from verl.utils.rollout_trace import rollout_trace_op
    from verl.workers.rollout.replica import TokenOutput

    _VERL_AVAILABLE = True
except ImportError as _e:  # ray / verl not installed in this env
    _VERL_IMPORT_ERROR = _e
    _VERL_AVAILABLE = False

    # Shim so the rest of the module parses & the unit tests for the pure
    # helpers can still run. Any attempt to actually construct a
    # MiniSweAgentLoop without verl raises at __init__ time.
    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"MiniSweAgentLoop requires verl (and ray) to be installed: {_VERL_IMPORT_ERROR}"
            )

    # Minimal shims — mirror the pydantic-model attribute access pattern so
    # mock tests that bypass ``AgentLoopBase.__init__`` can still construct
    # and inspect an ``AgentLoopOutput`` without importing verl.
    class AgentLoopMetrics:  # type: ignore[no-redef]
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class AgentLoopOutput:  # type: ignore[no-redef]
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TokenOutput:  # type: ignore[no-redef]
        pass

    def rollout_trace_op(fn):  # type: ignore[no-redef]
        return fn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


FINAL_MARKER = "### MINI_SWE_AGENT_FINAL_OUTPUT ###"
BASH_BEGIN = "### BEGIN BASH COMMAND ###"
BASH_END = "### END BASH COMMAND ###"

# Match ``FINAL_MARKER`` followed by the rest of the message. Everything after
# the marker up to end-of-text is the submitted patch. ``re.DOTALL`` so
# newlines inside the diff match ``.``.
_FINAL_RE = re.compile(
    re.escape(FINAL_MARKER) + r"\s*(.*)\Z",
    re.DOTALL,
)

# Capture everything between BEGIN/END delimiters, non-greedy so consecutive
# blocks don't collapse into one.
_BASH_RE = re.compile(
    re.escape(BASH_BEGIN) + r"\s*(.*?)\s*" + re.escape(BASH_END),
    re.DOTALL,
)

# Nudge text when the assistant forgets to wrap its commands. Kept as plain
# text (no template rendering needed) because this only ever appears inline
# in the observation.
NUDGE_MISSING_BASH = (
    "I did not find any bash command in your message. Please wrap each "
    "shell command in:\n\n"
    f"{BASH_BEGIN}\n"
    "<your command>\n"
    f"{BASH_END}\n\n"
    "When your patch is ready, emit it after:\n"
    f"{FINAL_MARKER}\n"
)

MAX_COMMANDS_PER_TURN = 8


# ----------------------------------------------------------------------------
# Pure helpers (testable without verl / ray)
# ----------------------------------------------------------------------------


def _load_prompt_config(config_path: Optional[str] = None) -> dict:
    """Load the vendored ``swebench_v2.yaml`` (or a supplied override)."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            "swebench_v2.yaml",
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "agent" not in cfg:
        raise ValueError(
            f"Prompt config at {config_path} missing required top-level 'agent' key"
        )
    agent_cfg = cfg["agent"]
    for required in ("system_template", "instance_template"):
        if required not in agent_cfg:
            raise ValueError(
                f"Prompt config at {config_path} missing required 'agent.{required}' key"
            )
    return cfg


def _render_initial_messages(
    cfg: dict,
    task: str,
    cwd: str = "/testbed",
    env_vars: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Render ``system`` + ``user`` (instance) messages via Jinja2 StrictUndefined.

    Returns a list of ``{"role", "content"}`` dicts ready to hand to
    :meth:`AgentLoopBase.apply_chat_template`.

    The ``env_vars`` dict is exposed to the templates but currently unused by
    the vendored templates; it mirrors the upstream mini-swe-agent contract
    where ``get_template_vars`` feeds env state into the instance prompt.
    """
    # Import here so the rest of the module is importable even when jinja2 is
    # not installed (pure unit tests can monkeypatch this function).
    from jinja2 import Environment, StrictUndefined

    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    agent_cfg = cfg["agent"]
    system_tmpl = env.from_string(agent_cfg["system_template"])
    instance_tmpl = env.from_string(agent_cfg["instance_template"])

    vars_ = {"task": task, "cwd": cwd, "env_vars": env_vars or {}}
    system_text = system_tmpl.render(**vars_)
    instance_text = instance_tmpl.render(**vars_)
    return [
        {"role": "system", "content": system_text.rstrip()},
        {"role": "user", "content": instance_text.rstrip()},
    ]


def _parse_actions(text: str) -> list[str]:
    """Extract bash commands from an assistant message.

    Commands appear between ``### BEGIN BASH COMMAND ###`` and
    ``### END BASH COMMAND ###`` markers. We strip leading/trailing whitespace
    from each command; empty commands are skipped. Cap at
    :data:`MAX_COMMANDS_PER_TURN`.
    """
    out: list[str] = []
    for m in _BASH_RE.finditer(text):
        cmd = m.group(1).strip()
        if cmd:
            out.append(cmd)
        if len(out) >= MAX_COMMANDS_PER_TURN:
            break
    return out


def _parse_final(text: str) -> Optional[str]:
    """Extract the final patch after ``### MINI_SWE_AGENT_FINAL_OUTPUT ###``.

    Returns ``None`` if the marker is not present, else the stripped text
    following it to end of message.
    """
    m = _FINAL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _format_observation(outputs: list[dict]) -> str:
    """Format a list of per-command execution results into a single user-turn
    observation string.

    Each element of ``outputs`` is the dict returned by
    :meth:`ApptainerEnvironment.execute` augmented with the originating
    ``command`` key. We emit a clearly delimited block per command so the
    model can tell which output corresponds to which input.
    """
    parts: list[str] = []
    for i, rec in enumerate(outputs, start=1):
        cmd = rec.get("command", "")
        output = rec.get("output", "")
        rc = rec.get("returncode", None)
        parts.append(
            f"[Command {i} / returncode={rc}]\n"
            f"$ {cmd}\n"
            f"{output.rstrip()}"
        )
    return "\n\n".join(parts) + "\n"


def _split_prompt_response(
    prompt_ids: list[int],
    response_mask: list[int],
    logprobs: Optional[list[float]],
) -> tuple[list[int], list[int], list[int], Optional[list[float]]]:
    """Split the running ``prompt_ids`` buffer on the first assistant token.

    ``response_mask`` is parallel to ``prompt_ids`` in our bookkeeping
    convention: mask=0 for the initial prompt bytes AND for observation-delta
    tokens spliced in between turns, mask=1 for assistant-generated tokens.

    If no assistant turn has happened yet (no 1 in the mask), everything is
    "prompt" and the response side is empty.

    Returns ``(prompt_ids_final, response_ids_final, response_mask_final,
    response_logprobs_final)``. ``response_logprobs_final`` is None iff
    ``logprobs`` was None.
    """
    # Locate the first assistant token.
    try:
        split = response_mask.index(1)
    except ValueError:
        split = len(prompt_ids)

    prompt_part = prompt_ids[:split]
    response_part = prompt_ids[split:]
    mask_part = response_mask[split:]
    if logprobs is not None:
        logprobs_part: Optional[list[float]] = logprobs[split:]
    else:
        logprobs_part = None

    return prompt_part, response_part, mask_part, logprobs_part


def _truncate(
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    logprobs: Optional[list[float]],
    max_prompt: int,
    max_response: int,
) -> tuple[list[int], list[int], list[int], Optional[list[float]]]:
    """Left-truncate prompt to ``max_prompt`` and right-truncate response
    side (ids + mask + optional logprobs) to ``max_response``.

    Left-truncation of the prompt is required for SWE-bench because the
    instance prompt plus long bash transcripts routinely exceed typical
    ``prompt_length`` caps. We keep the tail of the prompt (closest to the
    first assistant token) so the model's final context is not severed.
    """
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    response_ids = response_ids[:max_response]
    response_mask = response_mask[:max_response]
    if logprobs is not None:
        logprobs = logprobs[:max_response]
    return prompt_ids, response_ids, response_mask, logprobs


# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------


# NOTE: no ``@register("mini_swe")`` decorator here — the agent_loops.yaml
# registry file (delivered in M5) is the canonical registration mechanism and
# the decorator would overwrite its kwargs. See
# ``AlfWorldAgentLoop`` (alfworld_agent_loop.py:204-212) for the same pattern.
class MiniSweAgentLoop(AgentLoopBase):
    """Multi-turn mini-swe-agent loop for SWE-bench-style repo-patching tasks.

    Required per-sample fields (``kwargs["extra_info"]``):

    * ``instance_id`` : str — SWE-bench instance id (used for log filenames).
    * ``repo`` : str — e.g. ``django/django``.
    * ``problem_statement`` : str — task text surfaced to the model.
    * ``base_commit`` : str — commit to ``git reset --hard`` before rollout.
    * ``sif_path`` : str — absolute path to the apptainer SIF image.
    * ``install_spec`` : str — shell snippet to run after reset (optional).
    * ``fail_to_pass`` : list[str] — F2P test ids for scoring.
    * ``pass_to_pass`` : list[str] — P2P test ids for scoring.
    * ``test_runner`` : :class:`~recipe.mini_swe.dataset_adapters.swe_bench_runners.TestRunnerSpec`
       — how to run the tests for this repo/version (can be ``None`` in which
       case ``score_patch`` returns 0.0).

    Optional ``__init__`` kwargs (supplied by the YAML registry in M5):

    * ``prompt_config_path`` : override for the vendored
      ``configs/swebench_v2.yaml``.
    * ``trajectory_log_dir`` : if set, each rollout writes a per-turn JSONL
      file to ``{trajectory_log_dir}/{instance_id}_{request_id}.jsonl``. Skip
      silently if writes fail — never crash a rollout over observability.
    * ``max_assistant_turns`` : hard cap on generate calls (default from
      ``rollout_config.multi_turn.max_assistant_turns`` or 30).
    """

    def __init__(
        self,
        *args,
        prompt_config_path: Optional[str] = None,
        trajectory_log_dir: Optional[str] = None,
        max_assistant_turns: Optional[int] = None,
        name: Optional[str] = None,  # absorbed from YAML registry entry
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        mt = self.rollout_config.multi_turn
        default_max_asst = (mt.max_assistant_turns if mt is not None else None) or 30
        self.max_assistant_turns = int(
            max_assistant_turns if max_assistant_turns is not None else default_max_asst
        )

        # Load prompt templates once at init so per-call run() stays cheap.
        self._prompt_cfg = _load_prompt_config(prompt_config_path)

        self.trajectory_log_dir: Optional[str] = trajectory_log_dir
        if self.trajectory_log_dir:
            try:
                os.makedirs(self.trajectory_log_dir, exist_ok=True)
            except OSError as e:  # pragma: no cover - dumper must not crash run
                logger.warning(
                    "trajectory_log_dir %s could not be created: %s",
                    self.trajectory_log_dir,
                    e,
                )
                self.trajectory_log_dir = None

    # ------------------------------------------------------------------ run

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        instance_id: str = str(extra_info.get("instance_id", "unknown"))
        repo: str = str(extra_info.get("repo", ""))
        problem_statement: str = str(extra_info.get("problem_statement", ""))
        base_commit: str = str(extra_info.get("base_commit", ""))
        sif_path: str = str(extra_info.get("sif_path", ""))
        install_spec: str = str(extra_info.get("install_spec", "") or "")
        fail_to_pass: list[str] = list(extra_info.get("fail_to_pass", []) or [])
        pass_to_pass: list[str] = list(extra_info.get("pass_to_pass", []) or [])
        test_runner = extra_info.get("test_runner", None)
        # Dataset serializes test_runner as a plain dict for JSON/pickle
        # safety across verl's non_tensor_batch (see
        # ``recipe/mini_swe/dataset.py``). Rehydrate here so downstream
        # ``score_patch(..., runner)`` still receives a TestRunnerSpec. A
        # pre-constructed TestRunnerSpec instance (e.g. supplied by tests)
        # is left unchanged.
        if isinstance(test_runner, dict):
            from .dataset_adapters.swe_bench_runners import TestRunnerSpec

            test_runner = TestRunnerSpec(**test_runner)

        if not sif_path:
            raise ValueError(
                f"MiniSweAgentLoop: extra_info.sif_path is required (instance_id={instance_id})"
            )
        if not base_commit:
            raise ValueError(
                f"MiniSweAgentLoop: extra_info.base_commit is required (instance_id={instance_id})"
            )

        request_id = uuid4().hex

        # Trajectory bookkeeping buffers. Same convention as AlfWorldAgentLoop:
        # prompt_ids is the cumulative running prompt fed to vLLM, response_mask
        # is parallel to prompt_ids (0 = initial prompt / env obs, 1 = assistant
        # generated).
        prompt_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = []

        num_assistant_turns = 0
        gen_time_s = 0.0
        num_preempted = 0
        submission: Optional[str] = None
        exit_status = "running"
        trajectory_turns: list[dict] = []

        env: Optional[ApptainerEnvironment] = None
        try:
            # Constructing the apptainer instance blocks on ``apptainer instance
            # start`` which can take several seconds, so run it off-thread to
            # keep the asyncio event loop responsive.
            env = await self.loop.run_in_executor(
                None,
                lambda: ApptainerEnvironment(sif_path=sif_path),
            )

            # Reset the repo.
            await self.loop.run_in_executor(
                None,
                lambda: env.execute(
                    {"command": f"git reset --hard {base_commit}"}
                ),
            )

            # Optional install step.
            if install_spec.strip():
                await self.loop.run_in_executor(
                    None,
                    lambda: env.execute(
                        {"command": install_spec, "timeout": 600}
                    ),
                )

            # Build + tokenize the initial prompt (system + instance).
            messages = _render_initial_messages(
                self._prompt_cfg,
                task=problem_statement,
                cwd=env.config.cwd,
                env_vars=env.config.env,
            )
            initial_ids = await self.apply_chat_template(messages)
            prompt_ids = list(initial_ids)
            response_mask = [0] * len(prompt_ids)
            if response_logprobs is not None:
                response_logprobs = [0.0] * len(prompt_ids)

            # ---------------- turn loop ----------------
            while num_assistant_turns < self.max_assistant_turns:
                t0 = time.perf_counter()
                tok_out: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                gen_time_s += time.perf_counter() - t0
                preempt = getattr(tok_out, "num_preempted", 0) or 0
                num_preempted += preempt

                assistant_ids = list(tok_out.token_ids)
                prompt_ids += assistant_ids
                response_mask += [1] * len(assistant_ids)
                if response_logprobs is not None:
                    if tok_out.log_probs:
                        response_logprobs += list(tok_out.log_probs)
                    else:
                        # Length-invariant: if any turn is missing logprobs,
                        # drop the whole buffer rather than emit a misaligned
                        # partial list. Mirrors the guard in
                        # alfworld_agent_loop.py:631-635.
                        response_logprobs = None
                num_assistant_turns += 1

                # Decode assistant text for parsing (off-thread to keep loop
                # responsive on large tokenizers).
                assistant_text = await self.loop.run_in_executor(
                    None,
                    lambda ids=assistant_ids: self.tokenizer.decode(
                        ids, skip_special_tokens=True
                    ),
                )

                # Final-output marker wins over any bash commands in the same
                # message (mini-swe-agent convention).
                final = _parse_final(assistant_text)
                if final is not None:
                    submission = final
                    exit_status = "submitted"
                    trajectory_turns.append(
                        {
                            "turn": num_assistant_turns,
                            "text": assistant_text,
                            "actions": [],
                            "outputs": [],
                            "returncodes": [],
                            "final": submission,
                        }
                    )
                    break

                commands = _parse_actions(assistant_text)
                if not commands:
                    obs_text = NUDGE_MISSING_BASH
                    cmd_results: list[dict] = []
                else:
                    # Run each command serially through the apptainer instance.
                    cmd_results = []
                    for cmd in commands:
                        r = await self.loop.run_in_executor(
                            None,
                            lambda c=cmd: env.execute({"command": c}),
                        )
                        cmd_results.append(
                            {
                                "command": cmd,
                                "output": r.get("output", ""),
                                "returncode": r.get("returncode"),
                            }
                        )
                    obs_text = _format_observation(cmd_results)

                trajectory_turns.append(
                    {
                        "turn": num_assistant_turns,
                        "text": assistant_text,
                        "actions": [r["command"] for r in cmd_results],
                        "outputs": [r["output"] for r in cmd_results],
                        "returncodes": [r["returncode"] for r in cmd_results],
                    }
                )

                # Tokenize obs as user-turn delta (strip system prefix).
                obs_messages = [{"role": "user", "content": obs_text}]
                obs_ids = await self.apply_chat_template(
                    obs_messages, remove_system_prompt=True
                )
                obs_ids = list(obs_ids)
                prompt_ids += obs_ids
                response_mask += [0] * len(obs_ids)
                if response_logprobs is not None:
                    response_logprobs += [0.0] * len(obs_ids)

                # Length guard: if we've exhausted the full rollout window
                # (prompt + response), stop. After this point further generate
                # calls can only truncate into the past anyway.
                if len(prompt_ids) >= (self.prompt_length + self.response_length):
                    exit_status = "window_exhausted"
                    break

            if submission is None and exit_status == "running":
                exit_status = "max_turns"

            # ---------------- score ----------------
            if submission:
                reward_score = await self.loop.run_in_executor(
                    None,
                    lambda: score_patch(
                        submission,
                        env,
                        base_commit,
                        fail_to_pass,
                        pass_to_pass,
                        test_runner,
                    ),
                )
            else:
                reward_score = 0.0
        finally:
            if env is not None:
                try:
                    await self.loop.run_in_executor(None, env.close)
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "mini_swe: env.close failed for %s: %s", instance_id, e
                    )

        # ---------------- finalize ----------------
        (
            prompt_ids_final,
            response_ids_final,
            response_mask_final,
            response_logprobs_final,
        ) = _split_prompt_response(prompt_ids, response_mask, response_logprobs)

        (
            prompt_ids_final,
            response_ids_final,
            response_mask_final,
            response_logprobs_final,
        ) = _truncate(
            prompt_ids_final,
            response_ids_final,
            response_mask_final,
            response_logprobs_final,
            self.prompt_length,
            self.response_length,
        )

        submission_chars = len(submission) if submission else 0

        # Best-effort trajectory dump (never crash a rollout over logging).
        if self.trajectory_log_dir:
            self._dump_trajectory(
                instance_id=instance_id,
                request_id=request_id,
                turns=trajectory_turns,
                exit_status=exit_status,
                reward_score=float(reward_score),
                submission=submission,
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids_final,
            response_ids=response_ids_final,
            response_mask=response_mask_final,
            response_logprobs=response_logprobs_final,
            num_turns=num_assistant_turns,
            reward_score=float(reward_score),
            metrics=AgentLoopMetrics(
                generate_sequences=float(gen_time_s),
                num_preempted=int(num_preempted),
            ),
            extra_fields={
                "instance_id": instance_id,
                "repo": repo,
                "submission_chars": int(submission_chars),
                "exit_status": exit_status,
            },
        )

    # ------------------------------------------------------------------ dump

    def _dump_trajectory(
        self,
        *,
        instance_id: str,
        request_id: str,
        turns: list[dict],
        exit_status: str,
        reward_score: float,
        submission: Optional[str],
    ) -> None:
        """Write per-turn JSONL records to ``trajectory_log_dir``.

        One file per rollout; one JSON object per turn plus a trailing summary
        line. Best-effort: any IO error is swallowed with a warning.
        """
        if not self.trajectory_log_dir:
            return
        fname = f"{instance_id}_{request_id}.jsonl"
        path = Path(self.trajectory_log_dir) / fname
        try:
            with open(path, "w") as f:
                for t in turns:
                    f.write(json.dumps(t, default=str) + "\n")
                summary = {
                    "summary": True,
                    "instance_id": instance_id,
                    "request_id": request_id,
                    "exit_status": exit_status,
                    "reward_score": float(reward_score),
                    "submission_chars": len(submission) if submission else 0,
                }
                f.write(json.dumps(summary, default=str) + "\n")
        except OSError as e:  # pragma: no cover
            logger.warning(
                "mini_swe trajectory dump failed for %s: %s", path, e
            )
