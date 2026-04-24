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
"""Single-sample, synchronous, non-Ray ALFWorld env wrapper for upstream verl's
agent_loop framework.

Ported from verl-agent:
  - agent_system/environments/env_manager.py :: AlfWorldEnvironmentManager
  - agent_system/environments/env_package/alfworld/envs.py :: AlfworldWorker
  - agent_system/memory/memory.py :: SimpleMemory
  - agent_system/environments/prompts/alfworld.py (-> prompts.py)

Design:
  * One wrapper instance == one TextWorld env (batch_size=1) == one episode.
  * No Ray, no batching, no gym.Env inheritance. Plain synchronous class.
  * SimpleMemory is owned per-instance (episode-scoped).
"""
from __future__ import annotations

import os
from typing import Any

import yaml

# Upstream verl must not depend on verl-agent. alfworld is installed as a
# site-packages module in the conda env; import it directly.
from alfworld.agents.environment import get_environment

from .prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS

# ---------------------------------------------------------------------------
# SimpleMemory (single-env variant).
#
# verl-agent's SimpleMemory tracks a batch of envs; here we only ever have one,
# so we collapse batch_size==1 and store a single list of dicts.
# ---------------------------------------------------------------------------


class SimpleMemory:
    """Episode-scoped per-env history buffer (batch_size=1 equivalent)."""

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._data)

    def reset(self) -> None:
        self._data = []

    def store(self, text_obs: str, action: str) -> None:
        self._data.append({"text_obs": text_obs, "action": action})

    def fetch(self, history_length: int) -> tuple[str, int]:
        recent = self._data[-history_length:]
        valid_len = len(recent)
        start_idx = len(self._data) - valid_len
        lines: list[str] = []
        for j, rec in enumerate(recent):
            step_num = start_idx + j + 1
            lines.append(
                f"[Observation {step_num}: '{rec['text_obs']}', "
                f"Action {step_num}: '{rec['action']}']"
            )
        return "\n".join(lines), valid_len


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config_file(path: str) -> dict:
    assert os.path.exists(path), f"Invalid config file: {path}"
    with open(path) as reader:
        return yaml.safe_load(reader)


def _unwrap_infos(infos: dict) -> dict:
    """AlfredTWEnv.step/reset wraps every value in a length-1 list for
    batch_size=1 (see verl-agent envs.py:128-132). Unwrap them all."""
    out: dict[str, Any] = {}
    for k, v in infos.items():
        if isinstance(v, (list, tuple)) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def _compute_reward(info: dict) -> float:
    """Mirror verl-agent envs.py:48-53 (text-only case)."""
    return 10.0 * float(info.get("won", False))


def _extract_task(text_obs: str) -> str:
    """Parse 'Your task is to: <desc>' from the initial observation."""
    marker = "Your task is to: "
    idx = text_obs.find(marker)
    if idx == -1:
        return ""
    return text_obs[idx + len(marker):].strip()


# ---------------------------------------------------------------------------
# AlfWorldSingleEnv
# ---------------------------------------------------------------------------


class AlfWorldSingleEnv:
    """Per-sample ALFWorld TextWorld wrapper for the upstream agent_loop
    framework.

    Each instance owns exactly one underlying TextWorld env initialised with
    batch_size=1. Thread-safety of this class is determined by the probe
    script (``probe_thread_safety.py``).
    """

    def __init__(
        self,
        alf_config_path: str,
        seed: int,
        gamefile: str | None = None,  # reserved; AlfredTWEnv picks from pool
        history_length: int = 2,
        env_kwargs: dict | None = None,
        is_train: bool = True,
    ) -> None:
        self.alf_config_path = alf_config_path
        self.seed_value = int(seed)
        self.history_length = int(history_length)
        self.is_train = bool(is_train)
        self.env_kwargs = env_kwargs or {}

        config = _load_config_file(alf_config_path)
        env_type = config["env"]["type"]
        if env_type != "AlfredTWEnv":
            # Multi-modal AlfredThorEnv not supported by this wrapper.
            raise ValueError(
                f"AlfWorldSingleEnv only supports AlfredTWEnv, got {env_type}"
            )

        eval_dataset = self.env_kwargs.get("eval_dataset", "eval_in_distribution")
        train_eval = "train" if is_train else eval_dataset
        base_env = get_environment(env_type)(config, train_eval=train_eval)

        # batch_size=1: one TextWorld env per wrapper instance.
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(self.seed_value)

        # Episode-scoped state.
        self.memory = SimpleMemory()
        self.gamefile: str | None = gamefile
        self.task: str = ""
        self.pre_text_obs: str = ""
        self._last_info: dict = {}
        self._admissible: list[str] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core gym-like API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[str, list[str], dict]:
        obs_list, infos = self.env.reset()
        info = _unwrap_infos(infos)
        text_obs = obs_list[0]

        # gamefile is surfaced via the AlfredInfos wrapper as 'extra.gamefile'.
        if "extra.gamefile" in info and info["extra.gamefile"] is not None:
            self.gamefile = info["extra.gamefile"]

        self.memory.reset()
        self.task = _extract_task(text_obs)
        self.pre_text_obs = text_obs
        self._last_info = info
        self._admissible = list(info.get("admissible_commands", []) or [])
        self._step_count = 0

        # Surface a compact "task_type" attribute from the gamefile path
        # (pick_and_place / pick_heat_then_place_in_recep / ...).
        info["task_type"] = self._infer_task_type(self.gamefile)
        info["won"] = bool(info.get("won", False))

        return text_obs, list(self._admissible), info

    def step(self, action: str) -> tuple[str, list[str], float, bool, dict]:
        # AlfredTWEnv.step expects a list of actions (batch_size=1).
        obs_list, scores, dones, infos = self.env.step([action])
        info = _unwrap_infos(infos)
        text_obs = obs_list[0]
        done = bool(dones[0])

        # Preserve gamefile across steps (TextWorld only reports it on reset).
        if info.get("extra.gamefile") is None and self.gamefile is not None:
            info["extra.gamefile"] = self.gamefile
        info["task_type"] = self._infer_task_type(self.gamefile)
        info["won"] = bool(info.get("won", False))

        reward = _compute_reward(info)

        # Update episode state.
        self.memory.store(text_obs=self.pre_text_obs, action=action)
        self.pre_text_obs = text_obs
        self._last_info = info
        self._admissible = list(info.get("admissible_commands", []) or [])
        self._step_count += 1

        return text_obs, list(self._admissible), reward, done, info

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_prompt(
        self,
        current_obs: str,
        admissible: list[str],
        step: int,
    ) -> str:
        """Format the ALFWorld prompt for the current state.

        Mirrors AlfWorldEnvironmentManager.build_text_obs (env_manager.py:180-212)
        but for a single sample. ``step`` is the 0-indexed step count before
        the next action (i.e. len(memory) after the most recent store()).
        """
        reformatted_admissible = "\n ".join(
            f"'{s}'" for s in admissible if s != "help"
        )

        if step <= 0 or self.history_length <= 0:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=current_obs,
                admissible_actions=reformatted_admissible,
            )

        memory_context, valid_len = self.memory.fetch(self.history_length)
        return ALFWORLD_TEMPLATE.format(
            task_description=self.task,
            step_count=step,
            history_length=valid_len,
            action_history=memory_context,
            current_step=step + 1,
            current_observation=current_obs,
            admissible_actions=reformatted_admissible,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            # TextWorld envs don't always implement a clean close(); swallow.
            pass

    @staticmethod
    def _infer_task_type(gamefile: str | None) -> str | None:
        if not gamefile:
            return None
        for task in (
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ):
            if task in gamefile:
                return task
        return None
