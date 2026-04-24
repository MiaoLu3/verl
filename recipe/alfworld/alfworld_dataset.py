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
"""Thin ALFWorld dataset for the upstream verl agent-loop framework.

Each row is a gamefile stub:

* ``raw_prompt``  - a placeholder user message; the real initial prompt is
  rendered inside ``AlfWorldAgentLoop._handle_pending`` after ``env.reset()``
  surfaces the first observation. The placeholder is kept minimal because
  ``AgentLoopWorker`` forwards it to the loop via kwargs, but our loop
  ignores it.
* ``extra_info.gamefile`` - absolute path to the ``game.tw-pddl`` file; the
  loop can optionally use this to re-seed / bias the env (currently
  AlfredTWEnv picks from a pool of games and we surface whatever gamefile
  it chose back on ``reset()``; per-sample pinning is a follow-up).
* ``agent_name`` == ``"alfworld"`` - routes each row to ``AlfWorldAgentLoop``
  via the ``@register("alfworld")`` registry.
* ``index`` - unique int per sample, used by ``AgentLoopWorker`` for
  trajectory tagging.

We subclass ``torch.utils.data.Dataset`` directly because upstream's
``RLHFDataset`` is parquet-centric and the filter/resume/tokenize plumbing
(``_read_files_and_tokenize``, ``maybe_filter_out_long_prompts``) is
irrelevant here - we enumerate real filesystem paths, not HF rows.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import Dataset

try:
    from transformers import PreTrainedTokenizer, ProcessorMixin  # noqa: F401
except ImportError:  # pragma: no cover
    pass


def _expand(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(p)))


def _enumerate_gamefiles(root: str) -> list[str]:
    """Walk a TextWorld split root and return sorted absolute paths to every
    ``game.tw-pddl`` underneath it."""
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".tw-pddl"):
                matches.append(os.path.join(dirpath, fn))
    matches.sort()
    return matches


class AlfWorldDataset(Dataset):
    """Thin file-backed dataset that enumerates ALFWorld gamefiles for one
    split.

    Args:
        config: omega/dict config. Must contain ``alf_config_path`` (path to
            the TextWorld yaml) OR accept the default next to this module.
            May also override ``data_path`` / ``eval_id_data_path`` /
            ``eval_ood_data_path`` via ``config.custom_cls_args`` - but by
            default we read them straight out of the yaml's ``dataset``
            section.
        tokenizer: unused (placeholder for API compatibility with
            ``RLHFDataset``).
        processor: unused (same reason).
        split: one of ``"train"``, ``"valid_seen"``, ``"valid_unseen"``.
    """

    # Valid split names, mapped to their yaml keys.
    _SPLIT_KEYS = {
        "train": "data_path",
        "valid_seen": "eval_id_data_path",
        "valid_unseen": "eval_ood_data_path",
    }

    def __init__(
        self,
        config: DictConfig | dict = None,
        tokenizer=None,
        processor=None,
        split: Optional[str] = None,
        alf_config_path: Optional[str] = None,
        max_samples: int = -1,
        data_files: Any = None,
        **_ignored,
    ) -> None:
        # Upstream ``create_rl_dataset`` calls dataset_cls with
        # (data_files=..., tokenizer=..., processor=..., config=..., max_samples=...)
        # but ALFWorld gamefiles are enumerated from the ``config_tw.yaml`` dataset
        # section, not from ``data_files``. We accept ``data_files`` for API
        # compatibility and ignore it (a one-element dummy sentinel is fine).
        _ = data_files  # unused

        # Resolve split. Priority:
        #   1. Explicit kwarg (tests / programmatic callers).
        #   2. ``config.alfworld.split`` if present.
        #   3. Default to ``"train"`` -- the upstream main_ppo.create_rl_dataset
        #      path passes ``is_train`` separately but does NOT forward it to
        #      dataset __init__, so train and val datasets constructed this way
        #      would both be "train" unless we do our own splitting. Users who
        #      want valid_seen/valid_unseen should do a second programmatic
        #      instantiation or wire a two-call create path.
        if split is None and config is not None:
            alf_section = None
            if hasattr(config, "get"):
                alf_section = config.get("alfworld", None)
            elif isinstance(config, dict):
                alf_section = config.get("alfworld")
            if alf_section is not None:
                if hasattr(alf_section, "get"):
                    split = alf_section.get("split", None)
                elif isinstance(alf_section, dict):
                    split = alf_section.get("split")
        # Fallback: infer from ``data_files`` path. Upstream main_ppo creates
        # train vs val datasets with different data_files paths (from
        # ``data.train_files`` vs ``data.val_files``). If the path contains
        # "val" we pick ``valid_seen``, otherwise ``train``. This makes
        # ``val_before_train=True`` work without patching upstream.
        if split is None and data_files is not None:
            df_str = str(data_files).lower()
            if "val" in df_str:
                split = "valid_seen"
        if split is None:
            split = "train"
        if split not in self._SPLIT_KEYS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {list(self._SPLIT_KEYS)}"
            )

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.split = split
        self.max_samples = int(max_samples)

        # Resolve the TextWorld yaml path. Priority:
        #   1. Explicit kwarg (set by programmatic callers / tests).
        #   2. ``config.alf_config_path`` / ``config.custom_cls.alf_config_path``.
        #   3. ``config_tw.yaml`` sitting next to this file.
        resolved = alf_config_path
        if resolved is None and config is not None:
            if hasattr(config, "get"):
                resolved = config.get("alf_config_path", None)
            elif isinstance(config, dict):
                resolved = config.get("alf_config_path")
        if resolved is None:
            resolved = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config_tw.yaml"
            )
        self.alf_config_path = _expand(resolved)
        assert os.path.exists(self.alf_config_path), (
            f"TextWorld config not found: {self.alf_config_path}"
        )

        with open(self.alf_config_path) as f:
            alf_cfg = yaml.safe_load(f) or {}
        dataset_section = alf_cfg.get("dataset", {}) or {}
        split_path = _expand(dataset_section.get(self._SPLIT_KEYS[split]))
        if split_path is None:
            raise ValueError(
                f"split {split!r} key {self._SPLIT_KEYS[split]!r} not present in "
                f"{self.alf_config_path}"
            )
        if not os.path.isdir(split_path):
            raise ValueError(
                f"split path does not exist or is not a directory: {split_path}"
            )
        self.split_path = split_path

        self.gamefiles: list[str] = _enumerate_gamefiles(split_path)
        if not self.gamefiles:
            raise ValueError(
                f"no *.tw-pddl gamefiles under split path {split_path!r}"
            )
        if self.max_samples > 0:
            self.gamefiles = self.gamefiles[: self.max_samples]

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.gamefiles)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        gamefile = self.gamefiles[idx]
        row: dict[str, Any] = {
            "raw_prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": "You will play ALFWorld.",
                    }
                ],
                dtype=object,
            ),
            "extra_info": {
                "gamefile": gamefile,
                "split": self.split,
                "index": int(idx),
                "alf_config_path": self.alf_config_path,
                # Route val split to a separate AlfWorldEnvPool keyed on
                # is_train=False so AlfredTWEnv loads valid_seen/valid_unseen
                # games rather than the default train scheduler. Also bump
                # seed_base to mirror verl-agent's eval-seed convention
                # (train seed vs train seed + 1000).
                "is_train": self.split == "train",
                "seed_base": 42 if self.split == "train" else 1042,
            },
            "agent_name": "alfworld",
            "index": int(idx),
            # Keep the schema RLHFDataset-compatible: upstream agent loops
            # look up tools_kwargs / interaction_kwargs on every row.
            "tools_kwargs": {},
            "interaction_kwargs": {},
            # Dummy tensor keeps ``DataProto.batch`` non-empty (mirrors
            # RLHFDataset.__getitem__ line 374).
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
        return row

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "split_path": self.split_path,
            "alf_config_path": self.alf_config_path,
            "num_gamefiles": len(self.gamefiles),
        }
