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
"""Thin SWE-bench dataset shell for the upstream verl agent-loop framework.

All per-dataset logic lives in ``recipe.mini_swe.dataset_adapters``; this
file is just the glue between verl's ``RLHFDataset``-compatible contract and
a ``BaseAdapter`` instance. Each row is a normalized SWE-bench instance:

* ``raw_prompt`` - placeholder user message; the real initial prompt
  (system + instance_template) is rendered inside
  :meth:`MiniSweAgentLoop.run` from the vendored ``swebench_v2.yaml``. The
  placeholder is kept because ``AgentLoopWorker`` forwards ``raw_prompt``
  to the loop via kwargs and also surfaces it on the output for logging,
  but our loop doesn't use the dataset's text.
* ``extra_info`` - per-sample fields consumed by
  :meth:`MiniSweAgentLoop.run` (instance_id, repo, base_commit, sif_path,
  fail_to_pass, pass_to_pass, install_spec, problem_statement,
  test_runner, index).
* ``agent_name`` == ``"mini_swe"`` - routes to the registered
  ``MiniSweAgentLoop`` in ``configs/agent_loops.yaml``.

Design note on ``extra_info.test_runner``:
  :class:`~recipe.mini_swe.dataset_adapters.swe_bench_runners.TestRunnerSpec`
  is a plain dataclass and pickles fine, but verl's agent-loop worker shuttles
  the non_tensor_batch through Ray's object store and across process
  boundaries. To keep the row JSON-safe (and so it survives ``repr`` / logging
  round-trips without requiring callers to re-import ``TestRunnerSpec``) we
  serialize the spec to a plain dict here, and :meth:`MiniSweAgentLoop.run`
  rehydrates it via ``TestRunnerSpec(**runner_dict)`` at rollout time.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from recipe.mini_swe.dataset_adapters import DATASET_ADAPTERS, NormalizedRow

# Importing ``swe_bench`` triggers ``@register_adapter`` decorators and
# populates ``DATASET_ADAPTERS`` with ``swe_bench_lite`` /
# ``swe_bench_verified`` / ``swe_bench_full``. Side-effect import only.
from recipe.mini_swe.dataset_adapters import swe_bench  # noqa: F401


def _runner_to_dict(runner: Any) -> Optional[dict]:
    """Serialize a :class:`TestRunnerSpec` (or ``None``) to a plain dict.

    Kept tolerant of missing attributes so older pre-M5 TestRunnerSpec
    instances (without ``normalize_test_ids``) don't crash the dataset.
    """
    if runner is None:
        return None
    return {
        "shell_cmd_template": runner.shell_cmd_template,
        "test_id_separator": runner.test_id_separator,
        "pre_cmd": runner.pre_cmd,
        "outcome_parser": runner.outcome_parser,
        "normalize_test_ids": getattr(runner, "normalize_test_ids", False),
    }


class SweBenchDataset(Dataset):
    """File-free SWE-bench dataset that streams rows from a registered
    :class:`BaseAdapter`.

    Args:
        config: omega/dict config (accepted for API compat with
            ``RLHFDataset`` / ``create_rl_dataset``; unused here).
        tokenizer / processor: accepted for API compat; unused.
        dataset_name: key into
            :data:`recipe.mini_swe.dataset_adapters.DATASET_ADAPTERS`.
            Defaults to ``"swe_bench_lite"``.
        split: passed through to the adapter; defaults to the adapter's
            ``default_split``.
        sif_cache_dir: directory containing the pre-pulled SIF images (one
            per instance_id). Defaults to the ``SIF_CACHE_DIR`` env var.
        instance_ids: optional list; if set, only these instance_ids are
            loaded (used for smoke tests and per-instance SLURM arrays).
        max_samples: cap on rows loaded (``-1`` = unlimited).
        filter_verified_repos: drop rows whose repo is in DeepSWE's
            contamination list.
        data_files: accepted for API compat with upstream
            ``create_rl_dataset`` (which always passes it); ignored.
        **adapter_kwargs: forwarded to the adapter constructor.
    """

    def __init__(
        self,
        config: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        dataset_name: str = "swe_bench_lite",
        split: Optional[str] = None,
        sif_cache_dir: Optional[str] = None,
        instance_ids: Optional[list[str]] = None,
        max_samples: int = -1,
        filter_verified_repos: bool = False,
        data_files: Any = None,
        **adapter_kwargs,
    ) -> None:
        _ = data_files  # unused; accepted for API compat
        if dataset_name not in DATASET_ADAPTERS:
            raise KeyError(
                f"Unknown dataset_name={dataset_name!r}; "
                f"available: {sorted(DATASET_ADAPTERS)}"
            )
        Adapter = DATASET_ADAPTERS[dataset_name]
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_name = dataset_name
        self.adapter = Adapter(
            split=split,
            instance_ids=instance_ids,
            max_samples=max_samples,
            sif_cache_dir=sif_cache_dir or os.environ.get("SIF_CACHE_DIR", ""),
            filter_verified_repos=filter_verified_repos,
            **adapter_kwargs,
        )
        self.rows: list[NormalizedRow] = self.adapter.load()

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        runner_dict = _runner_to_dict(r.test_runner)
        return {
            "raw_prompt": np.array(
                [
                    {
                        "role": "user",
                        "content": "Placeholder - see extra_info.problem_statement.",
                    }
                ],
                dtype=object,
            ),
            "extra_info": {
                "instance_id": r.instance_id,
                "repo": r.repo,
                "problem_statement": r.problem_statement,
                "base_commit": r.base_commit,
                "sif_path": r.sif_path,
                "fail_to_pass": list(r.fail_to_pass),
                "pass_to_pass": list(r.pass_to_pass),
                "install_spec": r.install_spec,
                "test_runner": runner_dict,
                "index": int(idx),
            },
            "agent_name": "mini_swe",
            "index": int(idx),
            # Keep the schema RLHFDataset-compatible: upstream agent loops
            # look up tools_kwargs / interaction_kwargs on every row.
            "tools_kwargs": {},
            "interaction_kwargs": {},
            # Dummy tensor keeps ``DataProto.batch`` non-empty, mirroring
            # RLHFDataset.__getitem__ and recipe/alfworld/alfworld_dataset.py.
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "num_rows": len(self.rows),
            "split": getattr(self.adapter, "split", None),
            "sif_cache_dir": getattr(self.adapter, "sif_cache_dir", None),
        }

    # ------------------------------------------------------------------
    # CLI preview: ``python -m recipe.mini_swe.dataset --help``
    # ------------------------------------------------------------------

    @classmethod
    def _cli(cls) -> None:
        ap = argparse.ArgumentParser(
            description="Preview the SWE-bench dataset shell (loads 1 row)."
        )
        ap.add_argument(
            "--dataset-name",
            default="swe_bench_lite",
            help="Key into DATASET_ADAPTERS (default: swe_bench_lite).",
        )
        ap.add_argument(
            "--max", type=int, default=1, dest="max_samples",
            help="max_samples to load from the adapter (default: 1).",
        )
        ap.add_argument(
            "--sif-cache-dir",
            default=os.environ.get("SIF_CACHE_DIR", "/tmp/mswe_sifs"),
            help="SIF cache dir (default: $SIF_CACHE_DIR or /tmp/mswe_sifs).",
        )
        ap.add_argument(
            "--split",
            default=None,
            help="Adapter split (default: adapter's default_split).",
        )
        a = ap.parse_args()
        ds = cls(
            dataset_name=a.dataset_name,
            max_samples=a.max_samples,
            sif_cache_dir=a.sif_cache_dir,
            split=a.split,
        )
        print(f"len={len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print(
                json.dumps(
                    {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in item["extra_info"].items()
                    },
                    indent=2,
                    default=str,
                )
            )


if __name__ == "__main__":
    SweBenchDataset._cli()
