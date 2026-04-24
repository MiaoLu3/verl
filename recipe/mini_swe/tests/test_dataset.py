"""Unit tests for the thin :class:`SweBenchDataset` shell.

These tests do NOT hit the HuggingFace hub: we monkeypatch
``DATASET_ADAPTERS["swe_bench_lite"]`` with a fake adapter that returns
hand-crafted :class:`NormalizedRow`s. Mirrors the approach in
``test_swebench_pull.py``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

# This test module drives SweBenchDataset which inherits from
# torch.utils.data.Dataset; skip the whole file on environments without torch
# (e.g. a minimal test shell) rather than fail on collection.
torch = pytest.importorskip("torch")

from recipe.mini_swe import dataset as dataset_module
from recipe.mini_swe.dataset import SweBenchDataset
from recipe.mini_swe.dataset_adapters import NormalizedRow
from recipe.mini_swe.dataset_adapters.swe_bench_runners import TestRunnerSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    instance_id: str,
    *,
    with_runner: bool = True,
) -> NormalizedRow:
    runner = (
        TestRunnerSpec(
            shell_cmd_template="./tests/runtests.py --verbosity 2 "
            "--settings=test_sqlite {tests}",
            outcome_parser="django_runtests",
            normalize_test_ids=True,
        )
        if with_runner
        else None
    )
    return NormalizedRow(
        instance_id=instance_id,
        repo="django/django",
        problem_statement=f"fix {instance_id}",
        base_commit="deadbeef",
        image_uri=f"docker://swebench/{instance_id}:latest",
        sif_path=f"/sif/{instance_id}.sif",
        fail_to_pass=["test_a (mod.Class)"],
        pass_to_pass=["test_b (mod.Class)"],
        install_spec="",
        test_runner=runner,
        raw=None,
    )


class _FakeAdapter:
    """Stand-in for SweBenchAdapter that avoids any HF load_dataset call."""

    def __init__(self, *, rows: list[NormalizedRow], **_kwargs: Any):
        # Mirror BaseAdapter's attributes that describe() / downstream
        # diagnostics touch.
        self._rows = rows
        self.split = _kwargs.get("split", "test")
        self.sif_cache_dir = _kwargs.get("sif_cache_dir", "")

    @classmethod
    def bind(cls, rows: list[NormalizedRow]):
        return lambda **kw: cls(rows=rows, **kw)

    def load(self) -> list[NormalizedRow]:
        return list(self._rows)


@pytest.fixture
def patch_adapter(monkeypatch):
    """Replace DATASET_ADAPTERS['swe_bench_lite'] with a fake that returns rows."""

    def _apply(rows: list[NormalizedRow]):
        monkeypatch.setitem(
            dataset_module.DATASET_ADAPTERS,
            "swe_bench_lite",
            _FakeAdapter.bind(rows),
        )

    return _apply


# ---------------------------------------------------------------------------
# 1. len matches adapter.load() row count
# ---------------------------------------------------------------------------


def test_len_matches_adapter_rows(patch_adapter):
    rows = [_make_row(f"org__repo-{i}") for i in range(3)]
    patch_adapter(rows)
    ds = SweBenchDataset(dataset_name="swe_bench_lite")
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# 2. __getitem__ returns all required top-level fields
# ---------------------------------------------------------------------------


def test_getitem_has_required_fields(patch_adapter):
    patch_adapter([_make_row("org__repo-1")])
    ds = SweBenchDataset(dataset_name="swe_bench_lite")
    item = ds[0]
    required = {
        "raw_prompt",
        "extra_info",
        "agent_name",
        "index",
        "tools_kwargs",
        "interaction_kwargs",
        "dummy_tensor",
    }
    assert required.issubset(item.keys()), sorted(set(item.keys()))
    assert item["agent_name"] == "mini_swe"
    assert item["index"] == 0
    assert isinstance(item["raw_prompt"], np.ndarray)
    assert item["raw_prompt"].dtype == object
    # At least one message-dict with role=user
    assert item["raw_prompt"].shape == (1,)
    assert isinstance(item["raw_prompt"][0], dict)
    assert item["raw_prompt"][0]["role"] == "user"
    assert item["tools_kwargs"] == {}
    assert item["interaction_kwargs"] == {}
    assert isinstance(item["dummy_tensor"], torch.Tensor)
    assert item["dummy_tensor"].dtype == torch.uint8


# ---------------------------------------------------------------------------
# 3. extra_info keys are normalized / lowercase (no uppercase FAIL_TO_PASS)
# ---------------------------------------------------------------------------


def test_extra_info_normalized_keys(patch_adapter):
    patch_adapter([_make_row("org__repo-1")])
    ds = SweBenchDataset(dataset_name="swe_bench_lite")
    item = ds[0]
    extra = item["extra_info"]
    expected = {
        "instance_id",
        "repo",
        "problem_statement",
        "base_commit",
        "sif_path",
        "fail_to_pass",
        "pass_to_pass",
        "install_spec",
        "test_runner",
        "index",
    }
    assert expected.issubset(extra.keys())
    # No uppercase aliases leaked through.
    assert "FAIL_TO_PASS" not in extra
    assert "PASS_TO_PASS" not in extra
    # Payloads line up with the fabricated row.
    assert extra["instance_id"] == "org__repo-1"
    assert extra["repo"] == "django/django"
    assert extra["base_commit"] == "deadbeef"
    assert extra["sif_path"] == "/sif/org__repo-1.sif"
    assert extra["fail_to_pass"] == ["test_a (mod.Class)"]
    assert extra["pass_to_pass"] == ["test_b (mod.Class)"]
    assert extra["index"] == 0


# ---------------------------------------------------------------------------
# 4. test_runner is serialized as a plain dict
# ---------------------------------------------------------------------------


def test_test_runner_serialized_as_dict(patch_adapter):
    patch_adapter([_make_row("org__repo-1", with_runner=True)])
    ds = SweBenchDataset(dataset_name="swe_bench_lite")
    item = ds[0]
    runner = item["extra_info"]["test_runner"]
    assert isinstance(runner, dict)
    # The core four fields must be present.
    for k in ("shell_cmd_template", "test_id_separator", "pre_cmd", "outcome_parser"):
        assert k in runner, k
    assert runner["outcome_parser"] == "django_runtests"
    # And the django normalize flag must survive the round-trip.
    assert runner.get("normalize_test_ids") is True
    # Rehydration via TestRunnerSpec(**runner) must succeed (this is what
    # MiniSweAgentLoop.run does at rollout time).
    spec = TestRunnerSpec(**runner)
    assert isinstance(spec, TestRunnerSpec)
    assert spec.outcome_parser == "django_runtests"


# ---------------------------------------------------------------------------
# 5. test_runner is None when the adapter row has no runner
# ---------------------------------------------------------------------------


def test_test_runner_none_when_absent(patch_adapter):
    patch_adapter([_make_row("org__repo-1", with_runner=False)])
    ds = SweBenchDataset(dataset_name="swe_bench_lite")
    item = ds[0]
    assert item["extra_info"]["test_runner"] is None


# ---------------------------------------------------------------------------
# 6. Unknown dataset_name raises KeyError at construction time
# ---------------------------------------------------------------------------


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        SweBenchDataset(dataset_name="bogus_dataset_name_does_not_exist")


# ---------------------------------------------------------------------------
# 7. CLI --help smoke (subprocess so we catch import / argparse regressions)
# ---------------------------------------------------------------------------


def test_cli_smoke():
    proc = subprocess.run(
        [sys.executable, "-m", "recipe.mini_swe.dataset", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"CLI --help exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "dataset-name" in proc.stdout.lower() or "dataset-name" in proc.stdout
