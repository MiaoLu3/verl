"""Unit tests for the SIF batch pre-puller.

No real HF calls, no real apptainer pulls.  We fabricate NormalizedRow
instances inline and monkeypatch both DATASET_ADAPTERS["swe_bench_lite"]
and subprocess.run.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from unittest import mock

import pytest

from recipe.mini_swe.dataset_adapters import NormalizedRow
from recipe.mini_swe.scripts import swebench_pull


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(instance_id: str) -> NormalizedRow:
    return NormalizedRow(
        instance_id=instance_id,
        repo="some/repo",
        problem_statement="...",
        base_commit="deadbeef",
        image_uri=f"docker://swebench/{instance_id}:latest",
        sif_path=f"/unused/{instance_id}.sif",
        fail_to_pass=[],
        pass_to_pass=[],
        install_spec="",
        test_runner=None,
        raw=None,
    )


ROW_A = _make_row("org__repo-1")
ROW_B = _make_row("org__repo-2")
ROW_C = _make_row("org__repo-3")


class _FakeAdapter:
    """Stand-in for SweBenchAdapter that avoids any HF load_dataset call."""

    def __init__(self, *, rows: list[NormalizedRow], **_kwargs):
        self._rows = rows

    @classmethod
    def bind(cls, rows):
        return lambda **kw: cls(rows=rows, **kw)

    def load(self) -> list[NormalizedRow]:
        return list(self._rows)


@pytest.fixture
def patch_adapter(monkeypatch):
    """Replace DATASET_ADAPTERS['swe_bench_lite'] with a fake that returns rows."""

    def _apply(rows: list[NormalizedRow]):
        monkeypatch.setitem(
            swebench_pull.DATASET_ADAPTERS,
            "swe_bench_lite",
            _FakeAdapter.bind(rows),
        )

    return _apply


# ---------------------------------------------------------------------------
# 1. CLI args parse correctly
# ---------------------------------------------------------------------------


def test_parse_args_basic(tmp_path):
    args = swebench_pull._parse_args(
        [
            "--dataset-name",
            "swe_bench_lite",
            "--cache-dir",
            str(tmp_path),
            "--max",
            "3",
        ]
    )
    assert args.dataset_name == "swe_bench_lite"
    assert args.cache_dir == str(tmp_path)
    assert args.max_rows == 3
    assert args.index is None
    assert args.instance_ids is None
    assert args.concurrency == 1
    assert args.min_sif_bytes == 10_000_000
    assert args.dry_run is False
    assert args.apptainer_bin == "apptainer"
    assert args.apptainer_cachedir is None
    assert args.timeout_per_pull == 1800


def test_parse_args_index_and_instance_ids(tmp_path):
    args = swebench_pull._parse_args(
        [
            "--dataset-name",
            "swe_bench_lite",
            "--cache-dir",
            str(tmp_path),
            "--index",
            "7",
            "--instance-id",
            "foo__bar-1",
            "--instance-id",
            "foo__bar-2",
            "--concurrency",
            "4",
            "--dry-run",
            "--apptainer-bin",
            "/opt/apptainer/bin/apptainer",
            "--apptainer-cachedir",
            "/tmp/appcache",
            "--timeout-per-pull",
            "600",
            "--min-sif-bytes",
            "42",
            "--split",
            "test",
        ]
    )
    assert args.index == 7
    assert args.instance_ids == ["foo__bar-1", "foo__bar-2"]
    assert args.concurrency == 4
    assert args.dry_run is True
    assert args.apptainer_bin == "/opt/apptainer/bin/apptainer"
    assert args.apptainer_cachedir == "/tmp/appcache"
    assert args.timeout_per_pull == 600
    assert args.min_sif_bytes == 42
    assert args.split == "test"


def test_parse_args_requires_dataset_and_cache(tmp_path, capsys):
    with pytest.raises(SystemExit):
        swebench_pull._parse_args([])
    with pytest.raises(SystemExit):
        swebench_pull._parse_args(
            ["--dataset-name", "swe_bench_lite"]
        )
    with pytest.raises(SystemExit):
        swebench_pull._parse_args(
            ["--cache-dir", str(tmp_path)]
        )


# ---------------------------------------------------------------------------
# 2. filter_by_index
# ---------------------------------------------------------------------------


def test_filter_by_index_middle():
    rows = [ROW_A, ROW_B, ROW_C]
    selected = swebench_pull._filter_by_index(rows, 1)
    assert selected == [ROW_B]


def test_filter_by_index_out_of_range():
    rows = [ROW_A, ROW_B, ROW_C]
    with pytest.raises(IndexError):
        swebench_pull._filter_by_index(rows, 99)


# ---------------------------------------------------------------------------
# 3. filter_by_instance_id
# ---------------------------------------------------------------------------


def test_filter_by_instance_id_subset():
    rows = [ROW_A, ROW_B, ROW_C]
    selected = swebench_pull._filter_by_instance_id(
        rows, [ROW_A.instance_id, ROW_B.instance_id]
    )
    selected_ids = {r.instance_id for r in selected}
    assert selected_ids == {ROW_A.instance_id, ROW_B.instance_id}


def test_filter_by_instance_id_missing_is_warned(caplog):
    rows = [ROW_A, ROW_B]
    with caplog.at_level("WARNING", logger="swebench_pull"):
        selected = swebench_pull._filter_by_instance_id(
            rows, [ROW_A.instance_id, "missing__nope-99"]
        )
    assert [r.instance_id for r in selected] == [ROW_A.instance_id]
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "missing__nope-99" in joined


# ---------------------------------------------------------------------------
# 4. skip_when_exists
# ---------------------------------------------------------------------------


def test_skip_when_exists(tmp_path, patch_adapter, capsys):
    patch_adapter([ROW_A])
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / f"{ROW_A.instance_id}.sif"
    # Write a dummy file > min-sif-bytes.
    target.write_bytes(b"x" * 2048)

    fail_loud = mock.Mock(side_effect=AssertionError("apptainer must not run"))
    with mock.patch.object(subprocess, "run", fail_loud):
        rc = swebench_pull.main(
            [
                "--dataset-name",
                "swe_bench_lite",
                "--cache-dir",
                str(cache),
                "--min-sif-bytes",
                "1024",
            ]
        )
    assert rc == 0
    fail_loud.assert_not_called()
    captured = capsys.readouterr().out
    assert "SKIP" in captured
    assert ROW_A.instance_id in captured


# ---------------------------------------------------------------------------
# 5. lock_contention
# ---------------------------------------------------------------------------


def test_lock_contention_serializes(tmp_path, patch_adapter):
    """Two threads both race to pull the same SIF. flock serializes them;
    the second thread's size-check kicks in and it returns 'skipped'."""

    patch_adapter([ROW_A])
    cache = tmp_path / "cache"
    cache.mkdir()
    apptainer_cachedir = tmp_path / "appcache"
    apptainer_cachedir.mkdir()
    min_bytes = 1024

    call_count = {"n": 0}

    def fake_run(cmd, check, timeout, env):
        # cmd = [apptainer, pull, --force, tmp_path, image_uri]
        call_count["n"] += 1
        # Simulate a slow pull so the other thread has time to contend on
        # the lock.
        time.sleep(0.3)
        tmp_target = cmd[3]
        with open(tmp_target, "wb") as f:
            f.write(b"y" * (min_bytes + 16))

        class _R:
            returncode = 0

        return _R()

    results: list[swebench_pull.PullResult] = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            res = swebench_pull._pull_one(
                ROW_A,
                cache_dir=str(cache),
                min_sif_bytes=min_bytes,
                dry_run=False,
                apptainer_bin="apptainer",
                apptainer_cachedir=str(apptainer_cachedir),
                timeout_per_pull=60,
            )
        results.append(res)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    statuses = sorted(r.status for r in results)
    assert statuses == ["pulled", "skipped"], statuses
    # Only one thread actually invoked the apptainer subprocess.
    assert call_count["n"] == 1
    # And the final target exists with the written content.
    final = cache / f"{ROW_A.instance_id}.sif"
    assert final.exists()
    assert final.stat().st_size >= min_bytes


# ---------------------------------------------------------------------------
# 6. dry_run
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_subprocess_calls(tmp_path, patch_adapter, capsys):
    patch_adapter([ROW_A, ROW_B])
    cache = tmp_path / "cache"

    fail_loud = mock.Mock(side_effect=AssertionError("must not run on dry-run"))
    with mock.patch.object(subprocess, "run", fail_loud):
        rc = swebench_pull.main(
            [
                "--dataset-name",
                "swe_bench_lite",
                "--cache-dir",
                str(cache),
                "--dry-run",
            ]
        )
    assert rc == 0
    fail_loud.assert_not_called()
    out = capsys.readouterr().out
    assert "WOULD PULL" in out
    assert ROW_A.image_uri in out
    assert ROW_B.image_uri in out
    # And no actual SIFs were written.
    assert not (cache / f"{ROW_A.instance_id}.sif").exists()
    assert not (cache / f"{ROW_B.instance_id}.sif").exists()


# ---------------------------------------------------------------------------
# 7. atomic_rename
# ---------------------------------------------------------------------------


def test_atomic_rename_on_success(tmp_path, patch_adapter):
    patch_adapter([ROW_A])
    cache = tmp_path / "cache"
    cache.mkdir()
    payload = b"z" * 4096

    def fake_run(cmd, check, timeout, env):
        # The CLI passes [bin, "pull", "--force", tmp_path, image_uri]
        tmp_target = cmd[3]
        with open(tmp_target, "wb") as f:
            f.write(payload)

        class _R:
            returncode = 0

        return _R()

    with mock.patch.object(subprocess, "run", side_effect=fake_run):
        rc = swebench_pull.main(
            [
                "--dataset-name",
                "swe_bench_lite",
                "--cache-dir",
                str(cache),
                "--min-sif-bytes",
                "1024",
            ]
        )
    assert rc == 0

    final = cache / f"{ROW_A.instance_id}.sif"
    tmp = cache / f"{ROW_A.instance_id}.sif.tmp"
    assert final.exists()
    assert final.read_bytes() == payload
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# 8. Failure path: .tmp is cleaned up, exit code non-zero
# ---------------------------------------------------------------------------


def test_failure_cleans_tmp_and_nonzero_exit(tmp_path, patch_adapter):
    patch_adapter([ROW_A])
    cache = tmp_path / "cache"
    cache.mkdir()

    def fake_run(cmd, check, timeout, env):
        # Write a partial .tmp to ensure we verify it's cleaned up.
        tmp_target = cmd[3]
        with open(tmp_target, "wb") as f:
            f.write(b"partial")
        raise subprocess.CalledProcessError(1, cmd)

    with mock.patch.object(subprocess, "run", side_effect=fake_run):
        rc = swebench_pull.main(
            [
                "--dataset-name",
                "swe_bench_lite",
                "--cache-dir",
                str(cache),
                "--min-sif-bytes",
                "1024",
            ]
        )
    assert rc != 0
    assert not (cache / f"{ROW_A.instance_id}.sif").exists()
    assert not (cache / f"{ROW_A.instance_id}.sif.tmp").exists()
