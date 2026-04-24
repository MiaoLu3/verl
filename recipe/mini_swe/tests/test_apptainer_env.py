"""Integration tests for :class:`ApptainerEnvironment`.

These tests require a working ``apptainer`` binary and a pre-built SIF image.
If either is missing, every test in this module is skipped. Select with
``-m integration`` (or ``--no-integration`` equivalent) as needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from recipe.mini_swe.environments.apptainer import (
    ApptainerEnvironment,
    ApptainerEnvironmentConfig,
)

SIF_PATH = "/scratch/m000069-pm05/miaolu/apptainer-verify/sweb.django_1776_django-11099.sif"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("apptainer") is None,
        reason="apptainer binary not available on this host",
    ),
    pytest.mark.skipif(
        not os.path.exists(SIF_PATH),
        reason=f"SIF image not found at {SIF_PATH}",
    ),
]


def _list_instance_names() -> list[str]:
    """Return the set of currently-running apptainer instance names."""
    result = subprocess.run(
        ["apptainer", "instance", "list"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    names: list[str] = []
    # Output looks like:
    #   INSTANCE NAME    PID    IP    IMAGE
    #   mswe_abc123def456 12345        /path/to.sif
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


@pytest.fixture
def env():
    """Yield a fresh :class:`ApptainerEnvironment` and tear it down after."""
    e = ApptainerEnvironment(sif_path=SIF_PATH)
    try:
        yield e
    finally:
        e.close()


def test_smoke(env):
    """Construct, run one command, verify output and returncode."""
    result = env.execute({"command": "echo hello"})
    assert result["returncode"] == 0, result
    assert "hello" in result["output"], result


def test_state_persistence(env):
    """State set in one ``execute`` must be visible in the next.

    This exercises the core design promise of using a long-lived
    ``apptainer instance`` instead of repeated ``apptainer exec --writable-tmpfs``
    invocations.
    """
    r1 = env.execute({"command": "echo persist > /testbed/.persist_test"})
    assert r1["returncode"] == 0, r1

    r2 = env.execute({"command": "cat /testbed/.persist_test"})
    assert r2["returncode"] == 0, r2
    assert "persist" in r2["output"], r2


def test_conda_env_active(env):
    """The conda env activation fragment must wire up ``django``."""
    result = env.execute(
        {"command": "python -c 'import django; print(django.get_version())'"}
    )
    assert result["returncode"] == 0, result
    # Django version strings look like "3.0.dev20190424160632" or similar;
    # check that the output is non-empty and looks version-ish.
    first_line = result["output"].strip().splitlines()[0] if result["output"].strip() else ""
    assert first_line, f"expected version string, got {result!r}"
    assert any(ch.isdigit() for ch in first_line), result


def test_timeout():
    """A command exceeding the timeout returns 124 without raising."""
    env = ApptainerEnvironment(sif_path=SIF_PATH, timeout=2)
    try:
        result = env.execute({"command": "sleep 5"})
        assert result["returncode"] == 124, result
        assert "TimeoutExpired" in result["output"], result
    finally:
        env.close()


def test_patch_bind_mount(env):
    """Files written on the host to the patch scratch dir are visible in the container."""
    scratch = env.config.patch_scratch_dir
    assert scratch is not None and os.path.isdir(scratch)

    patch_path = os.path.join(scratch, "foo.patch")
    patch_body = "--- a/foo\n+++ b/foo\n@@ -0,0 +1 @@\n+hello from patch\n"
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_body)

    result = env.execute({"command": "cat /mswe/patches/foo.patch"})
    assert result["returncode"] == 0, result
    assert "hello from patch" in result["output"], result


def test_cleanup():
    """After ``close()``, the instance must no longer appear in ``instance list``."""
    env = ApptainerEnvironment(sif_path=SIF_PATH)
    name = env.instance_name
    assert name in _list_instance_names(), (
        f"instance {name} should be running after construction"
    )
    env.close()
    assert name not in _list_instance_names(), (
        f"instance {name} should be gone after close()"
    )


def test_double_close():
    """Calling ``close`` twice is a no-op and must not raise."""
    env = ApptainerEnvironment(sif_path=SIF_PATH)
    env.close()
    env.close()  # must not raise
