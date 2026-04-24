"""Deterministic reward computation for mini-swe-agent rollouts on SWE-bench.

M3 of the mini-swe-agent x verl integration. Given a patch produced by the
agent and a long-lived :class:`ApptainerEnvironment`, this module resets the
in-container repository, attempts to apply the patch, runs the relevant test
lists via a :class:`TestRunnerSpec`, and produces a 0.0 / 1.0 scalar reward.

Apply / test failures return ``0.0`` rather than raising. Only unexpected
infrastructure problems (e.g. reset fails completely) raise
:class:`RewardError`.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from .dataset_adapters.swe_bench_runners import TestRunnerSpec
from .environments.apptainer import ApptainerEnvironment

logger = logging.getLogger(__name__)


class RewardError(Exception):
    """Raised for unexpected reward-pipeline failures (reset, etc.).

    Apply / test failures return ``0.0`` instead of raising.
    """


def reset_repo_to_base(env: ApptainerEnvironment, base_commit: str) -> None:
    """Revert tracked files AND any pre-applied gold patch.

    DO NOT ``git clean -fdx``. That destroys compiled .so / *.egg-info baked
    into the testbed env at image build time for repos like astropy /
    scikit-learn / matplotlib. We only need tracked files reset.

    Falls back to ``git checkout -f base_commit`` if reset fails (handles
    the case where HEAD is not descended from ``base_commit``).
    """
    cmd = f"git reset --hard {shlex.quote(base_commit)}"
    r = env.execute({"command": cmd})
    if r["returncode"] != 0:
        # Fallback: detached-HEAD style checkout
        r2 = env.execute(
            {"command": f"git checkout -f {shlex.quote(base_commit)}"}
        )
        if r2["returncode"] != 0:
            raise RewardError(
                f"reset+checkout failed: {r['output']} | {r2['output']}"
            )


def apply_patch(
    env: ApptainerEnvironment,
    patch_text: str,
    patch_name: str = "submission.patch",
) -> tuple[bool, str]:
    """Apply ``patch_text`` via the pre-mounted host scratch dir.

    Returns ``(ok, output)``. ``ok`` is ``True`` iff ``git apply`` exits 0.
    """
    if not patch_text or not patch_text.strip():
        return False, "empty patch"
    scratch = env.config.patch_scratch_dir
    if not scratch or not os.path.isdir(scratch):
        return False, "apptainer env has no patch_scratch_dir; cannot apply"
    host_path = Path(scratch) / patch_name
    host_path.write_text(patch_text)
    # In-container path matches the bind: <scratch> -> /mswe/patches
    container_path = f"/mswe/patches/{patch_name}"
    check = env.execute(
        {"command": f"git apply --check {shlex.quote(container_path)}"}
    )
    if check["returncode"] != 0:
        return False, f"git apply --check failed: {check['output']}"
    apply_ = env.execute(
        {"command": f"git apply {shlex.quote(container_path)}"}
    )
    ok = apply_["returncode"] == 0
    return ok, apply_["output"]


def run_tests(
    env: ApptainerEnvironment,
    test_ids: list[str],
    runner: TestRunnerSpec,
    timeout: int = 300,
) -> tuple[int, int]:
    """Run ``test_ids`` via ``runner``, parse outcome with ``runner.parse_outcome``.

    Returns ``(passed, total)``. If ``test_ids`` is empty, returns ``(0, 0)``
    without calling the environment.
    """
    if not test_ids:
        return 0, 0
    cmd = runner.build_command(test_ids)
    prev = env.config.timeout
    try:
        env.config.timeout = max(timeout, prev)
        r = env.execute({"command": cmd})
    finally:
        env.config.timeout = prev
    return runner.parse_outcome(r["output"], test_ids)


def score_patch(
    patch_text: str,
    env: ApptainerEnvironment,
    base_commit: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    runner: TestRunnerSpec | None,
) -> float:
    """Deterministic 0.0 / 1.0 scoring. No partial credit in M3.

    Flow:
        1. Reset the repo to ``base_commit`` (raising returns 0.0).
        2. Apply ``patch_text`` (non-applying returns 0.0).
        3. If ``runner`` is ``None``, return 0.0 (no way to evaluate).
        4. Run FAIL_TO_PASS; every one must pass.
        5. Run PASS_TO_PASS; every one must pass.
        6. If both lists are empty, return 0.0 (safer default).
        7. Otherwise return 1.0.
    """
    try:
        reset_repo_to_base(env, base_commit)
    except RewardError as e:
        logger.warning("reward: reset failed, returning 0.0 (%s)", e)
        return 0.0
    ok, _msg = apply_patch(env, patch_text)
    if not ok:
        logger.info("reward: patch did not apply, returning 0.0")
        return 0.0
    if runner is None:
        logger.warning(
            "reward: no TestRunnerSpec for this instance; returning 0.0"
        )
        return 0.0
    f2p_pass, f2p_total = run_tests(env, fail_to_pass, runner)
    if f2p_total > 0 and f2p_pass != f2p_total:
        return 0.0
    p2p_pass, p2p_total = run_tests(env, pass_to_pass, runner)
    if p2p_total > 0 and p2p_pass != p2p_total:
        return 0.0
    # If both were empty, something's wrong - but treat as 0.0 not 1.0 (safer)
    if f2p_total == 0 and p2p_total == 0:
        return 0.0
    return 1.0
