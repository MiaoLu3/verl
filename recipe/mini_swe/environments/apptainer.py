"""Apptainer-backed environment for mini-swe-agent rollouts.

This module reimplements the mini-swe-agent ``Environment`` Protocol on top of
a long-lived ``apptainer instance``. One instance is started per environment
object and lives for the duration of a rollout, so that shell state (working
directory, files on the writable tmpfs, environment variables set by earlier
commands, etc.) persists across successive :meth:`ApptainerEnvironment.execute`
calls.

The ``--writable-tmpfs`` flag passed to separate ``apptainer exec`` invocations
does NOT share state across calls. Using ``apptainer instance start`` with
``apptainer exec instance://<name>`` is the only mechanism on the target
cluster that gives us the required per-rollout persistent container state
(without the unavailable ``--fakeroot``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
import subprocess
import tempfile
import uuid
from typing import Any


@dataclasses.dataclass
class ApptainerEnvironmentConfig:
    """Configuration for :class:`ApptainerEnvironment`.

    Attributes:
        sif_path: Absolute path to the SIF image backing the environment.
        cwd: Default working directory inside the container.
        env: Extra environment variables surfaced through ``get_template_vars``.
        timeout: Default per-command timeout in seconds, used when ``execute``
            does not receive an explicit per-call timeout.
        cleanenv: Whether to pass ``--cleanenv`` to ``apptainer instance start``.
        extra_binds: Additional ``host:container[:ro]`` bind mounts applied at
            instance-start time.
        activate_fragment: Shell fragment prepended to every executed command
            (used to activate the conda env and ``cd`` into the repo).
        environment_class: Label emitted by ``serialize`` so that downstream
            code can dispatch on the environment implementation.
        apptainer_bin: Path to / name of the ``apptainer`` executable.
        patch_scratch_dir: Host directory to be bind-mounted at
            ``/mswe/patches`` inside the container. If ``None``, a per-rollout
            temporary directory is created automatically and cleaned up on
            :meth:`ApptainerEnvironment.close`.
    """

    sif_path: str
    cwd: str = "/testbed"
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    timeout: int = 120
    cleanenv: bool = True
    extra_binds: list[str] = dataclasses.field(default_factory=list)
    activate_fragment: str = (
        "source /opt/miniconda3/etc/profile.d/conda.sh && "
        "conda activate testbed && cd /testbed"
    )
    environment_class: str = "apptainer"
    apptainer_bin: str = "apptainer"
    patch_scratch_dir: str | None = None


class ApptainerEnvironment:
    """Long-lived apptainer-instance-backed execution environment.

    The constructor starts a fresh ``apptainer instance`` with a writable
    tmpfs overlay and binds a host-side scratch directory at
    ``/mswe/patches`` so that reward computation can drop patch files on the
    host and read them from inside the container via ``git apply``.
    """

    class_name = "ApptainerEnvironment"

    def __init__(
        self,
        *,
        config_class: type = ApptainerEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        # Build the dataclass config from kwargs so that callers can pass
        # loose kwargs (matching the mini-swe-agent registry pattern).
        self.config: ApptainerEnvironmentConfig = config_class(**kwargs)

        # Unique name scoped to this rollout.
        self.instance_name: str = f"mswe_{uuid.uuid4().hex[:12]}"

        # Track whether we created the patch scratch dir ourselves so we know
        # whether to clean it up in ``close``. If the caller supplied a path we
        # leave it alone.
        self._owns_patch_scratch_dir: bool = False
        if self.config.patch_scratch_dir is None:
            self.config.patch_scratch_dir = tempfile.mkdtemp(prefix="mswe_rollout_")
            self._owns_patch_scratch_dir = True

        # Assemble the bind list: user-supplied binds plus the patch scratch
        # dir bound read-write at ``/mswe/patches``.
        binds: list[str] = list(self.config.extra_binds)
        binds.append(f"{self.config.patch_scratch_dir}:/mswe/patches")

        argv: list[str] = [self.config.apptainer_bin, "instance", "start"]
        if self.config.cleanenv:
            argv.append("--cleanenv")
        argv.append("--writable-tmpfs")
        for bind in binds:
            argv.extend(["--bind", bind])
        argv.append(self.config.sif_path)
        argv.append(self.instance_name)

        self._closed: bool = False
        try:
            subprocess.run(
                argv,
                check=True,
                capture_output=True,
                timeout=60,
            )
        except Exception:
            # If ``instance start`` fails we still want to clean up the scratch
            # dir we just created, to avoid leaking host state.
            self._closed = True
            if self._owns_patch_scratch_dir and self.config.patch_scratch_dir:
                shutil.rmtree(self.config.patch_scratch_dir, ignore_errors=True)
            raise

    # ------------------------------------------------------------------ Protocol

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        """Run ``action['command']`` inside the running apptainer instance.

        Returns a dict with ``output`` (combined stdout+stderr) and
        ``returncode``. On timeout, returns ``returncode=124`` rather than
        raising.
        """
        command: str = action.get("command", "") or ""
        # Allow callers to override the default timeout via either the
        # ``action`` dict or kwargs. Fall back to the config default.
        timeout: int = int(action.get("timeout", self.config.timeout))

        full_command = f"{self.config.activate_fragment} && {command}"
        argv = [
            self.config.apptainer_bin,
            "exec",
            f"instance://{self.instance_name}",
            "bash",
            "-lc",
            full_command,
        ]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Assemble whatever partial output we may have captured before the
            # timeout fired.
            partial_stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
            partial_stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            combined = partial_stdout + partial_stderr
            return {
                "output": f"{combined}<TimeoutExpired {timeout}s>",
                "returncode": 124,
            }

        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        return {
            "output": stdout + stderr,
            "returncode": int(proc.returncode),
        }

    def get_template_vars(self, **_: Any) -> dict[str, Any]:
        return {"cwd": self.config.cwd, "env": self.config.env}

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-friendly dict describing this environment.

        The runtime-only ``instance_name`` is intentionally excluded; the
        ``patch_scratch_dir`` is included so that the caller can resolve patch
        paths on the host.
        """
        return {"environment_class": "apptainer", **dataclasses.asdict(self.config)}

    # ------------------------------------------------------------------ Lifecycle

    def close(self) -> None:
        """Tear down the apptainer instance. Idempotent."""
        if self._closed:
            return
        self._closed = True

        with contextlib.suppress(Exception):
            subprocess.run(
                [
                    self.config.apptainer_bin,
                    "instance",
                    "stop",
                    self.instance_name,
                ],
                check=False,
                capture_output=True,
                timeout=30,
            )

        if self._owns_patch_scratch_dir and self.config.patch_scratch_dir:
            shutil.rmtree(self.config.patch_scratch_dir, ignore_errors=True)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
