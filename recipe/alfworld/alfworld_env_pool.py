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
"""Ray-actor based env pool for ALFWorld / TextWorld.

Motivation:
    ``probe_thread_safety.py`` showed that TextWorld's PDDL/grammar parser has
    module-level globals (``FailedToken`` etc.) so sharing a single process'
    parser state across threads corrupts it. The fix is process-level isolation
    via Ray actors -- each actor owns a separate Python process and therefore
    a separate copy of the parser globals.

Design:
    * ``AlfWorldActor`` is a ``@ray.remote`` class wrapping one
      ``AlfWorldSingleEnv`` instance. Each actor pays the ~15s gamefile-scan
      cost once at ``__init__``; all subsequent ``reset``/``step`` calls reuse
      the same env object, so per-episode overhead is the usual TextWorld
      reset cost only.
    * ``AlfWorldEnvPool`` owns ``pool_size`` actors and an ``asyncio.Queue``
      of handles. Async coroutines ``acquire()`` a handle, run an episode
      against it via ``await actor.reset.remote()`` / ``await
      actor.step.remote(...)``, then ``release()`` it back to the queue.
    * We do NOT share a pre-built ``base_env`` via ``ray.put`` -- ``AlfredTWEnv``
      holds C++ references that don't pickle cleanly and ``init_env`` has to
      run per-actor anyway.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import ray

logger = logging.getLogger(__name__)

# The import of AlfWorldSingleEnv happens *inside* AlfWorldActor.__init__
# so the heavy alfworld/textworld dependencies only load in the actor
# subprocess, not in the driver.


@ray.remote
class AlfWorldActor:
    """One-process wrapper around a single ``AlfWorldSingleEnv``.

    Each actor runs in its own Ray worker process, guaranteeing isolation of
    TextWorld's parser globals across the pool.
    """

    def __init__(
        self,
        alf_config_path: str,
        seed: int,
        is_train: bool = True,
        history_length: int = 2,
        env_kwargs: dict | None = None,
    ) -> None:
        # Imported lazily so this module can be imported on the driver without
        # pulling in alfworld/textworld.
        from recipe.alfworld.alfworld_env_wrapper import AlfWorldSingleEnv

        self.env = AlfWorldSingleEnv(
            alf_config_path=alf_config_path,
            seed=int(seed),
            is_train=bool(is_train),
            history_length=int(history_length),
            env_kwargs=env_kwargs or {},
        )

    # --- core gym-like API, each returns Ray object refs over the wire -----

    def reset(self, gamefile: str | None = None) -> tuple[str, list[str], dict]:
        return self.env.reset(gamefile=gamefile)

    def step(self, action: str) -> tuple[str, list[str], float, bool, dict]:
        return self.env.step(action)

    def render_prompt(
        self, current_obs: str, admissible: list[str], step: int
    ) -> str:
        return self.env.render_prompt(current_obs, admissible, step)

    def get_gamefile(self) -> str:
        return getattr(self.env, "gamefile", "") or ""

    def is_alive(self) -> bool:
        """Health probe used by the pool.

        Returns True iff the wrapped ``AlfWorldSingleEnv`` and its underlying
        TextWorld env are both non-None. This is still a lightweight check (no
        reset/step is performed) but catches a half-dead actor whose inner env
        was closed or failed to initialise. The fact that this call returns at
        all also doubles as the Ray-actor ``__init__`` warmup barrier.
        """
        try:
            return self.env is not None and getattr(self.env, "env", None) is not None
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class AlfWorldEnvPool:
    """Process-local pool of ``AlfWorldActor`` handles.

    Use ``async with pool.acquire_ctx() as actor:`` or ``acquire()``/``release()``
    pair directly. ``actor`` is a Ray actor handle; call methods as
    ``await actor.reset.remote()`` / ``await actor.step.remote(action)``.
    """

    def __init__(
        self,
        alf_config_path: str,
        pool_size: int,
        seed_base: int = 0,
        is_train: bool = True,
        history_length: int = 2,
        env_kwargs: dict | None = None,
        resources_per_actor: dict | None = None,
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")

        self.alf_config_path = alf_config_path
        self.pool_size = int(pool_size)
        self.seed_base = int(seed_base)
        self.is_train = bool(is_train)
        self.history_length = int(history_length)
        self.env_kwargs = env_kwargs or {}

        # Default per-actor resources: fractional CPU (alfworld is cheap and
        # mostly C++-bound). Matches verl-agent's paper-recommended 0.1 CPU.
        self._resources = (
            {"num_cpus": 0.1} if resources_per_actor is None else dict(resources_per_actor)
        )

        # Monotonic counter used to hand out fresh seeds when respawning a
        # dead actor in ``release()``. The initial warmup loop uses
        # ``seed_base + i`` for i in [0, pool_size); subsequent respawns pull
        # seeds from this counter so they never collide with the originals.
        self._next_seed_counter = self.seed_base + self.pool_size

        if not ray.is_initialized():
            # Callers generally init Ray before constructing the pool; fall
            # back to a local init so tests and single-machine runs work.
            ray.init(ignore_reinit_error=True)

        # Spin up actors concurrently -- Ray schedules __init__ in parallel,
        # so the effective warmup is ~1 × gamefile-scan wall-clock (given
        # enough cores).
        self._actors: list[Any] = [
            AlfWorldActor.options(**self._resources).remote(
                alf_config_path=self.alf_config_path,
                seed=self.seed_base + i,
                is_train=self.is_train,
                history_length=self.history_length,
                env_kwargs=self.env_kwargs,
            )
            for i in range(self.pool_size)
        ]

        # Block until each actor finished its (expensive) __init__.
        # is_alive() returns only after __init__ completes.
        ray.get([a.is_alive.remote() for a in self._actors])

        # Populate the async queue now that every actor is ready.
        self._queue: asyncio.Queue = asyncio.Queue()
        for a in self._actors:
            self._queue.put_nowait(a)

        self._closed = False

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire(self):
        """Await a free actor handle from the pool."""
        if self._closed:
            raise RuntimeError("AlfWorldEnvPool is closed")
        return await self._queue.get()

    def _next_seed(self) -> int:
        """Allocate a fresh seed for a respawned actor."""
        seed = self._next_seed_counter
        self._next_seed_counter += 1
        return seed

    async def release(self, actor_handle) -> None:
        """Return an actor handle to the pool.

        Performs a short-timeout health probe before re-enqueuing so that a
        Ray actor that died mid-episode (the very failure mode Ray isolation
        defends against) cannot be handed back out to the next ``acquire()``.
        If the probe fails we kill the dead actor and spin up a replacement.
        """
        if self._closed:
            return

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, ray.get, actor_handle.is_alive.remote()),
                timeout=2.0,
            )
            await self._queue.put(actor_handle)
            return
        except Exception as e:
            logger.warning(
                "AlfWorldEnvPool: actor health check failed (%s: %s). "
                "Respawning a replacement actor.",
                type(e).__name__,
                e,
            )

        # Kill the dead actor best-effort.
        try:
            ray.kill(actor_handle)
        except Exception:
            pass
        try:
            self._actors.remove(actor_handle)
        except ValueError:
            pass

        # Spin up a replacement. Pay the ~15s scan cost again by waiting for
        # __init__ (via is_alive) before re-enqueueing.
        new_actor = AlfWorldActor.options(**self._resources).remote(
            alf_config_path=self.alf_config_path,
            seed=self._next_seed(),
            is_train=self.is_train,
            history_length=self.history_length,
            env_kwargs=self.env_kwargs,
        )
        await loop.run_in_executor(None, ray.get, new_actor.is_alive.remote())
        self._actors.append(new_actor)
        await self._queue.put(new_actor)

    @asynccontextmanager
    async def acquire_ctx(self):
        """Context-manager form: ``async with pool.acquire_ctx() as actor: ...``."""
        actor = await self.acquire()
        try:
            yield actor
        finally:
            await self.release(actor)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Best-effort graceful shutdown of every actor."""
        if self._closed:
            return
        self._closed = True

        # Drain the queue so no one can acquire after close.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Ask each actor to close its env, then kill the actor outright.
        for a in self._actors:
            try:
                await a.close.remote()
            except Exception:
                pass
            try:
                ray.kill(a)
            except Exception:
                pass
        self._actors = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self.pool_size

    @property
    def available(self) -> int:
        return self._queue.qsize()
