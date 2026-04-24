"""
Batch pre-puller for SWE-bench SIF images.

Examples:
    # Pull 3 instances from SWE-bench-Lite for smoke testing
    python -m recipe.mini_swe.scripts.swebench_pull \\
        --dataset-name swe_bench_lite --max 3 \\
        --cache-dir /scratch/m000069-pm05/miaolu/swebench_sifs

    # SLURM array task: pull one instance by index
    python -m recipe.mini_swe.scripts.swebench_pull \\
        --dataset-name swe_bench_lite --index $SLURM_ARRAY_TASK_ID \\
        --cache-dir /scratch/m000069-pm05/miaolu/swebench_sifs

    # Pull specific instance_ids (space-separated or --instance-id repeat)
    python -m recipe.mini_swe.scripts.swebench_pull \\
        --dataset-name swe_bench_lite \\
        --instance-id django__django-11099 \\
        --cache-dir /scratch/m000069-pm05/miaolu/swebench_sifs
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from recipe.mini_swe.dataset_adapters import DATASET_ADAPTERS, NormalizedRow

# Importing the concrete adapter module triggers @register_adapter decorators
# and populates DATASET_ADAPTERS with {"swe_bench_lite", "swe_bench_verified",
# "swe_bench_full"}.  The import must happen before argparse reads the choices.
from recipe.mini_swe.dataset_adapters import swe_bench as _swe_bench_adapters  # noqa: F401


logger = logging.getLogger("swebench_pull")


@dataclass
class PullResult:
    instance_id: str
    status: str  # "pulled", "skipped", "failed", "dry_run"
    message: str = ""
    elapsed: float = 0.0
    size: int = 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. Extracted for unit testing."""
    parser = argparse.ArgumentParser(
        prog="swebench_pull",
        description="Batch pre-puller for SWE-bench SIF images.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=sorted(DATASET_ADAPTERS.keys()),
        help="Dataset adapter key (e.g. swe_bench_lite).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="HF split (default: adapter default, typically 'test').",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory to store pulled .sif files. Created if missing.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=-1,
        dest="max_rows",
        metavar="N",
        help="Stop after N rows (from the start of the dataset).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="Pull ONLY the Nth row (for SLURM array jobs).",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=None,
        dest="instance_ids",
        help="Specific instance_id to pull. Repeatable.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Parallel pulls within one process (default 1).",
    )
    parser.add_argument(
        "--min-sif-bytes",
        type=int,
        default=10_000_000,
        metavar="N",
        help="Files at or above this size are treated as complete (default 10MB).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pulled, don't invoke apptainer.",
    )
    parser.add_argument(
        "--apptainer-bin",
        default="apptainer",
        help="Apptainer binary (default 'apptainer').",
    )
    parser.add_argument(
        "--apptainer-cachedir",
        default=None,
        help="APPTAINER_CACHEDIR for subprocess (default: <cache-dir>/.apptainer-cache).",
    )
    parser.add_argument(
        "--timeout-per-pull",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="Timeout per apptainer pull in seconds (default 1800 = 30 min).",
    )
    return parser.parse_args(argv)


def _filter_by_index(rows: list[NormalizedRow], index: int) -> list[NormalizedRow]:
    if index < 0 or index >= len(rows):
        raise IndexError(
            f"--index {index} out of range for dataset of size {len(rows)}"
        )
    return [rows[index]]


def _filter_by_instance_id(
    rows: list[NormalizedRow], instance_ids: Iterable[str]
) -> list[NormalizedRow]:
    want = set(instance_ids)
    out = [r for r in rows if r.instance_id in want]
    missing = want - {r.instance_id for r in out}
    if missing:
        logger.warning("instance_ids not found in dataset: %s", sorted(missing))
    return out


def _load_rows(args: argparse.Namespace) -> list[NormalizedRow]:
    # The adapter's default load() filters out instances whose SIF doesn't
    # exist on disk — that's exactly what we're about to create here, so we
    # must bypass the filter. Set MSWE_SKIP_SIF_CHECK=1 for the duration of
    # the load; callers can still override with their own env if needed.
    os.environ.setdefault("MSWE_SKIP_SIF_CHECK", "1")
    adapter_cls = DATASET_ADAPTERS[args.dataset_name]
    adapter_kwargs = {
        "sif_cache_dir": args.cache_dir,
    }
    if args.split is not None:
        adapter_kwargs["split"] = args.split
    if args.max_rows and args.max_rows > 0:
        adapter_kwargs["max_samples"] = args.max_rows
    adapter = adapter_cls(**adapter_kwargs)
    return list(adapter.load())


@contextlib.contextmanager
def _flock(path: str):
    """Exclusive blocking flock on `path`. Lockfile is created if missing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _pull_one(
    row: NormalizedRow,
    *,
    cache_dir: str,
    min_sif_bytes: int,
    dry_run: bool,
    apptainer_bin: str,
    apptainer_cachedir: str,
    timeout_per_pull: int,
) -> PullResult:
    target = os.path.join(cache_dir, f"{row.instance_id}.sif")
    lockfile = target + ".lock"
    tmp = target + ".tmp"

    # Fast path: already present.
    if os.path.exists(target):
        size = os.path.getsize(target)
        if size >= min_sif_bytes:
            msg = f"SKIP {row.instance_id} (exists, {size} bytes)"
            logger.info(msg)
            return PullResult(row.instance_id, "skipped", msg, size=size)

    with _flock(lockfile):
        # Re-check post-lock.
        if os.path.exists(target):
            size = os.path.getsize(target)
            if size >= min_sif_bytes:
                msg = f"SKIP {row.instance_id} (exists, {size} bytes)"
                logger.info(msg)
                return PullResult(row.instance_id, "skipped", msg, size=size)

        if dry_run:
            msg = f"WOULD PULL {row.image_uri} -> {target}"
            logger.info(msg)
            return PullResult(row.instance_id, "dry_run", msg)

        # Clean up any stale .tmp from a prior failed run.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

        env = {**os.environ, "APPTAINER_CACHEDIR": apptainer_cachedir}
        start = time.monotonic()
        try:
            subprocess.run(
                [apptainer_bin, "pull", "--force", tmp, row.image_uri],
                check=True,
                timeout=timeout_per_pull,
                env=env,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Delete tmp and leave target missing.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            msg = f"FAILED {row.instance_id}: {type(e).__name__}: {e}"
            logger.error(msg)
            return PullResult(row.instance_id, "failed", msg)

        # Atomic rename.
        os.rename(tmp, target)
        elapsed = time.monotonic() - start
        size = os.path.getsize(target) if os.path.exists(target) else 0
        msg = f"PULLED {row.instance_id} ({size} bytes in {elapsed:.1f}s)"
        logger.info(msg)
        return PullResult(
            row.instance_id, "pulled", msg, elapsed=elapsed, size=size
        )


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    apptainer_cachedir = args.apptainer_cachedir or os.path.join(
        cache_dir, ".apptainer-cache"
    )
    os.makedirs(apptainer_cachedir, exist_ok=True)

    rows = _load_rows(args)
    if not rows:
        logger.info("No rows returned by adapter; nothing to do.")
        return 0

    if args.index is not None:
        rows = _filter_by_index(rows, args.index)
    elif args.instance_ids:
        rows = _filter_by_instance_id(rows, args.instance_ids)

    if not rows:
        logger.info("No rows after filtering; nothing to do.")
        return 0

    logger.info(
        "Plan: %d row(s) to consider, cache_dir=%s, concurrency=%d, dry_run=%s",
        len(rows),
        cache_dir,
        args.concurrency,
        args.dry_run,
    )

    start = time.monotonic()
    results: list[PullResult] = []

    kwargs = dict(
        cache_dir=cache_dir,
        min_sif_bytes=args.min_sif_bytes,
        dry_run=args.dry_run,
        apptainer_bin=args.apptainer_bin,
        apptainer_cachedir=apptainer_cachedir,
        timeout_per_pull=args.timeout_per_pull,
    )

    if args.concurrency <= 1:
        for row in rows:
            results.append(_pull_one(row, **kwargs))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(_pull_one, row, **kwargs): row for row in rows}
            for fut in as_completed(futures):
                results.append(fut.result())

    elapsed = time.monotonic() - start
    pulled = sum(1 for r in results if r.status == "pulled")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    dry = sum(1 for r in results if r.status == "dry_run")

    logger.info(
        "Summary: pulled=%d skipped=%d failed=%d dry_run=%d total=%d wall=%.1fs",
        pulled,
        skipped,
        failed,
        dry,
        len(results),
        elapsed,
    )

    return 0 if failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
