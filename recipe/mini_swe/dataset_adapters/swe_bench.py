import json
import os
from typing import Any

from . import BaseAdapter, NormalizedRow, register_adapter
from .swe_bench_runners import get_runner


@register_adapter
class SweBenchAdapter(BaseAdapter):
    name = "swe_bench_lite"
    hf_name = "princeton-nlp/SWE-bench_Lite"
    default_split = "test"

    def load(self) -> list[NormalizedRow]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("`datasets` package required: pip install datasets") from e
        ds = load_dataset(self.hf_name, split=self.split)
        if self.instance_ids:
            want = set(self.instance_ids)
            ds = ds.filter(lambda r: r["instance_id"] in want)
        if self.max_samples > 0:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        if self.filter_verified_repos:
            ds = ds.filter(lambda r: r["repo"] not in self.VERIFIED_REPOS)
        rows = [self._normalize(r) for r in ds]
        # Drop rows whose SIF hasn't been pulled yet. Opt-out via
        # MSWE_SKIP_SIF_CHECK=1 for flows where SIFs will be pulled lazily.
        if self.sif_cache_dir and os.environ.get("MSWE_SKIP_SIF_CHECK", "0") != "1":
            before = len(rows)
            rows = [r for r in rows if os.path.isfile(r.sif_path)]
            dropped = before - len(rows)
            if dropped:
                import logging
                logging.getLogger(__name__).warning(
                    "SweBenchAdapter dropped %d/%d instances with no SIF in %s "
                    "(set MSWE_SKIP_SIF_CHECK=1 to bypass).",
                    dropped, before, self.sif_cache_dir,
                )
        return rows

    def _normalize(self, r: dict) -> NormalizedRow:
        repo = r["repo"]
        version = r.get("version", "")
        runner = get_runner(repo, version)
        # Note: runner may be None for repos we haven't added yet - we DO NOT
        # raise here (that would block dataset loading); we just leave test_runner=None
        # and let MiniSweAgentLoop / reward surface a clear error at rollout time.
        return NormalizedRow(
            instance_id=r["instance_id"],
            repo=repo,
            problem_statement=r["problem_statement"],
            base_commit=r["base_commit"],
            image_uri=self._image_uri(r),
            sif_path=os.path.join(self.sif_cache_dir, f"{r['instance_id']}.sif"),
            fail_to_pass=self._list_or_json(r["FAIL_TO_PASS"]),
            pass_to_pass=self._list_or_json(r["PASS_TO_PASS"]),
            install_spec="",
            test_runner=runner,
            raw=dict(r),
        )

    def _image_uri(self, r: dict) -> str:
        iid = r["instance_id"].replace("__", "_1776_")
        return f"docker://swebench/sweb.eval.x86_64.{iid}:latest"

    @staticmethod
    def _list_or_json(v: Any) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        if v is None:
            return []
        return list(v)


@register_adapter
class SweBenchVerifiedAdapter(SweBenchAdapter):
    name = "swe_bench_verified"
    hf_name = "princeton-nlp/SWE-bench_Verified"


@register_adapter
class SweBenchFullAdapter(SweBenchAdapter):
    name = "swe_bench_full"
    hf_name = "princeton-nlp/SWE-bench"
