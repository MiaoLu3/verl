import os
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class NormalizedRow:
    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str
    image_uri: str  # "docker://..."
    sif_path: str  # "$SIF_CACHE_DIR/<instance_id>.sif"
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    install_spec: str = ""  # shell snippet, may be empty
    test_runner: Any = None  # TestRunnerSpec instance (see swe_bench_runners.py)
    raw: dict | None = None  # original HF row for debugging


class BaseAdapter:
    name: ClassVar[str] = ""
    hf_name: ClassVar[str] = ""
    default_split: ClassVar[str] = "test"

    # DeepSWE's contamination list - repos in SWE-bench-Verified
    VERIFIED_REPOS: ClassVar[frozenset[str]] = frozenset(
        {
            "astropy/astropy",
            "django/django",
            "pallets/flask",
            "matplotlib/matplotlib",
            "pylint-dev/pylint",
            "pytest-dev/pytest",
            "psf/requests",
            "scikit-learn/scikit-learn",
            "mwaskom/seaborn",
            "sphinx-doc/sphinx",
            "sympy/sympy",
            "pydata/xarray",
        }
    )

    def __init__(
        self,
        *,
        split: str | None = None,
        instance_ids: list[str] | None = None,
        max_samples: int = -1,
        sif_cache_dir: str | None = None,
        filter_verified_repos: bool = False,
        **_unused,
    ):
        self.split = split or self.default_split
        self.instance_ids = instance_ids
        self.max_samples = max_samples
        self.sif_cache_dir = sif_cache_dir or os.environ.get("SIF_CACHE_DIR", "")
        self.filter_verified_repos = filter_verified_repos

    def load(self) -> list[NormalizedRow]:
        raise NotImplementedError


DATASET_ADAPTERS: dict[str, type[BaseAdapter]] = {}


def register_adapter(cls: type[BaseAdapter]) -> type[BaseAdapter]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} missing `name` class attribute")
    DATASET_ADAPTERS[cls.name] = cls
    return cls
