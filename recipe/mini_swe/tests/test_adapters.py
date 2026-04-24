"""Unit tests for M5a dataset adapter layer.

These tests do NOT touch the HF network; they use hand-crafted rows
matching the SWE-bench schema.
"""

import json

import pytest

from recipe.mini_swe.dataset_adapters import DATASET_ADAPTERS, BaseAdapter, NormalizedRow
from recipe.mini_swe.dataset_adapters.swe_bench import (
    SweBenchAdapter,
    SweBenchFullAdapter,
    SweBenchVerifiedAdapter,
)
from recipe.mini_swe.dataset_adapters.swe_bench_runners import (
    RUNNER_MAP,
    TestRunnerSpec,
    get_runner,
    normalize_test_ids,
    normalize_unittest_repr,
    parse_django_runtests,
    parse_pytest,
)


SAMPLE_DJANGO_ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "version": "3.0",
    "problem_statement": "ASCIIUsernameValidator allows trailing newline...",
    "base_commit": "d26b2424437dabeeca94d7900b37d2df4410da0c",
    "patch": "...",
    "test_patch": "...",
    "FAIL_TO_PASS": '["auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator"]',
    "PASS_TO_PASS": "[]",
    "environment_setup_commit": "...",
    "hints_text": "",
    "created_at": "...",
}


def test_registry_populated():
    # Adapters register on import of swe_bench module (already imported above).
    assert "swe_bench_lite" in DATASET_ADAPTERS
    assert "swe_bench_verified" in DATASET_ADAPTERS
    assert "swe_bench_full" in DATASET_ADAPTERS
    assert DATASET_ADAPTERS["swe_bench_lite"] is SweBenchAdapter
    assert DATASET_ADAPTERS["swe_bench_verified"] is SweBenchVerifiedAdapter
    assert DATASET_ADAPTERS["swe_bench_full"] is SweBenchFullAdapter


def test_aliases_have_distinct_hf_names():
    names = {
        SweBenchAdapter.hf_name,
        SweBenchVerifiedAdapter.hf_name,
        SweBenchFullAdapter.hf_name,
    }
    assert len(names) == 3
    assert SweBenchAdapter.hf_name == "princeton-nlp/SWE-bench_Lite"
    assert SweBenchVerifiedAdapter.hf_name == "princeton-nlp/SWE-bench_Verified"
    assert SweBenchFullAdapter.hf_name == "princeton-nlp/SWE-bench"


def test_normalize_happy_path():
    adapter = SweBenchAdapter(sif_cache_dir="/tmp/sif_cache")
    row = adapter._normalize(SAMPLE_DJANGO_ROW)
    assert isinstance(row, NormalizedRow)
    assert row.instance_id == "django__django-11099"
    assert row.repo == "django/django"
    assert row.problem_statement == SAMPLE_DJANGO_ROW["problem_statement"]
    assert row.base_commit == "d26b2424437dabeeca94d7900b37d2df4410da0c"
    assert (
        row.image_uri
        == "docker://swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    )
    assert row.sif_path == "/tmp/sif_cache/django__django-11099.sif"
    assert row.fail_to_pass == [
        "auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator"
    ]
    assert row.pass_to_pass == []
    assert row.install_spec == ""
    assert row.test_runner is not None
    # build_command must shell-quote ids with spaces
    cmd = row.test_runner.build_command(["a::b", "c d"])
    assert "a::b" in cmd
    assert "'c d'" in cmd
    assert row.raw is not None
    assert row.raw["instance_id"] == "django__django-11099"

    # Parse a sample django runtests output
    sample_output = (
        "test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests) ... ok\n"
        "test_unicode_validator (auth_tests.test_validators.UsernameValidatorsTests) ... ok\n"
        "test_broken (auth_tests.test_validators.UsernameValidatorsTests) ... FAIL\n"
        "Ran 3 tests in 0.123s\n"
    )
    passed, total = parse_django_runtests(sample_output, ["t1", "t2", "t3"])
    assert passed == 2
    assert total == 3


def test_normalize_unknown_repo():
    adapter = SweBenchAdapter(sif_cache_dir="/tmp/sif_cache")
    unknown_row = {
        **SAMPLE_DJANGO_ROW,
        "instance_id": "someorg__somerepo-42",
        "repo": "someorg/somerepo",
        "version": "99.99",
    }
    # Must not raise
    row = adapter._normalize(unknown_row)
    assert row.test_runner is None
    assert row.instance_id == "someorg__somerepo-42"
    assert row.repo == "someorg/somerepo"


def test_list_or_json():
    # JSON string
    assert SweBenchAdapter._list_or_json('["a", "b"]') == ["a", "b"]
    # Already a list
    assert SweBenchAdapter._list_or_json(["x", "y"]) == ["x", "y"]
    # None
    assert SweBenchAdapter._list_or_json(None) == []
    # Empty JSON array string
    assert SweBenchAdapter._list_or_json("[]") == []


def test_runner_map_coverage():
    assert "django/django" in RUNNER_MAP
    assert "3.0" in RUNNER_MAP["django/django"]
    spec = get_runner("django/django", "3.0")
    assert spec is not None
    assert isinstance(spec, TestRunnerSpec)
    # Unknown repo -> None
    assert get_runner("no/such", "1.0") is None
    # Unknown version -> None
    assert get_runner("django/django", "99.99") is None


def test_build_command():
    spec = TestRunnerSpec("pytest -rA {tests}", outcome_parser="pytest")
    cmd = spec.build_command(
        ["tests/test_a.py::test_1", "tests/test b.py::test_2"]
    )
    assert (
        cmd
        == "pytest -rA 'tests/test_a.py::test_1' 'tests/test b.py::test_2'"
    )


def test_parse_pytest():
    output = "================ 3 passed, 1 failed in 0.4s ================"
    passed, total = parse_pytest(output, [])
    assert passed == 3
    assert total == 4


# ---------------------------------------------------------------------------
# Django unittest-repr -> dotted-form normalizer
# ---------------------------------------------------------------------------


def test_normalize_unittest_repr_repr_form():
    s = "test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)"
    assert (
        normalize_unittest_repr(s)
        == "auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator"
    )


def test_normalize_unittest_repr_already_dotted_untouched():
    s = "auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator"
    assert normalize_unittest_repr(s) == s


def test_normalize_unittest_repr_pytest_style_untouched():
    # pytest nodeids don't match the repr pattern -> unchanged
    s = "tests/test_a.py::TestClass::test_method"
    assert normalize_unittest_repr(s) == s


def test_normalize_test_ids_batch():
    ids = [
        "test_foo (mod.pkg.Class)",
        "mod.pkg.Class.test_bar",
    ]
    assert normalize_test_ids(ids) == [
        "mod.pkg.Class.test_foo",
        "mod.pkg.Class.test_bar",
    ]


def test_django_runner_emits_dotted_ids():
    spec = get_runner("django/django", "3.0")
    assert spec is not None
    assert spec.normalize_test_ids is True
    cmd = spec.build_command(
        ["test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)"]
    )
    # The command must contain the dotted form, NOT the repr form.
    assert (
        "'auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator'"
        in cmd
    )
    # Literal repr form with parentheses must NOT appear.
    assert "(auth_tests.test_validators.UsernameValidatorsTests)" not in cmd


def test_pytest_spec_does_not_normalize():
    spec = TestRunnerSpec("pytest {tests}", outcome_parser="pytest")
    cmd = spec.build_command(["test_foo (mod.Cls)"])
    # Pytest spec leaves the string as-is
    assert "test_foo (mod.Cls)" in cmd
