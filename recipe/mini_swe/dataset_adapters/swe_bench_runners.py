import re
import shlex
from dataclasses import dataclass, field
from typing import Callable


def _shell_quote(s: str) -> str:
    """Always wrap `s` in single quotes, escaping any embedded single quotes.

    Unlike :func:`shlex.quote`, which omits quotes when the argument contains
    only "safe" characters, this always emits quotes so the on-wire command
    is unambiguous and matches what downstream tooling expects to see logged.
    """
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---- Django test-ID normalizer ----
#
# SWE-bench stores Django test IDs in unittest repr form:
#   "test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)"
# but Django's ``runtests.py`` positional-label parser requires the dotted form:
#   "auth_tests.test_validators.UsernameValidatorsTests.test_ascii_validator"
# ``normalize_unittest_repr`` converts the former to the latter, and leaves any
# input that already looks dotted untouched. Applied via ``normalize_test_ids``
# flag on :class:`TestRunnerSpec` (see ``build_command``).

_UNITTEST_REPR_RE = re.compile(r"^(\w+)\s+\(([\w\.]+)\)\s*$")


def normalize_unittest_repr(test_id: str) -> str:
    """Convert unittest repr form ``method (module.Class)`` to dotted form
    ``module.Class.method``. Leave dotted ids unchanged.
    """
    m = _UNITTEST_REPR_RE.match(test_id)
    return f"{m.group(2)}.{m.group(1)}" if m else test_id


def normalize_test_ids(test_ids: list[str]) -> list[str]:
    return [normalize_unittest_repr(t) for t in test_ids]


# ---- outcome parsers (each returns (passed_count, total_count)) ----


def parse_django_runtests(output: str, test_ids: list[str]) -> tuple[int, int]:
    """Django's runtests.py prints 'test_foo (module.Class) ... ok' per test
    and 'Ran N tests in X.Ys' + 'OK' / 'FAILED' summary."""
    # Count lines ending with " ... ok" (passed) and " ... FAIL" / " ... ERROR"
    total = len(test_ids) if test_ids else 0
    ok = len(re.findall(r"\.\.\. ok$", output, re.MULTILINE))
    fail = len(re.findall(r"\.\.\. (FAIL|ERROR)", output, re.MULTILINE))
    # If total known, trust that; otherwise use ok+fail
    if total == 0:
        total = ok + fail
    # passed = ok (failures count against the total)
    return ok, total


def parse_pytest(output: str, test_ids: list[str]) -> tuple[int, int]:
    """Parse pytest short summary: '=== N passed, M failed ... ===' or similar."""
    m = re.search(r"(\d+) passed", output)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", output)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", output)
    errored = int(m.group(1)) if m else 0
    total = passed + failed + errored
    if total == 0 and test_ids:
        total = len(test_ids)
    return passed, total


def parse_sympy_bintest(output: str, test_ids: list[str]) -> tuple[int, int]:
    """sympy bin/test prints 'tests finished: N passed, M failed, ...'"""
    m = re.search(r"tests finished:\s*(\d+)\s+passed", output)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s+failed", output)
    failed = int(m.group(1)) if m else 0
    return passed, passed + failed


OUTCOME_PARSERS: dict[str, Callable[[str, list[str]], tuple[int, int]]] = {
    "django_runtests": parse_django_runtests,
    "pytest": parse_pytest,
    "sympy_bintest": parse_sympy_bintest,
}


# ---- TestRunnerSpec ----


@dataclass
class TestRunnerSpec:
    shell_cmd_template: str
    test_id_separator: str = " "
    pre_cmd: str = ""
    outcome_parser: str = "pytest"
    # When True, ``build_command`` rewrites unittest-repr test ids
    # ("method (mod.Class)") to dotted form ("mod.Class.method"). Django's
    # runtests.py positional-label parser requires the dotted form; SWE-bench
    # stores the repr form. Other runners (pytest) leave ids alone.
    normalize_test_ids: bool = False

    def build_command(self, test_ids: list[str]) -> str:
        ids = normalize_test_ids(test_ids) if self.normalize_test_ids else test_ids
        joined = self.test_id_separator.join(_shell_quote(t) for t in ids)
        pre = f"{self.pre_cmd} && " if self.pre_cmd else ""
        return pre + self.shell_cmd_template.format(tests=joined)

    def parse_outcome(self, output: str, test_ids: list[str]) -> tuple[int, int]:
        fn = OUTCOME_PARSERS[self.outcome_parser]
        return fn(output, test_ids)


# ---- RUNNER_MAP: (repo, version) -> spec ----
# Start with Django; other repos added in M6.
# django uses runtests.py with --settings=test_sqlite and NO --parallel 1 issues

RUNNER_MAP: dict[str, dict[str, TestRunnerSpec]] = {
    "django/django": {
        v: TestRunnerSpec(
            shell_cmd_template="./tests/runtests.py --verbosity 2 --settings=test_sqlite {tests}",
            outcome_parser="django_runtests",
            normalize_test_ids=True,
        )
        for v in ("2.2", "3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0")
    },
    # TODO(M6): sympy/sympy, astropy/astropy, scikit-learn, etc.
}


def get_runner(repo: str, version: str) -> TestRunnerSpec | None:
    return RUNNER_MAP.get(repo, {}).get(str(version))
