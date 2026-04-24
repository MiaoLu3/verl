# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Table-driven tests for alfworld_projection.

Parity note (fixes #1-#3, #5):
  The projection used to enforce admissibility, an ASCII-only guard, and a
  strict ``end > start`` tag-order check, and fell back to
  ``admissible_commands[0]`` on any failure. verl-agent's reference
  implementation does none of those things -- it only checks for CJK and for
  the presence of ``<think>`` and ``<action>`` tags, and its fallback on any
  failure is ``raw_response.lower()[-30:]`` (the tail of the raw response).

  We match verl-agent's behaviour exactly so training rewards line up. The
  test cases below were rewritten against the new (looser) contract.
"""
from __future__ import annotations

import sys

from recipe.alfworld.alfworld_projection import alfworld_projection


def _fallback(raw: str) -> str:
    """Replicate the module's fallback to keep expectations explicit."""
    if not raw:
        return ""
    return raw.lower()[-30:]


def _run_cases() -> tuple[int, int, list[str]]:
    """Run all cases. Returns (passed, total, failure_messages)."""
    cases = [
        # (name, raw_response, admissible, expected_action, expected_valid)
        (
            "valid_basic",
            "<think>look around</think><action>look</action>",
            ["look", "go north"],
            "look",
            True,
        ),
        (
            # With the <think> check disabled (to support no-think mode where
            # the chat template prefills <think></think> into the prompt and
            # raw_response starts directly with the model's answer), a bare
            # <action>X</action> is accepted as valid.
            "missing_think_still_valid",
            "<action>look</action>",
            ["look", "go north"],
            "look",
            True,
        ),
        (
            "missing_action",
            "<think>look</think>",
            ["look", "go north"],
            _fallback("<think>look</think>"),
            False,
        ),
        (
            "non_ascii_cjk",
            "<think>看</think><action>看看</action>",
            ["look"],
            _fallback("<think>看</think><action>看看</action>"),
            False,
        ),
        (
            # verl-agent does NOT check admissibility. The extracted action
            # is returned as-is with valid=True even if it isn't in the pool.
            "action_not_in_admissible_still_valid",
            "<think>x</think><action>fly</action>",
            ["look"],
            "fly",
            True,
        ),
        (
            "empty_response",
            "",
            ["look", "go north"],
            "",
            False,
        ),
        # Extra cases to lock down behaviour:
        (
            "valid_with_trailing_whitespace",
            "<think>plan</think><action>  go north  </action>",
            ["look", "go north"],
            "go north",
            True,
        ),
        (
            "case_insensitive_extraction",
            "<think>plan</think><action>LOOK</action>",
            ["look"],
            "look",
            True,
        ),
        (
            # Empty admissible list must NOT cause a failure -- projection
            # ignores the admissible list entirely for validation.
            "empty_admissible_list_with_valid_tags",
            "<think>x</think><action>look</action>",
            [],
            "look",
            True,
        ),
        (
            # Non-CJK non-ASCII (emoji / accents) must NOT be rejected:
            # verl-agent only filters CJK ideographs.
            "emoji_allowed",
            "<think>hmm</think><action>look 🙂</action>",
            ["look"],
            "look 🙂",
            True,
        ),
        (
            # Latin-accented chars must NOT be rejected either.
            "accented_latin_allowed",
            "<think>réfléchir</think><action>look</action>",
            ["look"],
            "look",
            True,
        ),
    ]

    passed = 0
    failures: list[str] = []
    for name, raw, adm, exp_action, exp_valid in cases:
        got_action, got_valid = alfworld_projection(raw, adm)
        ok = (got_action == exp_action) and (got_valid == exp_valid)
        if ok:
            passed += 1
            print(f"[PASS] {name}")
        else:
            failures.append(
                f"[FAIL] {name}: got=({got_action!r}, {got_valid}), "
                f"exp=({exp_action!r}, {exp_valid})"
            )
            print(failures[-1])
    return passed, len(cases), failures


def test_alfworld_projection_table() -> None:
    """pytest entry point: all table-driven cases must pass."""
    passed, total, failures = _run_cases()
    assert not failures, f"{passed}/{total} passed; failures:\n" + "\n".join(failures)


if __name__ == "__main__":
    passed, total, failures = _run_cases()
    print(f"\n{passed}/{total} cases passed.")
    if failures:
        sys.exit(1)
