# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Byte-equality test for the ALFWorld prompt templates.

The port at ``recipe/alfworld/prompts.py`` is declared as a verbatim copy of
verl-agent's ``agent_system/environments/prompts/alfworld.py``. Because the
templates are tokenized before being fed to the LLM, any whitespace drift
(e.g. a stripped trailing space after ``tags.``) changes the token IDs and
invalidates apples-to-apples evaluation against verl-agent baselines.

This test imports both modules via ``importlib`` (so we compare the
Python-level string constants, not the surrounding file headers) and asserts
that the two template strings are byte-identical.
"""
from __future__ import annotations

import importlib.util
import os


_SRC_PATH = "/scratch/m000069-pm05/miaolu/verl-agent/agent_system/environments/prompts/alfworld.py"
_DST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts.py",
)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alfworld_template_byte_identical() -> None:
    """ALFWORLD_TEMPLATE must match verl-agent byte-for-byte."""
    if not os.path.exists(_SRC_PATH):
        # verl-agent source tree is not vendored into this repo; skip on
        # environments where the reference is unavailable (e.g. CI without
        # the sibling checkout).
        import pytest

        pytest.skip(f"reference source not found: {_SRC_PATH}")

    src = _load_module("verl_agent_alfworld_src", _SRC_PATH)
    dst = _load_module("verl_alfworld_dst", _DST_PATH)

    src_bytes = src.ALFWORLD_TEMPLATE.encode("utf-8")
    dst_bytes = dst.ALFWORLD_TEMPLATE.encode("utf-8")
    assert src_bytes == dst_bytes, (
        "ALFWORLD_TEMPLATE drift:\n"
        f"  verl-agent: {src_bytes!r}\n"
        f"  recipe    : {dst_bytes!r}"
    )


def test_alfworld_template_no_his_byte_identical() -> None:
    """ALFWORLD_TEMPLATE_NO_HIS must match verl-agent byte-for-byte."""
    if not os.path.exists(_SRC_PATH):
        import pytest

        pytest.skip(f"reference source not found: {_SRC_PATH}")

    src = _load_module("verl_agent_alfworld_src", _SRC_PATH)
    dst = _load_module("verl_alfworld_dst", _DST_PATH)

    src_bytes = src.ALFWORLD_TEMPLATE_NO_HIS.encode("utf-8")
    dst_bytes = dst.ALFWORLD_TEMPLATE_NO_HIS.encode("utf-8")
    assert src_bytes == dst_bytes, (
        "ALFWORLD_TEMPLATE_NO_HIS drift:\n"
        f"  verl-agent: {src_bytes!r}\n"
        f"  recipe    : {dst_bytes!r}"
    )
