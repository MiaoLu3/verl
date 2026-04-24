# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke test for AlfWorldDataset: enumerates gamefiles and yields stub rows."""
from __future__ import annotations

import os

import pytest

from recipe.alfworld.alfworld_dataset import AlfWorldDataset


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config_tw.yaml",
)


@pytest.mark.parametrize("split,expected_min", [
    ("valid_seen", 100),
    ("valid_unseen", 100),
    ("train", 1000),
])
def test_dataset_split_enumerates(split, expected_min):
    if not os.environ.get("ALFWORLD_DATA"):
        pytest.skip("ALFWORLD_DATA not set; cannot resolve gamefile paths")

    ds = AlfWorldDataset(config={}, split=split, alf_config_path=CONFIG_PATH)
    info = ds.describe()
    assert info["split"] == split
    assert info["num_gamefiles"] >= expected_min, info

    row0 = ds[0]
    # Core shape.
    assert "raw_prompt" in row0
    assert "agent_name" in row0 and row0["agent_name"] == "alfworld"
    assert "extra_info" in row0
    assert "index" in row0 and row0["index"] == 0

    # Gamefile is absolute and exists on disk.
    gf = row0["extra_info"]["gamefile"]
    assert os.path.isabs(gf), gf
    assert os.path.exists(gf), gf
    assert gf.endswith(".tw-pddl"), gf

    # raw_prompt is a list of dicts (role/content).
    import numpy as np
    assert isinstance(row0["raw_prompt"], np.ndarray)
    assert len(row0["raw_prompt"]) >= 1
    assert row0["raw_prompt"][0]["role"] == "user"


def test_valid_seen_count_140():
    """Sanity: valid_seen should have the full 140 games."""
    if not os.environ.get("ALFWORLD_DATA"):
        pytest.skip("ALFWORLD_DATA not set")
    ds = AlfWorldDataset(config={}, split="valid_seen", alf_config_path=CONFIG_PATH)
    assert len(ds) == 140, len(ds)


def test_max_samples_cap():
    if not os.environ.get("ALFWORLD_DATA"):
        pytest.skip("ALFWORLD_DATA not set")
    ds = AlfWorldDataset(
        config={}, split="valid_seen",
        alf_config_path=CONFIG_PATH, max_samples=7,
    )
    assert len(ds) == 7
