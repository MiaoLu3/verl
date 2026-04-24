# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Trajectory-dumper smoke for AlfWorldAgentLoop.

Drives the same mock episode as test_alfworld_agent_loop_mock.py but with
``ALFWORLD_TRAJ_DUMP_DIR`` set to a tmp_path. Asserts that:

  * Exactly one JSONL file was written (named <request_id>.jsonl).
  * The record has all expected top-level keys.
  * ``turns`` has length >= 2 (mock runs >= 2 env steps via max_assistant_turns=3).
  * Per-turn record fields are present and typed sensibly.
  * ``response_mask_1_count + response_mask_0_count == response_length_final``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from recipe.alfworld.alfworld_agent_loop import AlfWorldAgentLoop
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.workers.rollout.replica import TokenOutput


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_DIR, "config_tw.yaml")
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
HF_CACHE = "/scratch/m000069/miaolu/.cache/huggingface"


class _FakeServerManager:
    def __init__(self, canned_ids: list[int]):
        self._canned_ids = list(canned_ids)
        self.calls: list[dict] = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_len": len(prompt_ids),
                "sampling_params": sampling_params,
            }
        )
        return TokenOutput(
            token_ids=list(self._canned_ids),
            log_probs=None,
            num_preempted=0,
        )


@pytest.fixture(scope="module")
def _tokenizer():
    if os.path.isdir(HF_CACHE):
        os.environ.setdefault("HF_HOME", HF_CACHE)
        os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def _ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=8, local_mode=False)
    yield


def _build_agent_loop(tokenizer, fake_server):
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 4096,
                    "response_length": 2048,
                    "multi_turn": {
                        "enable": True,
                        "max_assistant_turns": 3,
                        "max_user_turns": 10,
                        "tool_config_path": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 256,
                        "tool_response_truncate_side": "middle",
                        "interaction_config_path": None,
                        "format": "hermes",
                    },
                    "alfworld": {
                        "alf_config_path": CONFIG_PATH,
                        "pool_size": 2,
                        "history_length": 2,
                        "is_train": True,
                        "seed_base": 54321,
                        "max_steps": 5,
                    },
                },
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    return AlfWorldAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=fake_server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,
        data_config=DictConfigWrap(cfg.data),
    )


@pytest.mark.skipif(
    not os.environ.get("ALFWORLD_DATA"),
    reason="ALFWORLD_DATA env not set",
)
def test_alfworld_trajectory_dump(_tokenizer, _ray_init, tmp_path, monkeypatch):
    dump_dir = tmp_path / "alfworld_traj"
    monkeypatch.setenv("ALFWORLD_TRAJ_DUMP_DIR", str(dump_dir))

    canned_text = "<think>look around me for the next step</think><action>look</action>"
    canned_ids = _tokenizer(canned_text, add_special_tokens=False)["input_ids"]

    fake_server = _FakeServerManager(canned_ids=canned_ids)
    loop = _build_agent_loop(_tokenizer, fake_server)

    # Dump dir is read at __init__ from env var.
    assert loop.dump_dir == str(dump_dir)
    assert dump_dir.is_dir()

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "logprobs": False,
    }

    async def _run():
        return await loop.run(sampling_params)

    output = asyncio.get_event_loop().run_until_complete(_run())

    # --- Exactly one JSONL per episode ---------------------------------
    jsonl_files = sorted(dump_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, [p.name for p in jsonl_files]
    dump_path = jsonl_files[0]
    assert dump_path.stem.isalnum(), dump_path.stem  # uuid4().hex

    with open(dump_path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    record = json.loads(lines[0])

    # --- Top-level keys present ----------------------------------------
    expected_keys = {
        "request_id",
        "gamefile",
        "won",
        "final_reward",
        "num_turns",
        "num_invalid_actions",
        "turns",
        "prompt_length_final",
        "response_length_final",
        "response_mask_1_count",
        "response_mask_0_count",
    }
    assert expected_keys.issubset(record.keys()), record.keys()

    # --- Turns list shape ----------------------------------------------
    turns = record["turns"]
    assert isinstance(turns, list), type(turns)
    assert len(turns) >= 2, len(turns)
    per_turn_keys = {
        "turn_idx",
        "observation",
        "admissible",
        "raw_response",
        "parsed_action",
        "valid",
        "reward",
        "done",
    }
    for t in turns:
        assert per_turn_keys.issubset(t.keys()), t.keys()
        assert isinstance(t["admissible"], list)
        assert isinstance(t["raw_response"], str)
        assert isinstance(t["valid"], bool)
        assert isinstance(t["reward"], float)
        assert isinstance(t["done"], bool)
    # Canned raw response should have been captured.
    assert "<action>look</action>" in turns[0]["raw_response"], turns[0]["raw_response"][:200]

    # --- Mask accounting matches response length ------------------------
    assert (
        record["response_mask_1_count"] + record["response_mask_0_count"]
        == record["response_length_final"]
    ), record
    # Sanity: both halves are populated in the mock run.
    assert record["response_mask_1_count"] > 0
    assert record["response_mask_0_count"] > 0

    # --- IDs line up with the actual output -----------------------------
    assert record["request_id"] == dump_path.stem
    assert record["num_turns"] == len(turns)
    assert record["num_invalid_actions"] == output.extra_fields["num_invalid_actions"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
