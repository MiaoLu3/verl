# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end mock test for AlfWorldAgentLoop.

Drives a full PENDING -> GENERATING -> INTERACTING -> ... -> TERMINATED run
with:
  * A real Ray-actor env pool (pool_size=2, train split).
  * A mocked server_manager.generate that always returns the same canned
    '<think>...</think><action>look</action>' response, tokenized with the
    same Qwen2.5-1.5B-Instruct tokenizer used by the loop.
  * max_assistant_turns=3 so the loop terminates quickly even if the env
    never says `done`.

Checks the token-in-token-out bookkeeping contract:
  * len(response_ids) == len(response_mask)
  * response_mask has both 1s (assistant) and 0s (env observations)
  * decoding the mask==1 positions recovers the assistant text without env
    bleed-through.
"""
from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# Fake server manager
# ---------------------------------------------------------------------------


class _FakeServerManager:
    """Mock AsyncLLMServerManager that always returns a canned assistant turn."""

    def __init__(self, canned_ids: list[int]):
        self._canned_ids = list(canned_ids)
        self.calls: list[dict] = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs) -> TokenOutput:
        # Mimic tiny vLLM latency so other coros can interleave.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    # Don't shut ray down; other tests in the same session may reuse it.


def _build_agent_loop(tokenizer, fake_server):
    """Construct an AlfWorldAgentLoop with a minimal OmegaConf config."""
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
    loop = AlfWorldAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=fake_server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=object,  # unused for this path
        data_config=DictConfigWrap(cfg.data),
    )
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ALFWORLD_DATA"),
    reason="ALFWORLD_DATA env not set",
)
def test_alfworld_agent_loop_mock(_tokenizer, _ray_init):
    # Canned assistant response: a valid projection that picks the first
    # admissible action the env offers after reset (it's always 'look'
    # in the ALFRED TextWorld initial state).
    canned_text = "<think>look around me for the next step</think><action>look</action>"
    canned_ids = _tokenizer(canned_text, add_special_tokens=False)["input_ids"]

    fake_server = _FakeServerManager(canned_ids=canned_ids)
    loop = _build_agent_loop(_tokenizer, fake_server)

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

    # --- Core bookkeeping invariants ------------------------------------
    assert len(output.response_ids) == len(output.response_mask), (
        len(output.response_ids),
        len(output.response_mask),
    )
    assert len(output.response_mask) > 0, "empty response_mask"
    assert sum(output.response_mask) > 0, "no assistant tokens"
    n_env = sum(1 for m in output.response_mask if m == 0)
    assert n_env > 0, "no env-observation tokens spliced in"

    # --- Server was called multiple times with a growing prompt --------
    assert len(fake_server.calls) >= 2, fake_server.calls
    # Prefix-cache sticky session: same request_id every turn.
    req_ids = {c["request_id"] for c in fake_server.calls}
    assert len(req_ids) == 1, req_ids
    prompt_lens = [c["prompt_len"] for c in fake_server.calls]
    assert prompt_lens == sorted(prompt_lens), prompt_lens
    assert prompt_lens[-1] > prompt_lens[0], prompt_lens

    # --- Reward score / extras ------------------------------------------
    assert output.reward_score in (0.0, 1.0), output.reward_score
    assert output.num_turns >= 2, output.num_turns
    assert "gamefile" in output.extra_fields
    assert output.extra_fields["gamefile"], output.extra_fields
    assert "turn_scores" in output.extra_fields

    # --- Decoded assistant slice has no env bleed ------------------------
    # Extract just the assistant tokens and decode; we should see our
    # canned <think>/<action> text, never the "You are an expert agent"
    # template header.
    mask = output.response_mask
    resp = output.response_ids
    asst_ids = [t for t, m in zip(resp, mask) if m == 1]
    decoded = _tokenizer.decode(asst_ids, skip_special_tokens=True)
    assert "<action>look</action>" in decoded, decoded[:200]
    assert "expert agent" not in decoded, decoded[:200]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
