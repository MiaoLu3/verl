"""Unit tests for the M4 :class:`MiniSweAgentLoop`.

Four layers:

1. **Pure-regex parsers** (``_parse_actions`` / ``_parse_final``).
2. **Template rendering** (``_render_initial_messages``).
3. **Finalization bookkeeping** (``_split_prompt_response`` / ``_truncate``).
4. **Mock state-machine** — drives a synthetic rollout through the real
   :meth:`MiniSweAgentLoop.run` body by short-circuiting ``AgentLoopBase.__init__``
   and mocking ``server_manager.generate``, ``apply_chat_template``,
   ``tokenizer``, the apptainer env and :func:`score_patch`. Skipped if verl
   (and hence ray) isn't installed — the pure-unit tests above still run.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest import mock

import pytest

from recipe.mini_swe.agent_loop import (
    BASH_BEGIN,
    BASH_END,
    FINAL_MARKER,
    MAX_COMMANDS_PER_TURN,
    NUDGE_MISSING_BASH,
    _VERL_AVAILABLE,
    _format_observation,
    _load_prompt_config,
    _parse_actions,
    _parse_final,
    _render_initial_messages,
    _split_prompt_response,
    _truncate,
)


# ---------------------------------------------------------------------------
# 1. Pure-regex parsers
# ---------------------------------------------------------------------------


def test_parse_actions_single():
    text = f"preamble\n{BASH_BEGIN}\nls -la\n{BASH_END}\ntrailing"
    assert _parse_actions(text) == ["ls -la"]


def test_parse_actions_multi():
    text = (
        f"{BASH_BEGIN}\ncmd1\n{BASH_END}\n"
        f"thought\n{BASH_BEGIN}\ncmd2 with space\n{BASH_END}"
    )
    assert _parse_actions(text) == ["cmd1", "cmd2 with space"]


def test_parse_actions_none():
    assert _parse_actions("no delimiters anywhere") == []


def test_parse_actions_empty_block_skipped():
    text = f"{BASH_BEGIN}\n   \n{BASH_END}\n{BASH_BEGIN}\nreal\n{BASH_END}"
    assert _parse_actions(text) == ["real"]


def test_parse_actions_cap_at_max():
    blocks = "".join(
        f"{BASH_BEGIN}\ncmd{i}\n{BASH_END}\n"
        for i in range(MAX_COMMANDS_PER_TURN + 3)
    )
    cmds = _parse_actions(blocks)
    assert len(cmds) == MAX_COMMANDS_PER_TURN
    assert cmds[0] == "cmd0"
    assert cmds[-1] == f"cmd{MAX_COMMANDS_PER_TURN - 1}"


def test_parse_actions_multiline_command():
    text = f"{BASH_BEGIN}\nfor i in 1 2 3; do\n  echo $i\ndone\n{BASH_END}"
    cmds = _parse_actions(text)
    assert len(cmds) == 1
    assert "for i in 1 2 3; do" in cmds[0]
    assert "echo $i" in cmds[0]


def test_parse_final_present():
    text = f"reasoning\n{FINAL_MARKER}\ndiff --git a/b b/c\n+added line\n"
    patch = _parse_final(text)
    assert patch is not None
    assert patch.startswith("diff --git")
    assert "+added line" in patch


def test_parse_final_absent():
    assert _parse_final("no marker here") is None


def test_parse_final_trailing_only():
    text = f"{FINAL_MARKER}\n\nquick patch\n\n"
    assert _parse_final(text) == "quick patch"


# ---------------------------------------------------------------------------
# 2. Template rendering
# ---------------------------------------------------------------------------


def test_load_prompt_config_default():
    cfg = _load_prompt_config()
    assert "agent" in cfg
    assert "system_template" in cfg["agent"]
    assert "instance_template" in cfg["agent"]


def test_load_prompt_config_missing_agent_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("other: stuff\n")
    with pytest.raises(ValueError, match="agent"):
        _load_prompt_config(str(bad))


def test_load_prompt_config_missing_subkey(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("agent:\n  system_template: 's'\n")
    with pytest.raises(ValueError, match="instance_template"):
        _load_prompt_config(str(bad))


def test_render_initial_messages_basic():
    cfg = _load_prompt_config()
    messages = _render_initial_messages(
        cfg, task="Fix the bug in foo().", cwd="/testbed", env_vars={"FOO": "bar"}
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # Delimiters must appear in the system prompt so the model knows the
    # bash / final-output protocol.
    assert BASH_BEGIN in messages[0]["content"]
    assert BASH_END in messages[0]["content"]
    assert FINAL_MARKER in messages[0]["content"]
    # Task + cwd injected into the instance prompt.
    assert "Fix the bug in foo()." in messages[1]["content"]
    assert "/testbed" in messages[1]["content"]


def test_render_initial_messages_strict_undefined():
    """StrictUndefined: rendering with a template that references an
    unknown variable must raise, not silently emit empty string."""
    cfg = {
        "agent": {
            "system_template": "sys",
            "instance_template": "{{ not_a_real_variable }}",
        }
    }
    from jinja2.exceptions import UndefinedError

    with pytest.raises(UndefinedError):
        _render_initial_messages(cfg, task="t", cwd="/testbed")


# ---------------------------------------------------------------------------
# 3. Finalization bookkeeping helpers
# ---------------------------------------------------------------------------


def test_format_observation_single_command():
    out = _format_observation(
        [{"command": "ls", "output": "file1\nfile2\n", "returncode": 0}]
    )
    assert "[Command 1 / returncode=0]" in out
    assert "$ ls" in out
    assert "file1" in out


def test_format_observation_multi_commands():
    out = _format_observation(
        [
            {"command": "cmd_a", "output": "out_a", "returncode": 0},
            {"command": "cmd_b", "output": "out_b", "returncode": 2},
        ]
    )
    assert "[Command 1 / returncode=0]" in out
    assert "[Command 2 / returncode=2]" in out
    assert out.index("cmd_a") < out.index("cmd_b")


def test_split_prompt_response_no_assistant_turn():
    """When no assistant token has been appended yet, the entire buffer is
    prompt and the response side is empty."""
    prompt_ids = [1, 2, 3, 4, 5]
    response_mask = [0, 0, 0, 0, 0]
    p, r, m, lp = _split_prompt_response(prompt_ids, response_mask, None)
    assert p == [1, 2, 3, 4, 5]
    assert r == []
    assert m == []
    assert lp is None


def test_split_prompt_response_one_assistant_turn():
    """One assistant turn mid-sequence: split at the first mask==1 index."""
    # prompt=[10,11], assistant_ids=[20,21,22], obs=[30,31]
    prompt_ids = [10, 11, 20, 21, 22, 30, 31]
    response_mask = [0, 0, 1, 1, 1, 0, 0]
    p, r, m, lp = _split_prompt_response(prompt_ids, response_mask, None)
    assert p == [10, 11]
    assert r == [20, 21, 22, 30, 31]
    assert m == [1, 1, 1, 0, 0]
    assert lp is None


def test_split_prompt_response_with_logprobs():
    prompt_ids = [1, 2, 3, 4]
    response_mask = [0, 1, 1, 0]
    logprobs = [0.0, -0.5, -0.25, 0.0]
    p, r, m, lp = _split_prompt_response(prompt_ids, response_mask, logprobs)
    assert p == [1]
    assert r == [2, 3, 4]
    assert m == [1, 1, 0]
    assert lp == [-0.5, -0.25, 0.0]


def test_truncate_prompt_left_truncates():
    """Prompt longer than max_prompt: keep the tail, not the head."""
    prompt_ids = list(range(100))  # 0..99
    response_ids = [1000, 1001, 1002]
    response_mask = [1, 1, 1]
    p, r, m, lp = _truncate(
        prompt_ids, response_ids, response_mask, None, max_prompt=10, max_response=10
    )
    # Last 10 tokens kept → [90..99]
    assert p == list(range(90, 100))
    assert r == [1000, 1001, 1002]
    assert m == [1, 1, 1]
    assert lp is None


def test_truncate_response_right_truncates():
    prompt_ids = [1, 2]
    response_ids = list(range(20))
    response_mask = [1] * 20
    logprobs = [-0.1] * 20
    p, r, m, lp = _truncate(
        prompt_ids, response_ids, response_mask, logprobs, max_prompt=10, max_response=5
    )
    assert p == [1, 2]
    assert r == [0, 1, 2, 3, 4]
    assert m == [1, 1, 1, 1, 1]
    assert lp == [-0.1] * 5


def test_truncate_both_fit_noop():
    p, r, m, lp = _truncate(
        [1, 2, 3], [4, 5], [1, 1], [-0.2, -0.3], max_prompt=10, max_response=10
    )
    assert p == [1, 2, 3]
    assert r == [4, 5]
    assert m == [1, 1]
    assert lp == [-0.2, -0.3]


# ---------------------------------------------------------------------------
# 4. Mock state machine
# ---------------------------------------------------------------------------


def test_mock_state_machine_single_turn_final():
    """Smoke test: end-to-end rollout where the very first assistant message
    emits a FINAL marker. We bypass ``AgentLoopBase.__init__`` entirely by
    constructing an instance via ``__new__`` and manually wiring the attrs
    that :meth:`MiniSweAgentLoop.run` touches. All heavy deps (server_manager,
    apptainer env, score_patch, tokenizer, apply_chat_template) are mocked.
    """
    import types

    from recipe.mini_swe.agent_loop import MiniSweAgentLoop

    loop = MiniSweAgentLoop.__new__(MiniSweAgentLoop)

    # ---- minimal attribute wiring mirroring AgentLoopBase.__init__ ----
    loop.prompt_length = 128
    loop.response_length = 128
    loop.max_assistant_turns = 5
    loop._prompt_cfg = _load_prompt_config()
    loop.trajectory_log_dir = None
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []

    # Event loop for run_in_executor.
    loop.loop = asyncio.new_event_loop()

    # Tokenizer: trivial char-based.
    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            # Embed a FINAL marker in turn 0.
            if ids == [900, 901]:
                return f"some reasoning\n{FINAL_MARKER}\ndiff --git a/x b/x\n"
            return "noise"

    loop.tokenizer = _FakeTokenizer()

    # apply_chat_template: deterministic. Returns small token lists so we
    # can trace bookkeeping.
    call_log = []

    async def _fake_act(messages, tools=None, images=None, videos=None, remove_system_prompt=False):
        call_log.append({"messages": messages, "remove_system_prompt": remove_system_prompt})
        # Initial (system+user) → 4 tokens. Delta → 2 tokens.
        if remove_system_prompt:
            return [500, 501]
        return [100, 101, 102, 103]

    loop.apply_chat_template = _fake_act  # bound-method replacement

    # server_manager.generate → canned TokenOutput with the FINAL marker.
    class _Tok:
        token_ids = [900, 901]
        log_probs = None
        num_preempted = 0

    class _FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, *, request_id, prompt_ids, sampling_params, **kw):
            self.calls.append(
                {"request_id": request_id, "prompt_len": len(prompt_ids)}
            )
            return _Tok()

    loop.server_manager = _FakeServerManager()

    # Fake env: constructed inside run(), so we patch the ApptainerEnvironment
    # constructor at the module level.
    class _FakeEnv:
        def __init__(self, **kw):
            self.config = types.SimpleNamespace(cwd="/testbed", env={}, patch_scratch_dir="/tmp")
            self.closed = False
            self.calls = []

        def execute(self, action, cwd=""):
            self.calls.append(action)
            return {"output": "", "returncode": 0}

        def close(self):
            self.closed = True

    with (
        mock.patch("recipe.mini_swe.agent_loop.ApptainerEnvironment", _FakeEnv),
        mock.patch(
            "recipe.mini_swe.agent_loop.score_patch", return_value=1.0
        ) as score_mock,
    ):
        sampling_params = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
        output = loop.loop.run_until_complete(
            loop.run(
                sampling_params,
                extra_info={
                    "instance_id": "inst_x",
                    "repo": "foo/bar",
                    "problem_statement": "fix it",
                    "base_commit": "abc123",
                    "sif_path": "/fake/image.sif",
                    "install_spec": "",
                    "fail_to_pass": ["t1"],
                    "pass_to_pass": [],
                    "test_runner": None,
                },
            )
        )

    # --- invariants ---
    # num_turns == 1 (single assistant turn), reward_score == 1.0 from mock.
    assert output.num_turns == 1
    assert output.reward_score == 1.0
    assert output.extra_fields["instance_id"] == "inst_x"
    assert output.extra_fields["exit_status"] == "submitted"
    assert output.extra_fields["submission_chars"] > 0
    # Response / mask alignment.
    assert len(output.response_ids) == len(output.response_mask)
    # Assistant tokens 900, 901 should appear in the response.
    assert 900 in output.response_ids and 901 in output.response_ids
    # server_manager was called exactly once.
    assert len(loop.server_manager.calls) == 1
    # score_patch was called with our patch.
    score_mock.assert_called_once()
    loop.loop.close()


def test_mock_state_machine_multi_turn_bash_then_final():
    """Two-turn rollout: turn 0 emits a bash command, turn 1 emits FINAL."""
    import types

    from recipe.mini_swe.agent_loop import MiniSweAgentLoop

    loop = MiniSweAgentLoop.__new__(MiniSweAgentLoop)
    loop.prompt_length = 256
    loop.response_length = 256
    loop.max_assistant_turns = 5
    loop._prompt_cfg = _load_prompt_config()
    loop.trajectory_log_dir = None
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []
    loop.loop = asyncio.new_event_loop()

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            if ids == [201, 202]:
                return (
                    "I should list files.\n"
                    f"{BASH_BEGIN}\nls /testbed\n{BASH_END}\n"
                )
            if ids == [301, 302]:
                return f"{FINAL_MARKER}\ndiff --git a/x b/y\n"
            return ""

    loop.tokenizer = _FakeTokenizer()

    async def _fake_act(messages, tools=None, images=None, videos=None, remove_system_prompt=False):
        if remove_system_prompt:
            return [777, 778, 779]  # obs delta
        return [10, 11, 12, 13]  # initial prompt

    loop.apply_chat_template = _fake_act

    counter = {"n": 0}

    class _FakeSM:
        def __init__(self):
            self.calls = []

        async def generate(self, *, request_id, prompt_ids, sampling_params, **kw):
            self.calls.append({"request_id": request_id, "prompt_len": len(prompt_ids)})
            counter["n"] += 1
            if counter["n"] == 1:
                return types.SimpleNamespace(token_ids=[201, 202], log_probs=None, num_preempted=0)
            return types.SimpleNamespace(token_ids=[301, 302], log_probs=None, num_preempted=1)

    loop.server_manager = _FakeSM()

    class _FakeEnv:
        def __init__(self, **kw):
            self.config = types.SimpleNamespace(cwd="/testbed", env={}, patch_scratch_dir="/tmp")
            self.calls = []

        def execute(self, action, cwd=""):
            self.calls.append(action)
            return {"output": "file1\nfile2", "returncode": 0}

        def close(self):
            pass

    with (
        mock.patch("recipe.mini_swe.agent_loop.ApptainerEnvironment", _FakeEnv),
        mock.patch("recipe.mini_swe.agent_loop.score_patch", return_value=0.0),
    ):
        sampling_params = {"temperature": 0.0}
        output = loop.loop.run_until_complete(
            loop.run(
                sampling_params,
                extra_info={
                    "instance_id": "inst_y",
                    "repo": "foo/bar",
                    "problem_statement": "fix",
                    "base_commit": "sha",
                    "sif_path": "/img.sif",
                    "install_spec": "",
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                    "test_runner": None,
                },
            )
        )

    # Two server_manager.generate calls → num_turns == 2
    assert output.num_turns == 2
    # Sticky session: same request_id across both turns.
    req_ids = {c["request_id"] for c in loop.server_manager.calls}
    assert len(req_ids) == 1
    # Prompt length grew between turns.
    assert loop.server_manager.calls[0]["prompt_len"] < loop.server_manager.calls[1]["prompt_len"]
    # Both assistant turns must appear in response_ids.
    assert all(tid in output.response_ids for tid in [201, 202, 301, 302])
    # Mask invariant.
    assert len(output.response_ids) == len(output.response_mask)
    assert sum(output.response_mask) >= 4  # 2 + 2 assistant tokens
    # At least one obs-delta (mask=0) token spliced between turns.
    assert 0 in output.response_mask
    assert output.extra_fields["exit_status"] == "submitted"
    # num_preempted accumulates across turns.
    assert output.metrics.num_preempted == 1
    loop.loop.close()


def test_mock_state_machine_missing_bash_nudges():
    """Assistant turn with neither FINAL marker nor BASH block → nudge text
    gets inserted as the observation. Second turn hits max_turns since we
    only allow 1 assistant turn."""
    import types

    from recipe.mini_swe.agent_loop import MiniSweAgentLoop

    loop = MiniSweAgentLoop.__new__(MiniSweAgentLoop)
    loop.prompt_length = 256
    loop.response_length = 256
    loop.max_assistant_turns = 1
    loop._prompt_cfg = _load_prompt_config()
    loop.trajectory_log_dir = None
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []
    loop.loop = asyncio.new_event_loop()

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "hmm no commands and no marker"

    loop.tokenizer = _FakeTokenizer()

    captured = {"obs_msgs": []}

    async def _fake_act(messages, tools=None, images=None, videos=None, remove_system_prompt=False):
        if remove_system_prompt:
            captured["obs_msgs"].append(messages)
            return [99, 98]
        return [1, 2, 3]

    loop.apply_chat_template = _fake_act

    class _FakeSM:
        async def generate(self, *, request_id, prompt_ids, sampling_params, **kw):
            return types.SimpleNamespace(token_ids=[50], log_probs=None, num_preempted=0)

    loop.server_manager = _FakeSM()

    class _FakeEnv:
        def __init__(self, **kw):
            self.config = types.SimpleNamespace(cwd="/testbed", env={}, patch_scratch_dir="/tmp")

        def execute(self, action, cwd=""):
            return {"output": "", "returncode": 0}

        def close(self):
            pass

    with (
        mock.patch("recipe.mini_swe.agent_loop.ApptainerEnvironment", _FakeEnv),
        mock.patch("recipe.mini_swe.agent_loop.score_patch", return_value=0.0),
    ):
        output = loop.loop.run_until_complete(
            loop.run(
                {"temperature": 0.0},
                extra_info={
                    "instance_id": "inst_z",
                    "repo": "r",
                    "problem_statement": "p",
                    "base_commit": "b",
                    "sif_path": "/img.sif",
                    "install_spec": "",
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                    "test_runner": None,
                },
            )
        )

    assert output.num_turns == 1
    assert output.extra_fields["exit_status"] == "max_turns"
    assert output.extra_fields["submission_chars"] == 0
    # The nudge message must have been tokenized as the obs delta.
    assert captured["obs_msgs"], "no obs delta was tokenized"
    # Look for a known phrase of the nudge.
    nudge_msg = captured["obs_msgs"][0][0]["content"]
    assert "wrap each shell command" in nudge_msg
    loop.loop.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
