# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-sample action projection for ALFWorld.

Ported from verl-agent (agent_system/environments/env_package/alfworld/projection.py)
which operates on a batch of responses; here we operate on a single response string
because the upstream verl agent_loop framework processes per-sample coroutines.

Validation rules (verbatim from verl-agent):
  * The raw response must contain both ``<think>...</think>`` and
    ``<action>...</action>`` tags.
  * The raw response must not contain any CJK unified ideograph
    (verl-agent's Chinese-character guard).

On any failure we return ``(fallback, False)`` where ``fallback`` mirrors
verl-agent's behaviour: the last 30 characters of the lowercased raw response
(a likely no-op in the env), or the empty string if the response is empty.

``admissible_commands`` is accepted for API stability (Phase 2 consumers expect
the signature) but is NOT used to flip ``valid``. verl-agent's projection does
not filter by admissibility, so we must not either, to preserve reward parity.
"""
from __future__ import annotations

import re

# CJK unified ideographs (same range as verl-agent projection.py).
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def alfworld_projection(
    raw_response: str,
    admissible_commands: list[str],
) -> tuple[str, bool]:
    """Parse a single LLM response and validate the extracted action.

    Args:
        raw_response: raw string emitted by the LLM for one sample.
        admissible_commands: accepted for API stability; not used for
            validation (see module docstring).

    Returns:
        (action_str, valid):
            * action_str -- the extracted action if valid, otherwise
              ``raw_response.lower()[-30:]`` (or ``""`` if the response is
              empty), matching verl-agent's fallback.
            * valid -- True iff the response passed every check.
    """
    del admissible_commands  # accepted for API stability only; see docstring.

    if not isinstance(raw_response, str) or raw_response == "":
        return "", False

    # Lowercased view mirrors verl-agent (which lowercases `actions[i]` before
    # tag lookup). The fallback slice below is taken from this lowered string.
    lowered = raw_response.lower()
    fallback = lowered[-30:]

    # 1. <think> tag check -- DISABLED.
    #    Originally mirrored verl-agent which required both <think></think> in
    #    the raw response. When we run with
    #    ``apply_chat_template_kwargs.enable_thinking=False``, Qwen3's chat
    #    template PREFILLS ``<think>\n\n</think>\n\n`` into the assistant
    #    prompt, so those tags live in the context we fed vLLM, not in the
    #    model's generated tokens -- `raw_response` never contains them in
    #    no-think mode, which incorrectly failed every episode.
    #    We now skip this check and rely on <action> tag extraction + CJK
    #    guard below. Responses without reasoning are still accepted; the
    #    only failure modes are missing <action> tag or CJK content.

    # 2. <action> tag presence + extraction. verl-agent only requires that
    #    both tags be present; it does NOT enforce `end > start`. Accept any
    #    ordering here to match.
    start_idx = lowered.find("<action>")
    end_idx = lowered.find("</action>")
    if start_idx == -1 or end_idx == -1:
        return fallback, False

    action_str = lowered[start_idx + len("<action>"):end_idx].strip()

    # 3. CJK guard (check on the raw response, matching verl-agent).
    #    No ASCII-only guard: verl-agent allows emoji / accents / full-width
    #    punctuation -- only CJK ideographs are rejected.
    if _CJK_RE.search(raw_response):
        return fallback, False

    return action_str, True
