"""Chat-template prompt pieces for the configured model.

Shared by the RULER tasks (:mod:`tasks.bases.RulerBase`) and the
knowledge-base tasks (:mod:`tasks.bases.KBBase`). The reference data is
Mistral-formatted (``[INST]`` / ``[/INST]``); other models do not understand
that. Tasks therefore build the *text* of a model-appropriate chat prompt
(special tokens are written literally — the tokenizer tokenizes them
correctly). Two formats are supported:

* **Qwen chat** (default; covers Qwen2 / Qwen3 and any model that shares the
  format). A complete prompt is one user turn whose ``input`` is the body,
  followed by the assistant header and the RULER ``answer_prefix``:

    "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{answer_prefix}"

* **Muse Glimmer** (``<|start|>...<|message|>...<|eot|>`` style, Meta-derived;
  detected from ``architectures`` containing ``MuseGlimmer*``). The default
  system prompt is baked into :func:`UserContext` so the prefix cache stays
  stable across requests; the user turn ends with ``<|eot|>`` and the
  assistant turn opens with ``<|message|>`` so the ``answer_prefix`` slots
  in naturally:

    "<|start|>system<|message|>You are a helpful AI assistant.\n"
    "Knowledge cutoff: 2026-01-04.\n\nReasoning strength: high.\n\n"
    "# Valid recipients: \"self\", \"user\".<|eot|>"
    "<|start|>user<|message|>{input}<|eot|>"
    "<|start|>assistant<|message|>{answer_prefix}"

``UserContext`` / ``AssistantSuffix`` assemble the two reusable pieces that
reuse methods cache and fuse (the fresh query is the tail of the user turn,
closed by the assistant header).

Qwen3 thinking mode
-------------------
Qwen3-Instruct defaults to **thinking mode**: given ``<|im_start|>assistant\n``
it opens a ``<think>`` block and reasons before answering. KVBench runs greedy
(``temperature=0``) with a small token budget, so on the knowledge-base tasks
the whole budget is consumed by the reasoning trace and the real answer never
appears — F1 / ROUGE-L collapse to 0. (The RULER shuffle tasks only survive
because their ``answer_prefix`` forces the answer value out *before* the trace.)

vLLM's ``enable_thinking`` switch lives at the chat-template / entrypoint layer;
KVBench feeds the methods raw token ids, so the switch must be baked into the
prompt text. Qwen3's own chat template renders non-thinking mode as an
*empty, pre-closed* think block right after the assistant header:

    "<|im_start|>assistant\n<think>\n\n</think>"

:func:`AssistantSuffix` appends exactly that when the configured model defaults
to thinking (detected from the model's ``config.json`` ``architectures``), so
every method shares the fix. Non-Qwen models keep the plain header.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from core.Config import ModelPath as _ModelPath

ImStart = "<|im_start|>"
ImEnd = "<|im_end|>"

#: Architectures whose Instruct chat defaults to thinking mode (Qwen3). The
#: out-of-tree CacheBlend helper uses the same ``config.json`` signal to detect
#: Qwen3, so the two stay consistent.
_ThinkingDefaultArchs = ("Qwen3ForCausalLM",)

#: The exact prefix Qwen3's chat template emits for ``enable_thinking=False``:
#: an empty, pre-closed think block that tells the model to answer directly.
_NonThinkingBlock = "<think>\n\n</think>"

#: Architectures that use the Meta-style ``<|start|>/<|message|>/<|eot|>``
#: chat format. vLLM maps ``MuseGlimmerForConditionalGeneration`` to the
#: ``MuseGlimmerForCausalLM`` class (text-only), so both names appear.
_MuseGlimmerArchs = (
    "MuseGlimmerForCausalLM",
    "MuseGlimmerForConditionalGeneration",
)

#: The Muse Glimmer default system prompt. The model's chat template renders
#: this when no system message is supplied; we reproduce it verbatim here so
#: the prefix cache hits across requests (date is fixed at the model's
#: knowledge cutoff to keep the prefix stable).
_MuseSystemPrefix = (
    '<|start|>system<|message|>You are a helpful AI assistant.\n'
    'Knowledge cutoff: 2026-01-04.\n\n'
    'Reasoning strength: high.\n\n'
    '# Valid recipients: "self", "user".<|eot|>'
)


def UserContext(body: str) -> str:
    """Wrap ``body`` as the body of a user turn (context part)."""
    if _IsMuseGlimmer():
        return f"{_MuseSystemPrefix}<|start|>user<|message|>{body}"
    return f"{ImStart}user\n{body}"


def AssistantSuffix(
    tail: str, *, nonThinking: Optional[bool] = None
) -> str:
    """Close the user turn and open the assistant turn (suffix part).

    For a thinking-default model (Qwen3) the assistant header is followed by
    the empty pre-closed ``<think></think>`` block, so the model answers
    directly instead of burning the generation budget on a reasoning trace.
    ``nonThinking`` overrides the auto-detection.

    For Muse Glimmer the assistant turn is opened with ``<|message|>`` so
    RULER's ``answer_prefix`` slots in naturally (the chat template's
    generation prompt ends at ``<|start|>assistant``, but reusing methods
    need a complete turn start to reason about boundary tokens).
    """
    if _IsMuseGlimmer():
        return f"{tail}<|eot|><|start|>assistant<|message|>"
    header = f"{tail}{ImEnd}\n{ImStart}assistant\n"
    if nonThinking is False:
        return header
    if nonThinking is None and not _ThinksByDefault():
        return header
    return header + _NonThinkingBlock


@lru_cache(maxsize=1)
def _ThinksByDefault() -> bool:
    """Whether the configured model's Instruct chat defaults to thinking.

    Reads ``config.json`` at the configured ``ModelPath`` and looks for a
    thinking-default architecture (``Qwen3ForCausalLM``). Any failure (missing
    model dir / config) falls back to ``False`` — the plain assistant header.
    """
    modelPath = _ModelPath()
    if not modelPath:
        return False
    try:
        cfgPath = Path(modelPath) / "config.json"
        with open(cfgPath, encoding="utf-8") as f:
            archs = json.load(f).get("architectures", [])
    except (OSError, ValueError):
        return False
    return any(a in _ThinkingDefaultArchs for a in archs)


@lru_cache(maxsize=1)
def _IsMuseGlimmer() -> bool:
    """Whether the configured model is a Muse Glimmer variant.

    Reads ``config.json`` at the configured ``ModelPath`` and looks for a
    Muse-Glimmer architecture. Any failure (missing model dir / config)
    falls back to ``False`` — the plain Qwen-style header.
    """
    modelPath = _ModelPath()
    if not modelPath:
        return False
    try:
        cfgPath = Path(modelPath) / "config.json"
        with open(cfgPath, encoding="utf-8") as f:
            archs = json.load(f).get("architectures", [])
    except (OSError, ValueError):
        return False
    return any(a in _MuseGlimmerArchs for a in archs)
