"""Single thin adapter that owns every per-arch chat-rendering decision.

Tasks call into here. They never read ``config.json``, never branch on
``Qwen3`` vs ``MuseGlimmer``, never concatenate ```` /
```` by hand. The adapter resolves the configured model's
architecture, then routes through:

- the model's own ``chat_template`` (via ``tokenizer.apply_chat_template``)
  for prompt rendering — so every model sees the exact format it was
  trained on, and per-arch kwargs (``enable_thinking`` for Qwen3,
  ``reasoning_strength`` for Muse Glimmer) are set automatically;
- vLLM's ``ToolParserManager`` + ``ReasoningParserManager`` for parsing
  the model's output back into OpenAI-shaped tool / reasoning fields.

For RULER / KB tasks that split a prompt into a cacheable prefix and a
fresh tail (so reuse methods can store each segment's KV once), the
adapter also exposes :func:`user_turn_prefix` and
:func:`assistant_turn_suffix` — literal boundary strings derived from
the same jinja, so chunk boundaries stay byte-identical to what
``render_chat`` would emit if the whole prompt were rendered at once.

``modelPath`` is passed explicitly by every caller — the adapter does NOT
read ``core.Config`` itself, so :mod:`helpers` stays one-way (no upward
dependency from helpers into core).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from time import time_ns
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Architecture detection — the ONLY place in kvbench that knows arch names.
# ---------------------------------------------------------------------------

#: Architectures whose Instruct chat defaults to thinking mode (Qwen3). Adding
#: a new model means appending to one of these tuples.
_Qwen3Archs: Tuple[str, ...] = ("Qwen3ForCausalLM",)

#: Qwen3.5 / Qwen3.8 and later — distinct lineage from Qwen3:
#: - hybrid linear+full attention (Mamba-style SSM layers interleaved with
#:   full-attention layers — no per-token KV cache for the linear layers).
#: - ships as multimodal ``*ForConditionalGeneration`` (text-only selected
#:   at load time via vLLM's ``language_model_only=True``).
#: - chat template defaults to opening ``<think>\n`` and emits a system-level
#:   "Reasoning effort is set to xhigh" preamble unless the template is given
#:   ``enable_thinking=False`` (the same kwarg Qwen3 reads).
#: We treat this as its own family rather than folding it into ``qwen3``
#: because: (1) the architecture is materially different (linear attention
#: changes KV-cache layout), and (2) the default-rendered prompt format
#: diverges enough that concat-style chunk prompts (RAG/KBBase's legacy path)
#: hit different failure modes than Qwen3.
_Qwen3_5Archs: Tuple[str, ...] = (
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
)

#: Architectures that use the Meta-style ``<|start|>/<|message|>/<|eot|>`` chat
#: format. vLLM maps ``MuseGlimmerForConditionalGeneration`` to the
#: ``MuseGlimmerForCausalLM`` class (text-only), so both names appear.
_MuseGlimmerArchs: Tuple[str, ...] = (
    "MuseGlimmerForCausalLM",
    "MuseGlimmerForConditionalGeneration",
)

#: Test seam: when non-None, :func:`arch_family` returns this verbatim
#: instead of reading ``config.json``. Production code never sets this.
_ArchOverrideForTesting: Optional[str] = None


def _read_arch_from_config_json(modelPath: str) -> Optional[Literal["qwen3", "qwen3_5", "muse_glimmer"]]:
    """Read ``<modelPath>/config.json`` and map ``architectures`` to a family name.

    Returns ``None`` when the file is missing or malformed, when the modelPath
    is empty, or when no supported arch appears in the list. The caller
    (typically :func:`arch_family`) decides what ``None`` means.
    """
    if not modelPath:
        return None
    try:
        with open(Path(modelPath) / "config.json", encoding="utf-8") as f:
            archs = json.load(f).get("architectures", [])
    except (OSError, ValueError):
        return None
    if any(a in _MuseGlimmerArchs for a in archs):
        return "muse_glimmer"
    if any(a in _Qwen3Archs for a in archs):
        return "qwen3"
    if any(a in _Qwen3_5Archs for a in archs):
        return "qwen3_5"
    return None


@lru_cache(maxsize=8)
def arch_family(modelPath: str = "") -> Literal["qwen3", "qwen3_5", "muse_glimmer", "other"]:
    """Return the chat-format arch for the configured model.

    Reads ``<modelPath>/config.json`` and inspects the ``architectures`` list.
    Any I/O failure (missing model dir / malformed config) falls back to
    ``"other"``. Honors :data:`_ArchOverrideForTesting` for tests.
    """
    if _ArchOverrideForTesting is not None:
        return _ArchOverrideForTesting
    return _read_arch_from_config_json(modelPath) or "other"


def set_arch_for_testing(arch: Optional[str]) -> None:
    """Force :func:`arch_family` to return ``arch`` until cleared.

    Pass ``None`` to restore production behaviour. Test-only — production
    code never touches :data:`_ArchOverrideForTesting`.
    """
    global _ArchOverrideForTesting
    _ArchOverrideForTesting = arch
    arch_family.cache_clear()
    # Prompt boundaries depend on the selected chat template as well. Clear
    # their lazy caches when tests switch the synthetic architecture.
    for name in ("_chat_boundary_parts", "user_turn_prefix", "assistant_turn_suffix"):
        cached = globals().get(name)
        if cached is not None:
            cached.cache_clear()


# ---------------------------------------------------------------------------
# Tokenizer (lazy) — used by every chat-rendering path below.
# ---------------------------------------------------------------------------


def _tokenizer(modelPath: str):
    """Lazy-load a ``transformers`` tokenizer for the configured model.

    The Method (vLLM / transformers backend) keeps its own tokenizer; loading
    one here lets the adapter render prompts and instantiate parsers without
    holding a back-reference to the Method. Tokenizer files are small
    (<100MB), the load is ~2s and happens once per (modelPath).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(modelPath)


# ---------------------------------------------------------------------------
# Chat-prompt rendering via native jinja.
# ---------------------------------------------------------------------------


#: Per-arch (template kwarg name) -> True/False/None value mapping. ``None``
#: means "let the jinja decide" — the kwarg is omitted entirely so the
#: template's own default takes effect. ``True``/``False`` set the kwarg.
def _thinking_kwargs(thinking: Optional[bool], modelPath: str) -> Dict[str, Any]:
    arch = arch_family(modelPath)
    if thinking is None:
        return {}
    if arch in ("qwen3", "qwen3_5"):
        # Both Qwen3 and Qwen3.5 chat templates read the same
        # ``enable_thinking`` kwarg. Qwen3.5 ships an additional
        # ``reasoning_effort`` (``xhigh``/``medium``/``low``) that defaults
        # to ``xhigh`` and prepends a "Reasoning effort is set to xhigh"
        # system-prompt preamble unless explicitly passed.
        if arch == "qwen3_5":
            return {
                "enable_thinking": bool(thinking),
                "reasoning_effort": "low" if not thinking else "xhigh",
            }
        return {"enable_thinking": bool(thinking)}
    if arch == "muse_glimmer":
        return {"reasoning_strength": "high" if thinking else "low"}
    return {}


def _normalise_messages(
    messages: List[Dict[str, Any]],
    system_prefix: str,
) -> List[Dict[str, Any]]:
    """Fold ``system_prefix`` into the first system message and JSON-decode
    tool-call ``arguments`` strings back to dicts (ATEM template quirk)."""
    msgs: List[Dict[str, Any]] = []
    systemPrepended = False
    for m in messages:
        role = str(m.get("role", "user"))
        if role == "system" and not systemPrepended and system_prefix:
            baseContent = m.get("content")
            content = (
                f"{system_prefix.rstrip()}\n\n{baseContent}"
                if baseContent
                else system_prefix
            )
            msgs.append({"role": "system", "content": content})
            systemPrepended = True
            continue
        if role == "assistant" and m.get("tool_calls"):
            tcs: List[Dict[str, Any]] = []
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") if isinstance(tc, Mapping) else None
                if not isinstance(fn, Mapping):
                    continue
                fn_d = vars(fn) if hasattr(fn, "__dict__") else dict(fn)
                args = fn_d.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                tc_d = vars(tc) if hasattr(tc, "__dict__") else dict(tc)
                tcs.append({
                    "id": tc_d.get("id", f"call_{len(tcs)}"),
                    "type": tc_d.get("type", "function"),
                    "function": {
                        "name": fn_d.get("name"),
                        "arguments": args,
                    },
                })
            msgs.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": tcs,
            })
            continue
        msgs.append(m)
    if not systemPrepended and system_prefix:
        msgs.insert(0, {"role": "system", "content": system_prefix})
    return msgs


def render_chat(
    messages: List[Dict[str, Any]],
    *,
    modelPath: str,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    system_prefix: str = "",
    thinking: Optional[bool] = None,
) -> str:
    """Render an OpenAI chat message list to a text prompt for the configured
    model.

    Routing:

    1. ``tokenizer.apply_chat_template`` with ``add_generation_prompt=True`` —
       the model sees the exact chat format it was trained on (Qwen ChatML,
       Glimmer ATEM, Mistral [INST], …).
    2. The per-arch CoT toggle is passed as a top-level kwarg to
       ``apply_chat_template`` (Qwen3 reads ``enable_thinking``; Muse
       Glimmer reads ``reasoning_strength``; other archs ignore the kwarg).
       transformers' ``apply_chat_template`` reads top-level kwargs, not
       values nested in ``chat_template_kwargs`` — the dict-form only works
       on newer (>= 4.45) templates that opt in.

    ``system_prefix`` is folded into the first system message's content so
    the model's existing system role keeps its place. ``tools`` are passed
    through to the tokenizer as-is.
    """
    tok = _tokenizer(modelPath)
    msgs = _normalise_messages(list(messages), system_prefix)
    return tok.apply_chat_template(
        msgs,
        tools=list(tools) if tools else None,
        tokenize=False,
        add_generation_prompt=True,
        **_thinking_kwargs(thinking, modelPath),
    )


def render_user_prompt(
    userContent: str,
    *,
    modelPath: str,
    thinking: Optional[bool] = False,
) -> str:
    """Render a single ``user`` message as a complete, model-ready prompt.

    The chat-template renderer (see :func:`render_chat`) is the only thing
    that knows about the per-arch turn boundaries — Qwen3 / Qwen3.5 ChatML
    vs. Glimmer ATEM vs. Mistral [INST]. Tasks that build a single user
    message out of arbitrarily-many document chunks + a query suffix
    (KBBase / QABase / SumBase) should funnel through here rather than
    concatenating chunks + ``assistant_turn_suffix`` themselves: the chunk
    concat puts the user-opener marker in the middle of the prompt and
    Qwen3.5's chat template then lands trailing text outside the turn
    boundary, which makes the model predict EOS (verified: empty output
    on Qwen3.8-27B KB tasks with the concat path, correct answers via
    this function).

    ``thinking=False`` is the default — KB / RAG / summarisation tasks
    have short ``max_new_tokens`` and a CoT preamble would burn the budget
    before the answer. Multi-agent paths that want CoT pass ``thinking=True``.
    """
    return render_chat(
        [{"role": "user", "content": userContent}],
        modelPath=modelPath,
        thinking=thinking,
    )


# ---------------------------------------------------------------------------
# Chat-prompt boundary primitives — for tasks that split a prompt into chunks.
# ---------------------------------------------------------------------------


_BOUNDARY_MARKER = "KVBENCH_CHAT_BOUNDARY_MARKER_7f3a"


@lru_cache(maxsize=16)
def _chat_boundary_parts(
    modelPath: str, thinking: Optional[bool]
) -> Tuple[str, str]:
    """Return ``(prefix, assistant_suffix)`` for one user message.

    Splitting a rendered message around a marker is robust across ChatML,
    ATEM, ``[INST]`` and similar templates. In particular, an empty string is
    a valid ``str.find`` match at offset zero, so it must never be used as a
    boundary sentinel here.
    """
    rendered = render_chat(
        [{"role": "user", "content": _BOUNDARY_MARKER}],
        modelPath=modelPath,
        thinking=thinking,
    )
    idx = rendered.find(_BOUNDARY_MARKER)
    if idx < 0:
        raise RuntimeError(
            "could not locate chat boundary marker in rendered prompt: "
            f"{rendered!r}"
        )
    end = idx + len(_BOUNDARY_MARKER)
    return rendered[:idx], rendered[end:]


@lru_cache(maxsize=16)
def user_turn_prefix(
    modelPath: str, thinking: Optional[bool] = None
) -> str:
    """The literal prefix that opens a user turn in the configured model.

    Derived by rendering an empty user message: the jinja emits the user
    opener (``user\\n`` for Qwen), then the empty body, then the
    user-turn closer (``\\n``). We return everything up to but
    not including the closer. Tasks that build chunked prompts use this
    instead of literal ``user\\n`` — the boundary stays correct
    for every supported chat format without per-arch code.
    """
    return _chat_boundary_parts(modelPath, thinking)[0]


@lru_cache(maxsize=16)
def assistant_turn_suffix(
    modelPath: str, thinking: Optional[bool] = None
) -> str:
    """The literal suffix that closes the user turn and opens the assistant
    turn.

    Pairs with :func:`user_turn_prefix` for tasks that compose prompts by
    string concatenation rather than full-template rendering (RULER's
    shuffled variants, FreshGap). For Qwen3 / Qwen3.5 this is the raw
    template-rendered suffix (Qwen3's default emits nothing extra; Qwen3.5
    emits ``<think>\\n`` — the model's own CoT toggle). For Glimmer it is
    ``<|eot|>\\n<|start|>assistant<|message|>``.

    KB tasks do *not* use this — they route through :func:`render_user_prompt`
    because Qwen3.5's open-think-default prompts land outside the turn
    boundary when chunk-concatenated, and the model predicts EOS on the
    trailing text. The fix is the prompt structure, not this boundary.
    """
    return _chat_boundary_parts(modelPath, thinking)[1]


# ---------------------------------------------------------------------------
# Tool-call and reasoning parsing via vLLM-native parsers.
# ---------------------------------------------------------------------------


#: arch -> (tool_parser_name, reasoning_parser_name). The dispatcher is a
#: flat table so adding a new arch is one row. vLLM-native parsers handle
#: both protocol framing (Qwen3 XML, ATEM ``<|tool_call|>``) and
#: reasoning-channel stripping (``ReasoningParserManager``).
_ARCH_PARSERS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    # arch               tool_parser       reasoning_parser
    "muse_glimmer":      ("muse_glimmer",   "muse_glimmer"),
    "qwen3":             ("qwen3_xml",      "qwen3"),
    # Qwen3.5 / Qwen3.8 keep the same ``<tool_call><function=...>`` XML
    # framing as Qwen3 (see its ``chat_template.jinja``) so the vLLM-shipped
    # ``qwen3_xml`` parser handles both, and ``qwen3`` strips the
    # ``<think>...</think>`` reasoning block the same way.
    "qwen3_5":           ("qwen3_xml",      "qwen3"),
    # Anything not listed raises in parse_tool_calls.
}


def parse_tool_calls(
    output: str,
    payload: Mapping[str, Any],
    *,
    modelPath: str,
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Parse a vLLM ``generate`` output into OpenAI chat-completion fields.

    Returns ``(content, reasoning, tool_calls)``. Uses vLLM's
    ``ToolParserManager`` + ``ReasoningParserManager`` for the configured
    arch. Raises if the arch has no registered parser pair — every model
    KVBench currently supports has one.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import (  # type: ignore[import-not-found]
        ChatCompletionRequest,
    )
    from vllm.reasoning import ReasoningParserManager  # type: ignore[import-not-found]
    from vllm.tool_parsers import ToolParserManager  # type: ignore[import-not-found]

    arch = arch_family(modelPath)
    tool_name, reasoning_name = _ARCH_PARSERS.get(arch, (None, None))
    if tool_name is None or reasoning_name is None:
        raise ValueError(
            f"no vLLM-native parser pair registered for arch "
            f"{arch!r}; add a row to ModelAdapter._ARCH_PARSERS "
            f"or extend the supported arch list"
        )

    tok = _tokenizer(modelPath)
    tool_cls = ToolParserManager.get_tool_parser(tool_name)
    reasoning_cls = ReasoningParserManager.get_reasoning_parser(reasoning_name)
    tool_parser = tool_cls(tok)
    reasoning_parser = reasoning_cls(tok)

    request = ChatCompletionRequest.model_validate(dict(payload))
    reasoning, content = reasoning_parser.extract_reasoning(output, request)
    extracted = tool_parser.extract_tool_calls(content or output, request)
    raw_calls = list(extracted.tool_calls) if extracted.tool_calls else []

    # vLLM's parsers return pydantic ``ToolCall`` / ``FunctionCall`` models
    # (vllm.entrypoints.openai.engine.protocol.ToolCall extends
    # ``OpenAIBaseModel``). They carry a stray ``id=None`` that breaks strict
    # OpenAI clients, so reshape to plain dicts.
    clean_calls: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw_calls):
        function = tc.function
        arguments = function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(dict(arguments), ensure_ascii=False)
        clean_calls.append({
            "id": tc.id or f"call_{time_ns()}_{i}",
            "type": "function",
            "function": {
                "name": function.name,
                "arguments": arguments,
            },
        })

    return extracted.content or "", reasoning, clean_calls
