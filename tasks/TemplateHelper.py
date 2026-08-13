"""Chat-template prompt pieces for the configured model (Qwen2.5-Instruct).

Shared by the RULER tasks (via ``_Ruler``) and the cacheblend knowledge-base
tasks (``Cacheblend``). The reference data is Mistral-formatted (``[INST]`` /
``[/INST]``); a Qwen model does not understand that. Tasks therefore build the
*text* of a Qwen chat prompt (special tokens are written literally — the Qwen
tokenizer tokenizes them correctly). A complete prompt is one user turn whose
``input`` is the body, followed by the assistant header and the RULER
``answer_prefix``:

    "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{answer_prefix}"

``UserContext`` / ``AssistantSuffix`` assemble the two reusable pieces that
reuse methods cache and fuse (the fresh query is the tail of the user turn,
closed by the assistant header).
"""

ImStart = "<|im_start|>"
ImEnd = "<|im_end|>"


def UserContext(body: str) -> str:
    """Wrap ``body`` as the body of a user turn (context part)."""
    return f"{ImStart}user\n{body}"


def AssistantSuffix(tail: str) -> str:
    """Close the user turn and open the assistant turn (suffix part)."""
    return f"{tail}{ImEnd}\n{ImStart}assistant\n"
