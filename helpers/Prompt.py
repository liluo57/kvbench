"""Prompt helpers shared by tasks and reuse methods.

``SplitReuseParts`` is the contract between a task's ``Case`` and the prefix
reuse methods (CacheBlend, Naive): a method that cached the prepared context
``prepare`` tries to strip it from the complete prompt ``run`` to obtain the
fresh suffix. When the split fails (shuffled order, empty warm-up) the prefix
cannot be reused and a plain method falls back to a full prefill of ``run``.

``SplitReorderedReuse`` is the stronger contract a blend-capable method
(CacheBlend) additionally supports: it re-detects the prepared chunks inside a
reordered ``run`` (the shuffle tasks) so the segments can be reused by their
content-hash keys regardless of order.
"""

from typing import List, Optional, Tuple


def ContextText(prepare: List[str]) -> str:
    """Join the warm-up text segments into the one shared context string."""
    return "".join(prepare or [])


def SplitReuseParts(prepare: List[str], run: str) -> Tuple[str, Optional[str]]:
    """Return ``(contextText, suffixOrNone)`` for a prepare/run pair.

    - ``contextText`` is the concatenated prepared context.
    - ``suffixOrNone`` is the fresh tail of ``run`` after the context when
      ``run`` starts with it, else ``None`` (no prefix reuse possible).

    A ``None`` suffix means the prompt is not a superset of the prepared
    context (e.g. NIAHShuffleTask sends the segments in a different order, or
    the task prepared nothing). A blend-capable method may then call
    :func:`SplitReorderedReuse`; a plain one prefills ``run`` from scratch.
    """
    contextText = ContextText(prepare)
    if not contextText:
        return contextText, None
    if run.startswith(contextText):
        return contextText, run[len(contextText):]
    return contextText, None


def _FindOrder(prepare: List[str], run: str) -> Optional[List[str]]:
    """Search for an ordering of ``prepare`` (each chunk at most once) whose
    concatenation is a prefix of ``run``.

    Backtracking over ``run``'s start position; returns the longest cover found
    (an exact cover wins by consuming the whole ``run``). ``None`` when no
    chunk matches the front of ``run``. The search is cheap for the framework's
    tasks (chunks are prefix-distinct, so each position matches at most one
    chunk); ambiguous pools only widen the search, never produce a wrong answer
    — a mis-split cover just yields segments that miss their cache keys, which
    degrades to a fresh prefill of those spans.
    """
    n = len(prepare)
    used = [False] * n
    order: List[str] = []
    best: Tuple[int, List[str]] = (0, [])

    def bt(pos: int) -> None:
        nonlocal best
        if pos > best[0]:
            best = (pos, list(order))
        for i in range(n):
            if used[i]:
                continue
            c = prepare[i]
            if run.startswith(c, pos):
                used[i] = True
                order.append(c)
                bt(pos + len(c))
                order.pop()
                used[i] = False

    bt(0)
    return best[1] or None


def SplitReorderedReuse(
    prepare: List[str], run: str
) -> Tuple[Optional[List[str]], str]:
    """Return ``(order, suffix)`` when ``run`` is the ``prepare`` chunks joined
    in some (possibly different) order, plus a trailing fresh suffix.

    ``order`` is the chunk strings in the order they appear in ``run``, and
    ``suffix`` the text after the last chunk (empty when ``run`` is exactly the
    reordered chunks). The caller re-joins them at the token level with the
    segment separator ids: LMCache keys segments by *content hash*, so a
    reordered join still hits the stored per-chunk KV and the blender repairs
    positions. Returns ``(None, run)`` when ``run`` cannot be explained by the
    prepared chunks — the caller must then prefill ``run`` from scratch.
    """
    if not prepare:
        return None, run
    order = _FindOrder(prepare, run)
    if order is None:
        return None, run
    return order, run[sum(len(c) for c in order):]
