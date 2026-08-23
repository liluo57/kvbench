"""Prompt helpers shared by tasks and reuse methods.

``ComposeReuse`` supports suffix-style reuse: prepared chunks must form a
contiguous prefix of ``run``, followed by at most one fresh suffix.

``ComposeInterleavedReuse`` supports interleaved reuse: prepared chunks may
appear anywhere in ``run``, with fresh text before, between, or after reused
chunks.
"""

from typing import List, Optional, Tuple


def ContextText(prepare: List[str]) -> str:
    """Join the warm-up text segments into the one shared context string."""
    return "".join(prepare or [])


def ComposeReuse(
    prepare: List[str], run: str
) -> Tuple[Optional[List[str]], str]:
    """Find the longest prefix of ``run`` explainable by ``prepare`` chunks.

    Returns ``(order, suffix)`` where:
    - ``order`` is the list of prepare chunks (each used at most once) whose
      concatenation equals the matched prefix of ``run``, in the order they
      appear in ``run``. Prefer the original prepare order when possible.
    - ``suffix`` is the remaining text after the matched prefix (empty if the
      entire ``run`` is covered).

    Returns ``(None, run)`` when no prepare chunk matches the start of ``run``.

    The algorithm:
    1. First try matching prepare chunks in their original order (greedy).
    2. If that fails to consume any chunk, fall back to DFS search over all
       permutations (still each chunk used at most once).
    3. Return the longest match found (original order wins on ties).

    This supports suffix-only reuse: cached blocks form a contiguous prefix,
    and fresh text is always a trailing suffix.
    """
    if not prepare:
        return None, run

    n = len(prepare)

    # Strategy 1: Try original order first (fast path for typical cases)
    originalOrder: List[str] = []
    pos = 0

    for chunk in prepare:
        if run.startswith(chunk, pos):
            originalOrder.append(chunk)
            pos += len(chunk)
        else:
            break

    if originalOrder:
        bestLen = pos
        bestOrder: List[str] = originalOrder

        used = [False] * n

        for chunk in originalOrder:
            for i, preparedChunk in enumerate(prepare):
                if preparedChunk == chunk and not used[i]:
                    used[i] = True
                    break

        def DfsExtend(curPos: int, currentOrder: List[str]) -> None:
            nonlocal bestLen, bestOrder

            if curPos > bestLen:
                bestLen = curPos
                bestOrder = list(currentOrder)

            for i in range(n):
                if used[i]:
                    continue

                chunk = prepare[i]

                if run.startswith(chunk, curPos):
                    used[i] = True
                    currentOrder.append(chunk)

                    DfsExtend(curPos + len(chunk), currentOrder)

                    currentOrder.pop()
                    used[i] = False

        DfsExtend(pos, list(originalOrder))

        if bestLen > 0:
            return bestOrder, run[bestLen:]

    # Strategy 2: Full DFS search
    used = [False] * n
    order: List[str] = []
    best: Tuple[int, List[str]] = (0, [])

    def Dfs(curPos: int) -> None:
        nonlocal best

        if curPos > best[0]:
            best = (curPos, list(order))

        for i in range(n):
            if used[i]:
                continue

            chunk = prepare[i]

            if run.startswith(chunk, curPos):
                used[i] = True
                order.append(chunk)

                Dfs(curPos + len(chunk))

                order.pop()
                used[i] = False

    Dfs(0)

    if best[0] > 0:
        return best[1], run[best[0]:]

    return None, run


def ComposeInterleavedReuse(
    prepare: List[str],
    run: str,
) -> List[Tuple[Optional[int], str]]:
    """Split ``run`` into interleaved reusable and fresh segments.

    Each returned item is ``(prepareIndex, text)``:

    - ``prepareIndex is None`` means ``text`` is fresh and must be prefetched.
    - Otherwise, ``text`` corresponds to ``prepare[prepareIndex]`` and may be
      reused.

    Each prepared chunk can be used at most once.

    Unlike ``ComposeReuse``, reusable chunks do not need to form a contiguous
    prefix. Fresh text may appear before, between, or after reusable chunks.

    The search maximizes the total number of reused characters. When multiple
    solutions reuse the same amount of text, earlier chunks in ``prepare`` are
    preferred because DFS explores them first.

    Examples::

        prepare = ["A", "C"]
        run = "ABCD"

        -> [
            (0, "A"),
            (None, "B"),
            (1, "C"),
            (None, "D"),
        ]

        prepare = ["A", "C"]
        run = "BACD"

        -> [
            (None, "B"),
            (0, "A"),
            (1, "C"),
            (None, "D"),
        ]

    Empty prepared chunks are ignored because they carry no reusable text.
    """
    if not run:
        return []

    if not prepare:
        return [(None, run)]

    n = len(prepare)
    used = [False] * n

    currentMatches: List[Tuple[int, int, int]] = []
    bestMatches: List[Tuple[int, int, int]] = []
    bestReusedLen = 0

    def Dfs(curPos: int, reusedLen: int) -> None:
        nonlocal bestMatches, bestReusedLen

        # Strictly greater only: because candidates are visited in prepare
        # order, the first solution wins when reused length ties.
        if reusedLen > bestReusedLen:
            bestReusedLen = reusedLen
            bestMatches = list(currentMatches)

        for i in range(n):
            if used[i]:
                continue

            chunk = prepare[i]

            if not chunk:
                continue

            matchPos = run.find(chunk, curPos)

            if matchPos < 0:
                continue

            used[i] = True
            matchEnd = matchPos + len(chunk)

            currentMatches.append((i, matchPos, matchEnd))

            Dfs(
                matchEnd,
                reusedLen + len(chunk),
            )

            currentMatches.pop()
            used[i] = False

    Dfs(0, 0)

    if not bestMatches:
        return [(None, run)]

    result: List[Tuple[Optional[int], str]] = []
    pos = 0

    for prepareIndex, start, end in bestMatches:
        if start > pos:
            result.append((None, run[pos:start]))

        result.append((prepareIndex, run[start:end]))
        pos = end

    if pos < len(run):
        result.append((None, run[pos:]))

    return result