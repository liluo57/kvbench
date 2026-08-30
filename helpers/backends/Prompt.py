"""Prompt helpers shared by tasks and reuse methods.

``ComposeReuse`` supports suffix-style reuse: prepared chunks must form a
contiguous prefix of ``run``, followed by at most one fresh suffix.

``ComposeInterleavedReuse`` supports interleaved reuse: prepared chunks may
appear anywhere in ``run``, with fresh text before, between, or after reused
chunks.
"""

from collections import deque
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
    """
    Find non-overlapping occurrences of prepare chunks inside run.

    Ranking rules:

    1. Maximize the total number of covered characters.
    2. Prefer fewer selected chunks when coverage is equal.
    3. Prefer transitions that preserve prepare order.

    A chunk may be reused multiple times.

    The matcher uses a short prefix of every chunk as an Aho-Corasick
    pattern. Full chunks are checked with str.startswith after a prefix hit.
    This keeps the automaton small even when chunks themselves are very long.
    """
    if not run:
        return []

    if not prepare:
        return [(None, run)]

    # Empty chunks have infinitely many possible matches.
    patterns = [
        (index, chunk)
        for index, chunk in enumerate(prepare)
        if chunk
    ]

    if not patterns:
        return [(None, run)]

    # Long chunks are represented in the automaton by only a prefix.
    # Increase this value if the input has many common prefixes.
    anchor_size = 64

    # Each node stores:
    #   transitions[node]: character -> next node
    #   failure[node]: Aho-Corasick failure link
    #   terminal[node]: patterns ending at this node
    transitions: List[Dict[str, int]] = [{}]
    failure: List[int] = [0]
    terminal: List[List[Tuple[int, str, int]]] = [[]]

    for index, chunk in patterns:
        anchor_length = min(anchor_size, len(chunk))
        anchor = chunk[:anchor_length]

        node = 0

        for character in anchor:
            child = transitions[node].get(character)

            if child is None:
                child = len(transitions)

                transitions[node][character] = child
                transitions.append({})
                failure.append(0)
                terminal.append([])

            node = child

        terminal[node].append(
            (index, chunk, anchor_length)
        )

    # output_link[node] points to the nearest terminal failure state.
    # We use links instead of copying terminal lists into every node.
    output_link: List[int] = [-1] * len(transitions)

    queue = deque(transitions[0].values())

    while queue:
        node = queue.popleft()
        fail = failure[node]

        output_link[node] = (
            fail
            if terminal[fail]
            else output_link[fail]
        )

        for character, child in transitions[node].items():
            fallback = fail
            next_node = transitions[fallback].get(character)

            while next_node is None and fallback:
                fallback = failure[fallback]
                next_node = transitions[fallback].get(character)

            failure[child] = (
                0
                if next_node is None
                else next_node
            )

            queue.append(child)

    # ---------------------------------------------------------
    # Collect exact candidate intervals.
    # candidate = (prepare_index, start, end)
    # ---------------------------------------------------------

    candidates: List[Tuple[int, int, int]] = []
    node = 0

    for position, character in enumerate(run):
        next_node = transitions[node].get(character)

        while next_node is None and node:
            node = failure[node]
            next_node = transitions[node].get(character)

        node = (
            0
            if next_node is None
            else next_node
        )

        terminal_node = node

        while terminal_node != -1:
            for index, chunk, anchor_length in terminal[terminal_node]:
                end = position + 1
                start = end - anchor_length

                # The anchor is only a prefilter. This check guarantees
                # exact matching of the complete chunk.
                if run.startswith(chunk, start):
                    candidates.append(
                        (
                            index,
                            start,
                            start + len(chunk),
                        )
                    )

            terminal_node = output_link[terminal_node]

    if not candidates:
        return [(None, run)]

    # Sort by end position so a compatible predecessor always appears first.
    candidates.sort(
        key=lambda item: (
            item[2],  # end
            item[1],  # start
            item[0],  # prepare index
        )
    )

    # ---------------------------------------------------------
    # Weighted interval DP
    # ---------------------------------------------------------

    # score =
    #   (
    #       covered_characters,
    #       -number_of_selected_chunks,
    #       order_score,
    #   )
    #
    # Coverage is primary. The order preference cannot make us discard
    # otherwise useful matched text.
    count = len(candidates)

    scores: List[Tuple[int, int, int]] = [
        (0, 0, 0)
    ] * count

    parents: List[int] = [-1] * count

    for current in range(count):
        current_index, current_start, current_end = candidates[current]
        current_length = current_end - current_start

        best_score = (
            current_length,
            -1,
            0,
        )
        best_parent = -1

        for previous in range(current):
            previous_index, _, previous_end = candidates[previous]

            # Overlapping chunks cannot both be selected.
            if previous_end > current_start:
                continue

            previous_score = scores[previous]

            order_transition = (
                1
                if current_index >= previous_index
                else -1
            )

            candidate_score = (
                previous_score[0] + current_length,
                previous_score[1] - 1,
                previous_score[2] + order_transition,
            )

            if candidate_score > best_score:
                best_score = candidate_score
                best_parent = previous

        scores[current] = best_score
        parents[current] = best_parent

    # ---------------------------------------------------------
    # Recover selected intervals
    # ---------------------------------------------------------

    end_node = max(
        range(count),
        key=scores.__getitem__,
    )

    selected: List[Tuple[int, int, int]] = []

    while end_node >= 0:
        selected.append(candidates[end_node])
        end_node = parents[end_node]

    selected.reverse()

    # ---------------------------------------------------------
    # Build final result
    # ---------------------------------------------------------

    result: List[Tuple[Optional[int], str]] = []

    def append_fresh(text: str) -> None:
        if not text:
            return

        if result and result[-1][0] is None:
            previous_text = result[-1][1]
            result[-1] = (None, previous_text + text)
        else:
            result.append((None, text))

    position = 0

    for index, start, end in selected:
        if start > position:
            append_fresh(run[position:start])

        result.append(
            (
                index,
                run[start:end],
            )
        )

        position = end

    if position < len(run):
        append_fresh(run[position:])

    return result