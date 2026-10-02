"""Prompt helpers shared by tasks and reuse methods.

``ComposeReuse`` and ``ComposeInterleavedReuse`` operate on token-ID lists.
Their ``*Text`` counterparts locate source chunks in rendered text while a
Task compiles the public token inputs.

Suffix-style reuse requires prepared chunks to form a contiguous prefix of
``run``, followed by at most one fresh suffix.

Interleaved reuse allows prepared chunks anywhere in ``run``, with fresh
tokens before, between, or after reused chunks.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple


def _ComposeInterleavedReuse(
    prepare,
    run,
):
    """
    Find non-overlapping occurrences of prepare chunks inside run.

    Ranking rules:

    1. Maximize the total number of covered sequence elements.
    2. Prefer fewer selected chunks when coverage is equal.
    3. Prefer transitions that preserve prepare order.

    A chunk may be reused multiple times.

    The matcher uses a short prefix of every chunk as an Aho-Corasick
    pattern. Full chunks are checked after a prefix hit.
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
    transitions: List[Dict[object, int]] = [{}]
    failure: List[int] = [0]
    terminal: List[List[Tuple[int, object, int]]] = [[]]

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
                if run[start:start + len(chunk)] == chunk:
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

    result = []

    def append_fresh(piece) -> None:
        if not piece:
            return

        if result and result[-1][0] is None:
            result[-1] = (None, result[-1][1] + piece)
        else:
            result.append((None, piece))

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


def ComposeInterleavedReuseText(
    prepare: List[str], run: str
) -> List[Tuple[Optional[int], str]]:
    """Locate reusable text chunks with the shared interleaved matcher."""
    return _ComposeInterleavedReuse(prepare, run)


def ComposeReuse(
    prepare: List[List[int]], run: List[int]
) -> Tuple[Optional[List[List[int]]], List[int]]:
    """Find the longest prefix of a token prompt covered by prepared chunks.

    ``order`` contains the selected prepared chunks in prompt order. Each
    chunk is used at most once; the returned suffix is a copy of the fresh
    trailing token IDs. The original prepare order wins ties.
    """
    if not prepare:
        return None, run[0:]

    candidateIndices = [index for index, chunk in enumerate(prepare) if chunk]
    if not candidateIndices:
        return None, run[0:]
    n = len(prepare)

    def matches(pos: int, chunk: List[int]) -> bool:
        end = pos + len(chunk)
        return end <= len(run) and list.__getitem__(run, slice(pos, end)) == chunk

    originalOrder: List[List[int]] = []
    originalIndices: List[int] = []
    pos = 0
    for index in candidateIndices:
        chunk = prepare[index]
        if matches(pos, chunk):
            originalOrder.append(chunk)
            originalIndices.append(index)
            pos += len(chunk)
        else:
            break

    if originalOrder:
        bestLen = pos
        bestIndices = list(originalIndices)
        used = [False] * n
        for index in originalIndices:
            used[index] = True

        def extend(curPos: int, current: List[int]) -> None:
            nonlocal bestLen, bestIndices
            if curPos > bestLen:
                bestLen = curPos
                bestIndices = list(current)
            for index in candidateIndices:
                chunk = prepare[index]
                if used[index] or not matches(curPos, chunk):
                    continue
                used[index] = True
                current.append(index)
                extend(curPos + len(chunk), current)
                current.pop()
                used[index] = False

        extend(pos, list(originalIndices))
        if bestLen > 0:
            return [prepare[index] for index in bestIndices], run[bestLen:]

    used = [False] * n
    order: List[int] = []
    bestLen = 0
    bestOrder: List[int] = []

    def search(curPos: int) -> None:
        nonlocal bestLen, bestOrder
        if curPos > bestLen:
            bestLen = curPos
            bestOrder = list(order)
        for index in candidateIndices:
            chunk = prepare[index]
            if used[index] or not matches(curPos, chunk):
                continue
            used[index] = True
            order.append(index)
            search(curPos + len(chunk))
            order.pop()
            used[index] = False

    search(0)
    if bestLen > 0:
        return [prepare[index] for index in bestOrder], run[bestLen:]
    return None, run[0:]


def ComposeInterleavedReuse(
    prepare: List[List[int]], run: List[int]
) -> List[Tuple[Optional[int], List[int]]]:
    """Select reusable token-ID spans with the same matcher as text chunks."""
    return _ComposeInterleavedReuse(prepare, run)
