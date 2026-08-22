"""Prompt helpers shared by tasks and reuse methods.

``ComposeReuse`` is the contract between a task's ``Case`` and the reuse
methods (CacheBlend, CacheBlendRepo, Naive): given prepared chunks and a run
prompt, it returns the longest prefix of ``run`` that can be explained as a
concatenation of prepared chunks (in some order), plus the fresh suffix.

The algorithm prioritizes the original prepare order (cheap for typical cases
where run follows prepare order), then falls back to searching other
permutations. It only supports suffix-style reuse: cached chunks form a
contiguous prefix, and any fresh text appears as a trailing suffix.
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
    and fresh text is always a trailing suffix. Methods that cannot blend
    intermediate fresh segments can use this directly; methods with blending
    capability can further process the result.
    """
    if not prepare:
        return None, run

    n = len(prepare)
    
    # Strategy 1: Try original order first (fast path for typical cases)
    original_order: List[str] = []
    pos = 0
    for chunk in prepare:
        if run.startswith(chunk, pos):
            original_order.append(chunk)
            pos += len(chunk)
        else:
            break
    
    if original_order:
        # Found at least one chunk in original order
        # Check if we can extend with DFS from current position
        best_len = pos
        best_order: List[str] = original_order
        
        # Try to extend beyond what original order achieved
        used = [False] * n
        for i, chunk in enumerate(original_order):
            # Mark chunks used in original_order
            for j, p in enumerate(prepare):
                if p == chunk and not used[j]:
                    used[j] = True
                    break
        
        def dfs_extend(cur_pos: int, current_order: List[str]):
            nonlocal best_len, best_order
            if cur_pos > best_len:
                best_len = cur_pos
                best_order = list(current_order)
            for i in range(n):
                if used[i]:
                    continue
                c = prepare[i]
                if run.startswith(c, cur_pos):
                    used[i] = True
                    current_order.append(c)
                    dfs_extend(cur_pos + len(c), current_order)
                    current_order.pop()
                    used[i] = False
        
        dfs_extend(pos, list(original_order))
        
        if best_len > 0:
            return best_order, run[best_len:]
    
    # Strategy 2: Full DFS search (no match in original order)
    used = [False] * n
    order: List[str] = []
    best: Tuple[int, List[str]] = (0, [])

    def bt(cur_pos: int) -> None:
        nonlocal best
        if cur_pos > best[0]:
            best = (cur_pos, list(order))
        for i in range(n):
            if used[i]:
                continue
            c = prepare[i]
            if run.startswith(c, cur_pos):
                used[i] = True
                order.append(c)
                bt(cur_pos + len(c))
                order.pop()
                used[i] = False

    bt(0)
    
    if best[0] > 0:
        return best[1], run[best[0]:]
    
    return None, run
