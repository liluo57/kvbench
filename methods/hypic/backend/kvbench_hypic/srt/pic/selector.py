"""Token-selection adapters for hybrid PIC methods.

The LinearKV paper treats the selector as an independent plug-in.  HYPIC's
existing implementation only has a fixed seam selector, so the first
LinearKV selector is the static EPIC/LegoLink policy: recompute a prefix of
every matched reusable chunk.  The prefix can be configured as a fixed token
count or a per-chunk ratio.  The output is always absolute prompt positions in
context order; the recurrent backend must never infer a different order from
chunk or hash order.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def select_epic_positions(
    hit_segments: Sequence[Tuple[int, int, bytes]],
    tokens_per_chunk: Optional[int] = None,
    ratio: Optional[float] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Return static EPIC/LegoLink-style repair positions.

    EPIC's core idea is to repair the attention-sink tokens at each document
    beginning.  The budget can be expressed as a fixed prefix length or as a
    per-chunk ratio, but never both.  With neither argument, retain the
    historical two-token default for direct selector callers.
    """

    if tokens_per_chunk is not None and ratio is not None:
        raise ValueError("tokens_per_chunk and ratio are mutually exclusive")
    if tokens_per_chunk is None and ratio is None:
        tokens_per_chunk = 2
    if tokens_per_chunk is not None and (
        isinstance(tokens_per_chunk, bool)
        or not isinstance(tokens_per_chunk, int)
        or tokens_per_chunk < 0
    ):
        raise ValueError("tokens_per_chunk must be a non-negative integer")
    if ratio is not None and (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0.0 <= float(ratio) <= 1.0
    ):
        raise ValueError("ratio must be a finite number in [0, 1]")

    selected: Dict[Tuple[int, int], List[int]] = {}
    for start, end, _seg_hash in hit_segments:
        length = end - start
        if length <= 0:
            continue
        if ratio is None:
            count = int(tokens_per_chunk)
        else:
            count = math.floor(float(ratio) * length)
            if ratio > 0.0 and count == 0:
                count = 1
        if count > 0:
            selected[(start, end)] = list(
                range(start, min(end, start + count))
            )
    return selected


def merge_online_positions(
    miss_segments: Iterable[Tuple[int, int]],
    selected_positions: Dict[Tuple[int, int], Sequence[int]],
) -> List[int]:
    """Merge ordinary miss tokens and selected hit tokens in prompt order."""

    positions: List[int] = []
    for start, end in miss_segments:
        positions.extend(range(start, end))
    for selected in selected_positions.values():
        positions.extend(int(pos) for pos in selected)
    positions.sort()
    return positions
