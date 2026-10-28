"""Token-selection adapters for hybrid PIC methods.

EPIC repairs chunk prefixes, excluding the already-positioned first chunk
for partial repair. CacheBlend ranks squared value drift; ProphetKV ranks
context-conditioned query attention to frozen cached keys.
The dynamic probe orchestration lives in runtime_selector. Outputs always
use absolute prompt positions in context order, irrespective of score order.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def select_initial_positions(hit_segments, selector="epic", *, ratio=None, tokens_per_chunk=None):
    """Rows scheduled before selection: EPIC repair, or dynamic probe input.

    A zero ratio schedules no hit rows for every selector. Positive dynamic
    budgets expose all hit rows to the worker; it narrows the final replay
    after scoring. This function never changes the mandatory miss/query rows.
    """
    if selector == "epic":
        # Preserve the full-replay endpoint. Partial EPIC repair has no work
        # to do on the first chunk, whose independent positions already match
        # the request. Do not redistribute its budget to later chunks.
        eligible = hit_segments if ratio == 1 else [seg for seg in hit_segments if seg[0] != 0]
        return select_epic_positions(eligible, ratio=ratio, tokens_per_chunk=tokens_per_chunk)
    if selector not in {"cacheblend", "prophetkv"}:
        raise ValueError("unknown LinearKV selector")
    if (tokens_per_chunk is not None or isinstance(ratio, bool)
            or not isinstance(ratio, (float, int)) or not math.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError("dynamic LinearKV selectors require a ratio in [0, 1]")
    return select_epic_positions(hit_segments, ratio=1.0 if ratio > 0 else 0.0)


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


def select_scored_positions(hit_segments, scores, ratio):
    """Global cached-token budget, stable ties, then original context order."""
    import torch

    positions = sorted(pos for start, end, _ in hit_segments for pos in range(start, end))
    if scores.ndim != 1 or scores.numel() != len(positions):
        raise ValueError("selector scores must align with cached positions in context order")
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be in [0, 1]")
    if not torch.isfinite(scores).all():
        raise ValueError("selector produced non-finite scores")
    count = math.floor(ratio * len(positions))
    if ratio > 0 and positions:
        count = max(1, count)
    if count == 0:
        return {}
    if count == len(positions):
        return {(start, end): list(range(start, end)) for start, end, _ in sorted(hit_segments) if end > start}
    # Only the cutoff value is needed; score order is discarded by replay.
    # Resolve cutoff ties in original-position order, exactly as stable sort.
    values = scores.float()
    cutoff = torch.topk(values, count, sorted=False).values.min()
    above = values > cutoff
    tied = values == cutoff
    tie_rank = tied.cumsum(0)
    chosen_mask = above | (tied & (tie_rank <= count - above.sum()))
    indices = chosen_mask.nonzero().flatten().cpu().tolist()
    chosen = {positions[index] for index in indices}
    selected = {}
    for start, end, _ in sorted(hit_segments):
        rows = [pos for pos in range(start, end) if pos in chosen]
        if rows:
            selected[(start, end)] = rows
    return selected


def cacheblend_value_scores(current, cached):
    """CacheBlend's squared V difference, summed over local KV heads/channels."""
    if current.shape != cached.shape or current.ndim != 3:
        raise ValueError("CacheBlend expects aligned [tokens, kv_heads, head_dim] values")
    return (current.float() - cached.float()).square().sum(dim=(1, 2))


def prophet_attention_scores(query, cached_keys, query_block_size=16):
    """Frozen-K' Stage I: softmax over context, mean heads, sum query rows.

    Inputs are RoPE-corrected [tokens, heads, head_dim]. GQA is evaluated
    without materializing repeated keys. Query blocking bounds the temporary
    attention matrix while retaining the full context softmax denominator.
    """
    import torch

    if query.ndim != 3 or cached_keys.ndim != 3 or query.shape[-1] != cached_keys.shape[-1]:
        raise ValueError("ProphetKV expects query/key tensors with matching head dimensions")
    heads, kv_heads = query.shape[1], cached_keys.shape[1]
    if heads % kv_heads or not query.shape[0] or not cached_keys.shape[0]:
        raise ValueError("ProphetKV requires non-empty queries/keys and compatible GQA heads")
    keys = cached_keys.float()
    scores = keys.new_zeros(keys.shape[0])
    for block in query.split(query_block_size):
        grouped = block.float().reshape(block.shape[0], kv_heads, heads // kv_heads, -1)
        logits = torch.einsum("qkgd,ckd->kgqc", grouped, keys) / math.sqrt(query.shape[-1])
        scores += logits.softmax(dim=-1).sum(dim=(0, 1, 2)) / heads
    return scores
