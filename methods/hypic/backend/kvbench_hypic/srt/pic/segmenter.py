"""Split a prompt on PIC_SEPARATOR_STR, tokenize each segment, return ids +
per-segment (start,end). Also exposes `segment_hash` (formerly pic/hasher.py).
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple, Union

import torch

TokenIdsLike = Union[Iterable[int], torch.Tensor]


def split_and_tokenize(
    text: str,
    tokenizer,
    separator: str = "<<PIC_SEP>>",
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Returns:
        (concatenated_token_ids, [(start, end)] per non-empty segment).
        Segment endpoints index into the concatenated token list.
    """
    parts = [p for p in text.split(separator)]
    ids: List[int] = []
    offsets: List[Tuple[int, int]] = []
    for part in parts:
        seg_ids = tokenizer.encode(part, add_special_tokens=False)
        if not seg_ids:
            continue
        start = len(ids)
        ids.extend(seg_ids)
        offsets.append((start, len(ids)))
    return ids, offsets


def segment_hash(token_ids: TokenIdsLike) -> bytes:
    """sha256(token_ids_as_int32_little_endian)[:16].

    Hash collisions are extremely unlikely with 128 bits; PICache falls back to
    full token_ids equality on the rare collision path (see design §1 decision 11).
    """
    if isinstance(token_ids, torch.Tensor):
        arr = token_ids.detach().to(torch.int32).cpu().contiguous().numpy().tobytes()
    else:
        import array

        arr = array.array("i", list(token_ids)).tobytes()
    return hashlib.sha256(arr).digest()[:16]


def validate_segments(input_ids, segments):
    """Check explicit half-open boundaries without decoding or retokenizing."""
    if not isinstance(input_ids, list) or not input_ids or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in input_ids
    ):
        raise ValueError("pic_segments requires a single non-empty token-ID list")
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("pic_segments must partition the input token IDs")
    cursor = 0
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise ValueError("pic_segments must contain (start, end) pairs")
        start, end = segment
        if any(isinstance(x, bool) or not isinstance(x, int) for x in segment):
            raise ValueError("PIC segment offsets must be integers")
        if start != cursor or end <= start or end > len(input_ids):
            raise ValueError("pic_segments must form contiguous non-empty ranges")
        cursor = end
    if cursor != len(input_ids):
        raise ValueError("pic_segments must cover every input token")
