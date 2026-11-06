"""On-demand native mixed GDN/attention supervision for the existing repairer."""

from __future__ import annotations

import torch

from .cache import canonicalize_target_kv, pack_target_kv, reposition_target_kv
from .target import TargetGeometry, text_backbone


@torch.no_grad()
def make_pair(target, chunks, *, geometry=None, chunk_batch_size=1, pack_chunks=False):
    """Independent chunk forwards and a genuine joint forward, in canonical FP32.

    Every independent call starts with no cache, including no GDN state. The
    model executes every GDN layer normally. Only attention KV leaves this
    function. Tokens are already compiled and are never re-tokenized here.
    """
    if target.training or any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError("Supervision requires a frozen target in eval mode")
    geometry = geometry or TargetGeometry.from_target(target)
    device = target.get_input_embeddings().weight.device
    rotary = text_backbone(target).rotary_emb
    if not isinstance(chunks, list) or len(chunks) < 2:
        raise ValueError("Repair supervision requires at least two nonempty chunks")
    for index, chunk in enumerate(chunks):
        if (not isinstance(chunk, list) or not chunk or any(
            isinstance(token, bool) or not isinstance(token, int)
            or not 0 <= token < geometry.vocab_size for token in chunk
        )):
            raise ValueError(f"Invalid token IDs or empty chunk at index {index}")
    if sum(map(len, chunks)) > geometry.max_position_embeddings:
        raise ValueError("Supervision prefix exceeds the target position limit")
    if not isinstance(chunk_batch_size, int) or chunk_batch_size < 1:
        raise ValueError("chunk_batch_size must be positive")
    if pack_chunks:
        from .packed import packed_stale
        stale, tokens, chunk_ids = packed_stale(target, chunks, geometry)
        positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
        output = target(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                        position_ids=positions, use_cache=True, logits_to_keep=1)
        native = pack_target_kv(output.past_key_values, geometry, device=device)
        del output
        return dict(stale=stale, joint=canonicalize_target_kv(native, positions, rotary),
                    tokens=tokens, chunk_ids=chunk_ids)
    parts, token_parts, chunk_parts = ([None] * len(chunks) for _ in range(3))
    offsets, offset = [], 0
    for chunk in chunks:
        offsets.append(offset)
        offset += len(chunk)
    order = list(range(len(chunks)))
    if chunk_batch_size > 1:
        order.sort(key=lambda index: len(chunks[index]))
    groups, group, total = [], [], 0
    for index in order:
        projected = max([len(chunks[i]) for i in group] + [len(chunks[index])]) * (len(group) + 1)
        if group and (len(group) >= chunk_batch_size or projected > 1.25 * (total + len(chunks[index]))):
            groups.append(group)
            group, total = [], 0
        group.append(index)
        total += len(chunks[index])
    if group:
        groups.append(group)
    for indices in groups:
        group = [chunks[index] for index in indices]
        length = max(map(len, group))
        tokens = torch.zeros(len(group), length, device=device, dtype=torch.long)
        mask = torch.zeros_like(tokens)
        for row, chunk in enumerate(group):
            tokens[row, :len(chunk)] = torch.tensor(chunk, device=device)
            mask[row, :len(chunk)] = 1
        positions = torch.arange(length, device=device).expand(len(group), -1)
        output = target(input_ids=tokens, attention_mask=mask,
                        position_ids=positions, use_cache=True, logits_to_keep=1)
        packed = pack_target_kv(output.past_key_values, geometry, device=device)
        del output
        for row, chunk in enumerate(group):
            index = indices[row]
            offset = offsets[index]
            native = packed[row:row + 1, :len(chunk)].contiguous()
            local_positions = positions[row:row + 1, :len(chunk)]
            if offset:
                native = reposition_target_kv(native, local_positions, local_positions + offset, rotary)
            parts[index] = canonicalize_target_kv(native, local_positions + offset, rotary)
            actual_tokens = tokens[row:row + 1, :len(chunk)]
            token_parts[index] = actual_tokens
            chunk_parts[index] = torch.full_like(actual_tokens, index)
        del packed
    tokens = torch.cat(token_parts, dim=1)
    offset = sum(map(len, chunks))
    positions = torch.arange(offset, device=device).unsqueeze(0)
    output = target(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    position_ids=positions, use_cache=True, logits_to_keep=1)
    joint = pack_target_kv(output.past_key_values, geometry, device=device)
    del output
    return dict(stale=torch.cat(parts, dim=1),
                joint=canonicalize_target_kv(joint, positions, rotary),
                tokens=tokens, chunk_ids=torch.cat(chunk_parts, dim=1))
