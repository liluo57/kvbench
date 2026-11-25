"""LinearKV's dynamic selector probes, followed by ordinary selective replay.

CacheBlend probes through a configurable FA layer and ranks squared V drift.
ProphetKV runs fresh/query tokens against the assembled document cache, scoring
terminal-query attention to frozen keys at every FA layer. Probe state is
request-private and restored before replay;
cached local recurrent states and public KV are never published by a probe.
This two-pass integration prioritizes selection/quality experiments. It does
not implement CacheBlend's fused layer-wise pruning latency optimization.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from .selector import (
    cacheblend_value_scores,
    merge_online_positions,
    prophet_attention_scores,
    select_scored_positions,
)


class SelectorProbeComplete(Exception):
    """Stop the model before the probe's final FA attention/MLP/logit work."""


class SelectorProbe:
    def __init__(self, selector, rope_meta, hit_segments, layers, selector_layer=None, query_span=None):
        self.selector = selector
        self.query_span = query_span
        self.rope_meta = rope_meta
        self.positions = sorted(p for start, end, _ in hit_segments for p in range(start, end))
        self.layers = sorted(layers)
        if not self.layers:
            raise ValueError("LinearKV dynamic selectors require full-attention layers")
        self.target_layer = self.layers[0] if selector_layer is None else selector_layer
        if self.target_layer not in self.layers:
            raise ValueError("selectorLayer must identify a full-attention layer")
        self.scores = None
        self.observed_layers = []
        self.complete = False
        self._prepared_kv_layers = set()
        self._slots = None
        self._query_rows = None

    def observe(self, backend, layer, forward_batch, q, v):
        layer_id = layer.layer_id
        if layer_id not in self.layers:
            return
        if self.selector == "cacheblend" and layer_id != self.target_layer:
            return
        hits = sorted(self.rope_meta[0]["local_hit"], key=lambda item: item[0])
        pool = backend.token_to_kv_pool
        if self.selector == "cacheblend":
            rows = torch.tensor(self.positions, device=v.device, dtype=torch.long)
            # The CacheBlend probe contains the complete original prompt.
            current = v.reshape(v.shape[0], layer.tp_k_head_num, layer.head_dim)[rows]
            slots = torch.cat([public for _, _, _, public in hits]).to(v.device).long()
            cached = pool.get_value_buffer(layer_id)[slots]
            scores = cacheblend_value_scores(current, cached)
        else:
            # Public keys retain each chunk's local RoPE. Reuse the same
            # relocation as serving, writing only request-private copies.
            if backend._pic_rope_cos_sin_cache is None:
                backend._init_pic_rope_cache()
            backend._pic_rope_phase_b(layer, forward_batch, self.rope_meta)
            if self._slots is None:
                self._slots = torch.cat([private for _, _, private, _ in hits]).to(q.device).long()
            slots = self._slots
            keys = pool.get_key_buffer(layer_id)[slots]
            # Fresh spans before the terminal query must also run so Query
            # can read their KV. Only terminal-query rows contribute scores;
            # TP padding and earlier fresh rows must not enter the reduction.
            count = forward_batch.extend_seq_lens_cpu[0]
            query = q[:count].reshape(count, layer.tp_q_head_num, layer.head_dim)
            start, end = self.query_span
            if self._query_rows is None:
                positions = forward_batch.positions[:count]
                self._query_rows = ((positions >= start) & (positions < end)).nonzero().flatten()
            query = query.index_select(0, self._query_rows)
            scores = prophet_attention_scores(query, keys)
        self.scores = scores if self.scores is None else self.scores + scores
        self.observed_layers.append(layer_id)
        if self.selector == "cacheblend" or layer_id == self.layers[-1]:
            self.scores /= len(self.observed_layers)
            self.complete = True
            raise SelectorProbeComplete()


def _packed_subset(source, positions):
    """Create fresh physical-token metadata while preserving logical KV length."""
    result = copy.copy(source)
    positions = torch.as_tensor(positions, dtype=torch.long, device=source.positions.device)
    rows = torch.searchsorted(source.positions, positions)
    if rows.numel() and (
        int(rows.max()) >= source.positions.numel()
        or not torch.equal(source.positions[rows], positions)
    ):
        raise RuntimeError("LinearKV selector requested a token absent from its initial stream")
    for field in ("input_ids", "out_cache_loc"):
        setattr(result, field, getattr(source, field)[rows])
    result.positions = positions
    if source.mrope_positions is not None:
        result.mrope_positions = source.mrope_positions[..., rows]
    count = positions.numel()
    result.extend_num_tokens = count
    result.extend_seq_lens = torch.tensor([count], dtype=torch.int32, device=positions.device)
    result.extend_seq_lens_cpu = [count]
    result.extend_start_loc = torch.zeros(1, dtype=torch.int32, device=positions.device)
    result.num_token_non_padded_cpu = count
    if source.num_token_non_padded is not None:
        result.num_token_non_padded = torch.full_like(source.num_token_non_padded, count)
    # Attention plans must be rebuilt for each physical stream.
    result.forward_metadata_ready = False
    result.forward_metadata_planned_bs = None
    result.forward_metadata_planned_num_tokens = None
    return result


@contextmanager
def _probe_state(runner, original, probe_batch):
    backend = runner.attn_backend
    pool = backend.req_to_token_pool
    # Apply deferred scheduler ownership operations before snapshotting. They
    # must not run again after restoring the probe's request-private state.
    backend.linear_attn_backend._execute_deferred_mamba_cow_and_clear(original)
    for field in ("mamba_clear_indices", "mamba_cow_src_indices", "mamba_cow_dst_indices"):
        setattr(probe_batch, field, None)
    indices = pool.get_mamba_indices(original.req_pool_indices).long()
    cache = pool.mamba_pool.mamba_cache
    # Reuse request-private snapshots. index_select already copies: a second
    # clone only allocated/copied the same state again on every probe.
    workspace = getattr(backend, '_pic_selector_snapshot', None)
    if workspace is None:
        workspace = backend._pic_selector_snapshot = {}
    def snapshot(name, state):
        shape = (state.shape[0], indices.numel(), *state.shape[2:])
        saved = workspace.get(name)
        if saved is None or saved.shape != shape or saved.dtype != state.dtype or saved.device != state.device:
            saved = workspace[name] = state.new_empty(shape)
        torch.index_select(state, 1, indices, out=saved)
        return saved
    temporal = snapshot('temporal', cache.temporal)
    conv = [snapshot(f'conv-{i}', state) for i, state in enumerate(cache.conv)]
    # clear_slots also touches the auxiliary tails/transition buffers. Restore
    # the entire active slot, including when the probe raises an exception.
    tail_buffers = getattr(cache, "conv_tails", None) or []
    tails = [snapshot(f'tail-{i}', state) for i, state in enumerate(tail_buffers)]
    transition_buffer = getattr(cache, "transition", None)
    transition = None if transition_buffer is None else snapshot('transition', transition_buffer)
    try:
        backend._pic_rope_phase_b_cache = None
        yield
    finally:
        cache.temporal.index_copy_(1, indices, temporal)
        for state, saved in zip(cache.conv, conv):
            state.index_copy_(1, indices, saved)
        for state, saved in zip(tail_buffers, tails):
            state.index_copy_(1, indices, saved)
        if transition is not None:
            transition_buffer.index_copy_(1, indices, transition)
        # The probe can populate planner caches using different positions.
        backend._pic_rope_phase_b_cache = None


def prepare_linearkv_selection(batch, forward_batch, runner):
    """Return the final physical stream; leave scheduler miss accounting alone."""
    args = runner.server_args
    selector = getattr(args, "pic_linearkv_selector", "epic")
    if selector == "epic":
        return forward_batch
    if forward_batch.forward_mode.is_split_prefill():
        raise ValueError("Dynamic LinearKV selectors do not support split prefill")
    if len(batch.reqs) != 1 or forward_batch.batch_size != 1:
        raise ValueError("LinearKV selector probes require one request")
    req = batch.reqs[0]
    if not req.pic_hit_segments:
        return forward_batch
    ratio = args.pic_epic_ratio
    if ratio in (0, 1):
        return forward_batch
    if forward_batch.input_embeds is not None or forward_batch.replace_embeds is not None:
        raise ValueError("LinearKV selector probes require native token IDs without embedding overrides")
    if forward_batch.spec_info is not None:
        raise ValueError("LinearKV selector probes do not support speculative decoding")
    if selector == "prophetkv" and req.pic_segments[-1] not in req.pic_miss_segments:
        raise ValueError("ProphetKV requires a fresh terminal Query segment")
    backend = runner.attn_backend
    collector = SelectorProbe(selector, forward_batch.pic_rope_meta, req.pic_hit_segments,
                              backend.full_attn_layers, args.pic_linearkv_selector_layer,
                              query_span=req.pic_segments[-1])
    if selector == "prophetkv":
        # This is a zero-repair LinearKV forward, not an isolated query.
        # Keep logical sequence length, document slot mapping, and PIC mode:
        # FA reads all causal document KV and GDN restores the paired LB
        # state/conv history. Earlier fresh spans (if any) run as well.
        probe_batch = _packed_subset(forward_batch, merge_online_positions(req.pic_miss_segments, {}))
        probe_batch.pic_linearkv_selected_positions = [{}]
        probe_batch.pic_rope_meta = [{**forward_batch.pic_rope_meta[0], "repair": {"positions": {}}}]
    else:
        probe_batch = copy.copy(forward_batch)
    for field in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
        setattr(probe_batch, field, None)
    probe_batch.pic_linearkv_independent_prefill = [False]
    probe_batch.pic_selector_probe = collector
    with _probe_state(runner, forward_batch, probe_batch):
        try:
            runner.forward(probe_batch)
        except SelectorProbeComplete:
            pass
    if not collector.complete or collector.scores is None:
        raise RuntimeError("LinearKV selector probe did not reach its full-attention scoring layers")
    from methods.hypic.backend.kvbench_hypic.srt.distributed import tensor_model_parallel_all_reduce

    # All ranks must replay exactly the same positions. Sum local V drift or
    # local-head query attention before globally ranking tokens.
    scores = tensor_model_parallel_all_reduce(collector.scores)
    selected = select_scored_positions(req.pic_hit_segments, scores, ratio)
    positions = merge_online_positions(req.pic_miss_segments, selected)
    result = _packed_subset(forward_batch, positions)
    result.pic_linearkv_selected_positions = [selected]
    result.pic_rope_meta = [{**forward_batch.pic_rope_meta[0], "repair": {"positions": selected}}]
    req.pic_linearkv_selected_positions = selected
    req.pic_online_token_positions = torch.tensor(positions, dtype=torch.int64)
    req.pic_miss_token_positions = req.pic_online_token_positions
    return result


def selection_metadata(req, selector, include_positions=False):
    matched = sum(end - start for start, end, _ in req.pic_hit_segments)
    selected = (getattr(req, 'pic_hit_seam_positions', None) if selector == 'hypic_seam'
                else getattr(req, 'pic_linearkv_selected_positions', None)) or {}
    count = sum(len(rows) for rows in selected.values())
    result = dict(selector=selector, num_recomputed_tokens=count,
                  num_matched_tokens=matched, recompute_ratio=count / matched if matched else 0.0)
    if include_positions:
        result['repair_positions'] = sorted(int(p) for rows in selected.values() for p in rows)
    return result
