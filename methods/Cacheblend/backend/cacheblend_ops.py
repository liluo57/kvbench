"""Shared CacheBlend operations used directly by the bundled attention models."""


def indexed_causal_mask(cfm, native_mask, *, align_single_row=True):
    """Mask sparse queries by their original positions, keeping CUTLASS strides."""
    import torch

    indices = cfm.get("imp_indices")
    key_len = int(cfm.get("org_seq_len") or 0)
    if indices is None or key_len <= 0:
        return native_mask()
    rows = indices.to(device=indices.device, dtype=torch.int64)
    padded_key_len = (key_len + 7) // 8 * 8
    keys = torch.arange(padded_key_len, device=rows.device, dtype=torch.int64)
    dtype = cfm.get("kv_cache_dtype") or torch.float32
    bias = torch.where(
        keys.unsqueeze(0) <= rows.unsqueeze(1),
        torch.zeros((), device=rows.device, dtype=dtype),
        torch.full((), torch.finfo(dtype).min, device=rows.device, dtype=dtype),
    )
    if align_single_row and rows.numel() == 1:
        bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)[..., :key_len]
    else:
        bias = bias[:, :key_len].view(1, 1, 1, rows.numel(), key_len)
    return bias.expand(
        1, cfm["_num_kv_heads"], cfm["_num_queries_per_kv"], rows.numel(), key_len
    )


def capture_kv(attention, key, value, status, metadata):
    """Capture pre-RoPE KV for collect, or retain each eager decode token."""
    collector = metadata.get("retain_collector")
    retaining = collector is not None and status == -1
    if metadata["collect"] or retaining:
        attention.hack_kv = [key.clone(), value.clone()]
        if retaining:
            collector.append(attention, attention.hack_kv)


def register_retained_output(
    chunk_ids, chunk_kv, output_ids, retained, *, expected_layers=None
):
    """Register one complete native output fragment and its matching KV rows.

    The caller requests one extra decode token so every returned output token
    has completed a forward pass and has a captured KV row. Never truncate IDs
    to fit a partial collector result: that would make the cache key diverge
    from the workflow's output fragment.
    """
    ids = [int(token) for token in output_ids]
    if not ids:
        return 0
    if retained is None:
        raise RuntimeError(
            "CacheBlend output retention did not capture KV for the output fragment"
        )
    if not retained or (
        expected_layers is not None and len(retained) != expected_layers
    ):
        raise RuntimeError(
            "CacheBlend output retention captured an incomplete layer set: "
            f"captured={len(retained)}, expected={expected_layers}"
        )

    required_rows = len(ids)
    stored = []
    for layer_index, kv in enumerate(retained):
        if not isinstance(kv, (list, tuple)) or len(kv) != 2:
            raise RuntimeError(
                f"CacheBlend output retention has invalid KV for layer {layer_index}"
            )
        key, value = kv
        key_rows = int(key.shape[0])
        value_rows = int(value.shape[0])
        if key_rows < required_rows or value_rows < required_rows:
            raise RuntimeError(
                "CacheBlend output retention captured fewer KV rows than native "
                f"output IDs for layer {layer_index}: ids={required_rows}, "
                f"key_rows={key_rows}, value_rows={value_rows}"
            )
        stored.append([
            key[:required_rows].detach(),
            value[:required_rows].detach(),
        ])

    cache_key = tuple(ids)
    chunk_ids[cache_key] = ids
    chunk_kv[cache_key] = stored
    return required_rows


class RetainedOutputCollector:
    def __init__(self, layers):
        self.indices = {id(layer.self_attn): i for i, layer in enumerate(layers)}
        self.items = [[] for _ in layers]

    def append(self, attention, captured):
        index = self.indices.get(id(attention))
        if index is not None:
            self.items[index].append(captured)

    def finish(self):
        import torch

        if any(not items for items in self.items):
            return None
        return [
            [torch.cat([kv[0] for kv in items]).detach(),
             torch.cat([kv[1] for kv in items]).detach()]
            for items in self.items
        ]
