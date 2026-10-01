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
