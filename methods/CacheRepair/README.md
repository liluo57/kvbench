# CacheRepair for KVBench

The Qwen3.8-27B repairer body is available separately in
[`qwen38/`](qwen38/README.md). It includes runtime full-attention geometry,
partial MRoPE cache conversion, and checkpoint validation. Data preparation and
the upstream-loop training adapter are described in [`qwen38/TRAINING.md`](qwen38/TRAINING.md).
The completed 256-example training pilot is recorded in [`qwen38/STAGE_B_REPORT.md`](qwen38/STAGE_B_REPORT.md).
The hybrid target's online Method is [`HybridCacheRepair`](../HybridCacheRepair/README.md),
combining Hypic TRR recurrent history with the trained attention repairer.
The adapter below continues to serve the existing Qwen2.5 checkpoints.

This adapter runs the upstream CacheRepair inference code over KVBench's
existing PREPARE/RUN token IDs. It does not add or alter Tasks and does not
train weights.

## Target and repairer

The published repairer is target-specific. Before using this method, set
`ModelPath` in the project `config.yaml` to the same local Qwen2.5-3B-Instruct
checkout used to tokenize the Tasks and load the target model. For the download
started for this integration, that path is:

```yaml
ModelPath: /data/lyh/models/Qwen2.5-3B-Instruct
```

The `CacheRepair.RepairerPath` and `CacheRepair.RepairerSize` settings in
`config.yaml` select the checkpoint; defaults are `gwang3456/cacherepair` and
`M`, whose checkpoint is under `Qwen2.5-3B-Instruct/M`. Set the size to `S` or
`L` to select another upstream checkpoint. `RepairerPath` may also point to a
local checkpoint directory. Upstream validates target model geometry and RoPE
frequencies when it loads the checkpoint.

```python
from methods import CacheRepairMethod

method = CacheRepairMethod()
```

Use `CacheRepairMethod(fullPrefill=True)` for the full-prefill control. It
loads the same target model but skips repairer loading and all PREPARE cache
work, then generates from each complete RUN prompt.

## Adapter behavior

`Initialize` warms the target model's prefill and first-token forward on the
assigned GPU in both modes. Repair mode additionally warms every power-of-two
serving bucket from 128 through the configured context limit (bounded by the
target's position limit). Synthetic KV and two chunks exercise padding and
the fresh-chat-header plus single-document layout, followed by global RoPE,
HF cache conversion, query prefill, and first-token forward. Upstream fresh-
span prefill/canonicalization and host-to-device cache transfer are warmed
as well. CUDA is synchronized before initialization returns; this work is
outside TTFT.
Warmup does not consume sampling RNG or retain any synthetic case state.
`Reset` preserves compiled kernels across cases; `Close` clears the adapter's
bucket tracking for subsequent initialization.

`Prepare` uses upstream `compile_stale` to prefill each non-empty segment
independently and stores its canonical stale KV. `Run` uses KVBench's
`ComposeInterleavedReuse` to preserve the Task's exact token sequence. Any
uncached spans before the final query suffix are prefetched independently;
the ordered prefix is repaired through the upstream CacheRepair model, then
the unmatched query suffix is prefetched against the repaired cache. A prompt
with no matching prepared segment uses full prefill. TTFT includes online
composition, fresh-span prefill, repair, query prefill, and generation.

Repair mode supports single-turn workflows and raises on `retainOutput=True`
because generated KV is not represented by the upstream stale-cache training
format. The `fullPrefill=True` control ignores output retention, matching the
full-recompute baseline behavior. The upstream implementation and its
license/notice are retained in `backend/`; the integration code is in
`Method.py`.

## Runtime note

The upstream `backend/pyproject.toml` declares Torch `>=2.6,<2.7` and
Transformers `>=4.51.3,<4.58`. Match the CacheRepair upstream runtime when
running experiments; this adapter intentionally does not alter the shared
Python environment.
