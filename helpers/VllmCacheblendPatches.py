"""Portable wiring for vLLM 0.25 + LMCache 0.5.2 in-process CacheBlend.

The stock packages don't make blending work out of the box on this class of
hosts. lmcache 0.5.2 defines ``VLLMModelTracker.register_model`` but never
calls it, so ``LMCBlenderBuilder.get_or_create`` raises during connector init.
This is a wiring gap in the lmcache↔vLLM pairing — confirmed model
*independent* (both Qwen2.5 and Mistral hit the same ``ValueError``), so it is
not a Qwen-specific quirk. It is applied **from the framework** here — not from
a host-level ``sitecustomize.py`` and not by editing vLLM/lmcache — so the repo
stays portable: a lazy meta-path import hook patches ``GPUModelRunner.load_model``
to register the loaded model. The hook is installed before vLLM is imported and
must not import vLLM/torch eagerly (that would initialise CUDA in the parent
before the EngineCore is spawned and break vLLM's process model).

These are only active when the CacheBlend method turns blending on (the
``LMCACHE_ENABLE_BLENDING`` env var), so the rest of the framework's vLLM path
is untouched. See ``methods/Cacheblend.py`` for how the env var also makes the
patches self-apply inside the spawned EngineCore child.
"""

import os
from typing import List

#: Env var that gates all blending patches and the lmcache engine behaviour.
_BlendEnvKey = "LMCACHE_ENABLE_BLENDING"

#: vLLM model-runner modules whose ``GPUModelRunner.load_model`` needs patching
#: (V2 runner in 0.25, plus the V1 runner path).
_BlendPatchTargets = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
)

_PatchesApplied = False

#: Idempotence guard for :func:`_EnsureBlendLayerPatched`.
_BlendLayerPatched = False

#: Idempotence guard for :func:`_EnsureGpuConnectorPatched`.
_GpuConnectorPatched = False

#: Idempotence guard for :func:`_EnsureProcessQkvPatched`.
_ProcessQkvPatched = False

#: LMCache adapter module(s) whose ``start_load_kv`` blend branch needs the
#: partial-retrieve reconciliation (see :func:`_patch_adapter_module`).
_BlendSafetyTargets = ("lmcache.integration.vllm.vllm_v1_adapter",)

#: LMCache token-database module whose store-side segment filtering drops the
#: segment that straddles the chunk-aligned store-mask boundary (see
#: :func:`_patch_token_db_module`).
_BlendTokenDbTargets = ("lmcache.v1.token_database",)


# ---------------------------------------------------------------- env setup
def SetBlendEnv(
    *,
    recompRatio: float = 0.15,
    chunkSize: int = 256,
    maxLocalCpuSize: int = 48,
    blendCheckLayers: int = 1,
) -> None:
    """Set the LMCache env vars that turn on in-process CacheBlend.

    Must be called before the ``LLM`` is constructed (the spawned EngineCore
    inherits them). Idempotent — call once per method instance.
    """
    os.environ[_BlendEnvKey] = "True"
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = str(blendCheckLayers)
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(recompRatio)
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunkSize)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(maxLocalCpuSize)
    # The in-process engine spawns an EngineCore child; `spawn` (not `fork`)
    # keeps the CUDA runtime clean across processes.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# --------------------------------------------------------------- patch bodies
def _EnsureGpuConnectorPatched(connector) -> None:
    """Idempotently wrap ``VLLMBufferLayerwiseGPUConnector.batched_to_gpu`` to
    record the retrieve's physical buffer length.

    ``batched_to_gpu`` sizes its GPU buffer ``num_all_tokens = ends[-1] -
    starts[0]``, which spans the *whole* retrieved range *including* the
    2-token ``' # # '`` separator gaps. ``retrieve_layer``'s first yield is
    ``sum(ret_mask)`` — only the segment spans, *excluding* those gaps — so it
    is always smaller than the buffer length whenever the stream has
    separators. ``process_qkv`` compares fresh compute against the buffer
    (``old_k``), so the compute clip must be the buffer length, not the
    segment-token count; stashing it here lets ``blend_layer`` pick the right
    clip. Class-level patch, applied once per EngineCore child.
    """
    global _GpuConnectorPatched
    if _GpuConnectorPatched:
        return
    cls = type(connector)
    if not hasattr(cls, "batched_to_gpu"):
        return
    _orig = cls.batched_to_gpu

    def _kvbench_batched_to_gpu(self, starts, ends, **kwargs):
        if len(starts):
            orig_start = int(starts[0])
            orig_end = int(ends[-1])
            self._kvbench_num_all_tokens = orig_end - orig_start
            # If a blend wants more tokens than the cache physically delivered,
            # extend the buffer + paged-transfer range to the full blend claim so
            # the blend can recompute the missing tail in place (prefix blend +
            # tail recompute) instead of leaving it unpopulated. ``blend_layer``
            # stashes ``_kvbench_blend_input`` before the retriever is created;
            # the extra (orig_end, blend_input) "segment" has no memory object,
            # so the CPU fill loop (zipped over the real objects) skips it and
            # the tail stays zeroed for ``process_qkv`` to overwrite.
            blend_input = getattr(self, "_kvbench_blend_input", None)
            if blend_input is not None and orig_end < blend_input:
                starts = list(starts)
                ends = list(ends)
                starts.append(orig_end)
                ends.append(blend_input)
        gen = _orig(self, starts, ends, **kwargs)
        yield from gen

    cls.batched_to_gpu = _kvbench_batched_to_gpu
    _GpuConnectorPatched = True


def _EnsureBlendLayerPatched() -> None:
    """Idempotently patch ``LMCBlender.blend_layer`` for partial-retrieve safety.

    The blend path feeds the lookup-claimed token count to the blender, but the
    retrieve inside can physically under-deliver (partial store / eviction): the
    retrieve breaks at the first missing segment, so its KV buffer is shorter
    than the blend input and ``process_qkv`` crashes on a ``k``/``old_k`` shape
    mismatch. The patch probes the first retrieve ``next`` (which yields the
    number of tokens actually retrieved), stashes the clip on the blender, and
    always computes the *full* claimed range: ``_kvbench_batched_to_gpu`` extends
    the partial retrieve's buffer/paged-transfer to the full claim and
    ``_kvbench_process_qkv`` recomputes the missing tail in place (prefix blend +
    tail recompute). The adapter skips its failed-block reconcile for this path
    because the scheduler recovery fires too late for a fresh chunked request.
    In the normal fully-retrieved case the clip is a no-op.
    """
    global _BlendLayerPatched
    if _BlendLayerPatched:
        return
    try:
        from lmcache.v1.compute.blend.blender import LMCBlender
    except Exception:  # noqa: BLE001
        return  # no blender yet; retried lazily on the next start_load_kv call
    _orig = LMCBlender.blend_layer
    if getattr(_orig, "_kvbench_patched", False):
        _BlendLayerPatched = True
        return

    def _kvbench_blend_layer(self, tokens, mask=None, **kwargs):
        # Retrieve first so we know what the cache physically delivered before
        # deciding how much fresh compute to run. ``compute_layer`` is a lazy
        # generator, so building it after the probe is safe.
        _EnsureGpuConnectorPatched(self.gpu_connector)
        _EnsureProcessQkvPatched()
        full_len = len(tokens)
        # Stash the blend claim *before* the retriever is created so the
        # connector's ``batched_to_gpu`` can extend a partial retrieve to the
        # full claim (prefix blend + tail recompute).
        self.gpu_connector._kvbench_blend_input = full_len
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)
        retrieved = next(layerwise_retriever)  # None if nothing found, else sum(ret_mask)

        if retrieved is None:
            clip = 0
        else:
            # Clip to the gpu_connector's *buffer* length, not ``sum(ret_mask)``:
            # the buffer spans the whole retrieved range including the 2-token
            # ' # # ' separator gaps, and ``process_qkv`` compares fresh compute
            # against that buffer. Clipping to the segment-token count would
            # leave the compute shorter than ``old_k`` and crash on a shape
            # mismatch (``_kvbench_batched_to_gpu`` stashes the length).
            num_all = getattr(self.gpu_connector, "_kvbench_num_all_tokens", None)
            if num_all is None:
                num_all = int(retrieved)  # fallback: no gap info
            clip = num_all if num_all < full_len else full_len
        self._kvbench_blend_clip = clip
        # True when the retrieve under-delivered and the fix extends the blend
        # to the full claim in place. The adapter uses this to skip the
        # failed-block reconcile (which fires too late for a fresh chunked
        # request and poisons the output) because the paged cache now carries
        # correct KV for the whole claim.
        self._kvbench_blend_partial = 0 < clip < full_len

        if clip <= 0:
            # Nothing physically retrievable: drain the retriever, clean the
            # per-request blend metadata, and yield the normal step count so
            # ``blend``'s driver loop does not hit StopIteration. The adapter
            # sees clip == 0 and marks the whole range failed -> vLLM recomputes.
            try:
                for _i in range(self.num_layers + 1):
                    next(layerwise_retriever)
            except Exception:  # noqa: BLE001
                pass
            self.metadata.clean()
            self.gpu_connector._kvbench_blend_input = None
            for _i in range(self.num_layers + 2):
                yield
            return

        # Partial or full: always compute the *full* claim range. On a partial
        # retrieve the connector already extended the buffer/transfer to
        # ``full_len`` and ``_kvbench_process_qkv`` recomputes the tail in place.

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        yield

        for i in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        next(layerwise_retriever)
        self.metadata.clean()
        # The retrieve's ``batched_to_gpu`` has already read the stash; clear it
        # so a later non-blend retrieve in the same step cannot be extended.
        self.gpu_connector._kvbench_blend_input = None
        yield

    LMCBlender.blend_layer = _kvbench_blend_layer
    LMCBlender.blend_layer._kvbench_patched = True
    _BlendLayerPatched = True


def _EnsureProcessQkvPatched() -> None:
    """Idempotently patch ``LMCBlender.process_qkv`` for the partial-retrieve
    tail recompute.

    When the retrieve under-delivers (``_kvbench_blend_partial``), the blend
    runs over the *full* claim range ``[0, L)`` instead of the clipped prefix.
    The connector has already extended the retrieved GPU buffer to length ``L``
    (retrieved prefix ``[0, C)`` + zeroed tail ``[C, L)``). This patch:
      * writes the freshly computed tail ``[C, L)`` and the prefix's top-k
        diff positions into the buffer in place, and
      * returns the *full* query batch with the *unmodified* attention metadata,
        so ``forward_contiguous`` runs full causal attention: every query at
        its true token index attends to the blended prefix ``[0, C)`` from the
        buffer plus the freshly computed ``[C, q]`` tail — exactly "prefix
        blend + tail recompute". The stock subset-attention (trimmed queries +
        ``update_from_top_indices``) would attend tail queries to ``[0, q]``
        batch positions, dropping the blended prefix, so it cannot be reused.
    """
    global _ProcessQkvPatched
    if _ProcessQkvPatched:
        return
    try:
        from lmcache.v1.compute.blend.blender import LMCBlender

        import torch  # noqa: PLC0415  (lazy: runs in the EngineCore child)
    except Exception:  # noqa: BLE001
        return  # no blender yet; retried lazily on the next start_load_kv call
    _orig = LMCBlender.process_qkv
    if getattr(_orig, "_kvbench_patched", False):
        _ProcessQkvPatched = True
        return

    def _kvbench_process_qkv(
        self, q, k, v, residual, layer_id, attn_output, attn_metadata
    ):
        # ``_kvbench_blend_input`` lives on the *connector* (stashed by
        # ``_kvbench_blend_layer`` before the retriever is created); the blender
        # itself never carries it.
        L = getattr(self.gpu_connector, "_kvbench_blend_input", None)
        C = getattr(self, "_kvbench_blend_clip", None)
        partial = getattr(self, "_kvbench_blend_partial", False)

        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if partial and L is not None and C is not None:
            # ---- partial retrieve: prefix blend + tail recompute ----
            imp = self.metadata.imp_indices
            if imp is None:
                # First layer through the partial blend: choose the recompute
                # set = the top-k most-changed *retrieved* prefix positions
                # (cacheblend style) plus the entire under-delivered tail
                # [C, L). The diff is measured only against the prefix because
                # the tail of ``old_k`` is the zeroed extension region, not real
                # KV. Selection happens at the first layer (0) so the zeroed
                # tail is overwritten into the connector buffer before it is
                # pipelined into the paged cache.
                ratio = self.common_metadata.recomp_ratios[0] if (
                    self.common_metadata.recomp_ratios is not None
                ) else 0.15
                topk_num = max(int(C * ratio), 1)
                topk_num = min(topk_num, C)
                diff_k = torch.sum(
                    (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
                )
                top_indices = torch.topk(diff_k[:C], k=topk_num).indices
                imp = torch.unique(
                    torch.cat([top_indices, torch.arange(C, L, device=q.device)])
                )
                imp, _ = torch.sort(imp)
                self.metadata.imp_indices = imp
        elif layer_id in self.common_metadata.check_layers:
            # ---- full retrieve: V-diff check layer (hybrid) ----
            # CacheBlend's drift signal is measured on *V* (not K): V is not
            # RoPE'd, so a V mismatch between the retrieved KV and the fresh
            # compute means the token's *content* context changed (deep-layer
            # KV drifted), independent of position re-rotation. ``diff_v`` at
            # the hardcoded check layer (1) selects exactly the positions whose
            # cached KV is stale and must be recomputed; the rest keeps the
            # retrieved KV.
            total_len = q.shape[0]
            ratio = self.common_metadata.recomp_ratios[0] if (
                self.common_metadata.recomp_ratios is not None
            ) else 0.15
            topk_num = max(int(total_len * ratio), 1)
            topk_num = min(topk_num, total_len)
            diff_v = torch.sum(
                (v.to(torch.float32) - old_v.to(torch.float32)) ** 2, dim=[1]
            )
            top_indices = torch.topk(diff_v, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)
            # The retrieved buffer zeroes the ' # # ' separator-gap positions
            # after RoPE, so their old KV is not real; always recompute them.
            gaps = getattr(self.gpu_connector, "current_gap_positions", None)
            if gaps is not None and gaps.numel() > 0:
                top_indices = torch.unique(
                    torch.cat([
                        top_indices,
                        gaps.to(device=top_indices.device, dtype=top_indices.dtype),
                    ])
                )
                top_indices, _ = torch.sort(top_indices)
            self.metadata.imp_indices = top_indices
            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[: top_indices.shape[0]]
            attn_metadata.update_from_top_indices(top_indices)

        if self.metadata.imp_indices is not None:
            # Fresh compute for the important positions, in place. The connector
            # transfers this buffer to the paged cache two blend steps after
            # this layer, so the blended KV reaches vLLM.
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        return q, k, v, residual, attn_output, attn_metadata

    LMCBlender.process_qkv = _kvbench_process_qkv
    LMCBlender.process_qkv._kvbench_patched = True
    _ProcessQkvPatched = True
    _InstallBlendDebug()  # child-side, after the real patch is in place


def _patch_adapter_module(module) -> None:
    """Patch ``LMCacheConnectorV1Impl.start_load_kv``'s blend branch with the
    failed-block reconciliation the non-blend retrieve path already has.

    The stock blend path calls ``blender.blend`` with the lookup-claimed token
    count and never checks what the retrieve physically delivered, so a partial
    store/eviction leaves vLLM thinking the whole blend range is loaded when only
    the clipped prefix was. This mirrors the non-blend path: after the blend, the
    clip stashed by ``_EnsureBlendLayerPatched`` is compared against the claim
    and the missing tail is recorded via ``record_failed_blocks`` ->
    ``_invalid_block_ids`` -> ``get_block_ids_with_load_errors``, so vLLM
    recomputes exactly the missing blocks and the retrieved prefix stays reused.
    """
    connector = getattr(module, "LMCacheConnectorV1Impl", None)
    if connector is None or getattr(connector, "_kvbench_blend_safe_patched", False):
        return
    import torch  # noqa: PLC0415  (lazy: this runs in the child, not the parent)

    logger = module.logger
    ConnectorMetadata = module.LMCacheConnectorMetadata

    def _kvbench_start_load_kv(self, forward_context, **kwargs):
        _EnsureBlendLayerPatched()

        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, ConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        # LMCache failed to initialize and is running in degraded mode; skip the
        # KV load so vLLM falls back to recompute instead of crashing EngineCore.
        if self.lmcache_engine is None:
            return

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                sync = idx == last_idx
                if self.enable_blending:
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                    # kvbench: reconcile the blend's physical retrieval against
                    # the lookup claim (parity with the non-blend path below).
                    # When the partial-retrieve fix extended the blend to the
                    # full claim (``_kvbench_blend_partial``), the paged cache
                    # already carries correct KV for the whole range, so marking
                    # blocks failed here would only trigger the late scheduler
                    # recovery and poison the output — skip it.
                    clip = getattr(self.blender, "_kvbench_blend_clip", None)
                    covered = getattr(self.blender, "_kvbench_blend_partial", False)
                    if (
                        clip is not None
                        and clip < lmcache_cached_tokens
                        and not covered
                    ):
                        ret_mask = token_mask[:lmcache_cached_tokens].clone()
                        ret_mask[clip:] = False
                        missing_blocks = self.record_failed_blocks(
                            request.req_id,
                            token_mask[:lmcache_cached_tokens],
                            ret_mask,
                            slot_mapping[:lmcache_cached_tokens],
                        )
                        self._invalid_block_ids.update(missing_blocks)
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    # Report failed block IDs in case of partial failure.
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

    connector.start_load_kv = _kvbench_start_load_kv
    connector._kvbench_blend_safe_patched = True


def _patch_token_db_module(module) -> None:
    """Keep the segment that straddles the store-mask boundary.

    ``SegmentTokenDatabase.process_tokens`` (used because ``enable_blending``)
    yields a segment only when ``start_idx >= num_falses`` — the number of
    leading tokens the incremental store already marked as saved. Under a
    concurrent batch the vLLM scheduler chunks each long prefill, so the
    completion store's ``skip_leading_tokens`` (256-aligned) boundary falls
    *inside* one of this framework's essay-scale sep-delimited segments, and
    that whole segment — a full RULER essay, up to ~1700 tokens — is dropped
    from the CPU cache. The reordered run then misses its key and the
    contiguous-prefix claim collapses: this is the exact cause of the
    ~0.4-average low-reuse half of the shuffle tasks (reproduced to the token
    on all partial-store cases).

    The fix yields the boundary-straddling segment in full
    (``start < num_falses < end``) ahead of the stock generator, so the
    completion store persists the entire essay. ``lookup`` passes no mask
    (``num_falses == 0``) and is untouched; fully-False segments are still
    dropped (they were stored by the earlier chunk).
    """
    cls = getattr(module, "SegmentTokenDatabase", None)
    if cls is None or getattr(cls, "_kvbench_seg_patched", False):
        return
    import torch as _torch  # noqa: PLC0415  (lazy: runs in the child, not parent)

    _orig_split = cls._fast_split_by_subtensor

    def _kvbench_fast_split(self, tokens):
        # Stock lmcache yields ``tokens`` and then FALLS THROUGH into the
        # ``unfold`` when the stream is shorter than the 2-token sep, so any
        # 1-token store slice (a decode step persisting its new token) crashes
        # with ``maximum size for tensor at dimension 0 is 1 but size is 2``.
        if self.sep_len == 0 or len(tokens) < self.sep_len:
            yield tokens
            return
        yield from _orig_split(self, tokens)

    cls._fast_split_by_subtensor = _kvbench_fast_split

    _orig = cls.process_tokens

    def _kvbench_process_tokens(
        self,
        tokens=None,
        hashes=None,
        offsets=None,
        mask=None,
        make_key=True,
        request_configs=None,
    ):
        num_falses = 0
        if tokens is not None and mask is not None:
            num_falses = mask.numel() - mask.long().sum().item()
        if tokens is not None and num_falses > 0:
            if not isinstance(tokens, _torch.Tensor):
                tok = _torch.tensor(tokens, dtype=_torch.long, device="cpu")
            else:
                tok = tokens.to(device="cpu", dtype=_torch.long)
            start_idx = 0
            for idx, chunk in enumerate(self._fast_split_by_subtensor(tok)):
                end_idx = start_idx + len(chunk)
                if idx > 0:
                    start_idx += self.sep_len
                    end_idx += self.sep_len
                if start_idx < num_falses < end_idx:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(
                                self._hash_tokens(chunk), request_configs
                            ),
                        )
                    else:
                        yield start_idx, end_idx, self._hash_tokens(chunk)
                start_idx = end_idx
        for start, end, key in _orig(
            self,
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            mask=mask,
            make_key=make_key,
            request_configs=request_configs,
        ):
            # Guard against stock 0-length segments. ``_fast_split_by_subtensor``
            # yields an empty leading chunk when a chunked-prefill store slice
            # begins exactly at a sep boundary; ``store_layer`` then asks the CPU
            # allocator for ``get_shape(0)`` bytes and lmcache's
            # ``batched_allocate`` divides by a zero aligned size (ZeroDivisionError).
            # An empty segment stores no tokens, so dropping it is lossless.
            if end > start:
                yield start, end, key

    cls.process_tokens = _kvbench_process_tokens
    cls._kvbench_seg_patched = True


def _InstallRegisterHook() -> None:
    """Lazily patch ``GPUModelRunner.load_model`` to register the loaded model.

    Installs a meta-path finder that wraps the model-runner module's loader and
    patches the class once the module is actually imported. This must not import
    vllm/torch eagerly here (see the module docstring).
    """
    try:
        import importlib.util
        import sys

        def _patch_runner_module(module):
            runner = getattr(module, "GPUModelRunner", None)
            if runner is None or getattr(runner, "_kvbench_blend_patched", False):
                return
            from lmcache.v1.compute.models.utils import VLLMModelTracker
            from lmcache.integration.vllm.utils import ENGINE_NAME

            _orig_load_model = runner.load_model

            def _kvbench_load_model(self, *a, **kw):
                _orig_load_model(self, *a, **kw)
                try:
                    VLLMModelTracker.register_model(ENGINE_NAME, self.model)
                except Exception:  # noqa: BLE001
                    pass

            runner.load_model = _kvbench_load_model
            runner._kvbench_blend_patched = True

        _BlendPatchDispatch = {
            name: _patch_runner_module for name in _BlendPatchTargets
        }
        _BlendPatchDispatch.update(
            {name: _patch_adapter_module for name in _BlendSafetyTargets}
        )
        _BlendPatchDispatch.update(
            {name: _patch_token_db_module for name in _BlendTokenDbTargets}
        )

        class _BlendPatchLoader:
            def __init__(self, orig_loader, patch_fn):
                self._orig_loader = orig_loader
                self._patch_fn = patch_fn

            def create_module(self, spec):
                create = getattr(self._orig_loader, "create_module", None)
                return create(spec) if create is not None else None

            def exec_module(self, module):
                self._orig_loader.exec_module(module)
                self._patch_fn(module)

        class _BlendPatchFinder:
            def find_spec(self, fullname, path=None, target=None):
                patch_fn = _BlendPatchDispatch.get(fullname)
                if patch_fn is None:
                    return None
                try:
                    sys.meta_path.remove(self)
                    real = importlib.util.find_spec(fullname)
                finally:
                    sys.meta_path.insert(0, self)
                if real is None or real.loader is None:
                    return None
                loader = _BlendPatchLoader(real.loader, patch_fn)
                if hasattr(real.loader, "path"):
                    loader.path = real.loader.path
                spec = importlib.util.spec_from_loader(
                    fullname, loader, origin=real.origin
                )
                spec.submodule_search_locations = real.submodule_search_locations
                return spec

        if not any(isinstance(f, _BlendPatchFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _BlendPatchFinder())
    except Exception:  # noqa: BLE001
        pass  # hook not installed -> CacheBlend init will fail loudly later


def _InstallBlendDebug() -> None:
    """KVBE_BLEND_DEBUG=1: log per-layer blend geometry (q/k shapes, imp count,
    timing) from inside the EngineCore child. Diagnostic only; no-op unless the
    env var is set. Must be called from a child-side lazy patch (not from
    ``ApplyPatches``) so lmcache/torch are never imported in the parent."""
    if os.environ.get("KVBE_BLEND_DEBUG") != "1":
        return
    try:
        import time  # noqa: PLC0415  (lazy: child only)

        from lmcache.v1.compute.blend.blender import LMCBlender
        from lmcache.v1.compute.attention.flash_attn import LMCFlashAttnBackend
    except Exception:  # noqa: BLE001
        return

    _t0 = [None]

    _orig_process_qkv = LMCBlender.process_qkv
    if getattr(_orig_process_qkv, "_kvbench_dbg", False):
        return

    def _dbg_process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
        _t0[0] = time.perf_counter()
        out = _orig_process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata)
        dt = (time.perf_counter() - _t0[0]) * 1000.0
        imp = getattr(self.metadata, "imp_indices", None)
        n_imp = None if imp is None else int(imp.numel())
        print(
            f"[blend-debug] layer={layer_id} dt={dt:7.1f}ms "
            f"q_in={q.shape[0]} k_in={k.shape[0]} v_in={v.shape[0]} "
            f"q_out={out[0].shape[0]} k_out={out[1].shape[0]} imp={n_imp} "
            f"clip={getattr(self, '_kvbench_blend_clip', None)} "
            f"partial={getattr(self, '_kvbench_blend_partial', False)}",
            flush=True,
        )
        return out

    LMCBlender.process_qkv = _dbg_process_qkv
    LMCBlender.process_qkv._kvbench_dbg = True

    _orig_forward = LMCFlashAttnBackend.forward_contiguous

    def _dbg_forward_contiguous(self, query, key, value, output, attn_metadata, **kw):
        dt = (time.perf_counter() - _t0[0]) * 1000.0 if _t0[0] else -1.0
        print(
            f"[blend-debug]   attn q={query.shape} k={key.shape} v={value.shape} "
            f"dt={dt:7.1f}ms",
            flush=True,
        )
        return _orig_forward(self, query, key, value, output, attn_metadata, **kw)

    LMCFlashAttnBackend.forward_contiguous = _dbg_forward_contiguous
    LMCFlashAttnBackend.forward_contiguous._kvbench_dbg = True


def ApplyPatches() -> None:
    """Apply the lmcache/vllm wiring patch the framework needs. Idempotent."""
    global _PatchesApplied
    if _PatchesApplied:
        return
    _InstallRegisterHook()
    _PatchesApplied = True


# ------------------------------------------------------------------ LLM build
def CreateBlendLlm(
    modelPath: str,
    gpuIds: str = "0",
    *,
    gpuMemoryUtilization: float = 0.7,
    maxModelLen: int = 40960,
    dtype: str = "bfloat16",
    enforceEager: bool = True,
    tensorParallelSize: int = 1,
):
    """Build the vLLM ``LLM`` wired for in-process LMCache CacheBlend.

    ``kv_role="kv_both"`` + ``LMCacheConnectorV1`` makes the engine store KV in
    (and retrieve from) the CPU cache. Assumes :func:`SetBlendEnv` +
    :func:`ApplyPatches` already ran (they must precede vLLM import).

    Tensor parallelism: ``tensor_parallel_size`` (default 1, the single-GPU
    path). With ``tensor_parallel_size>1``, ``gpuIds`` must name that many
    devices (e.g. ``"0,1"``), each TP rank gets its own vLLM worker process and
    its own LMCache engine, and the blend patches run per rank over that rank's
    KV shard.
    """
    from helpers.VllmHelper import SanitizedModelDir, SetCudaVisibleDevices

    SetCudaVisibleDevices(gpuIds)
    from vllm import LLM
    from vllm.config import KVTransferConfig

    return LLM(
        model=str(SanitizedModelDir(modelPath)),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            # A partial retrieve (blend clips to what the cache physically
            # delivered) marks the missing tail as failed blocks; the default
            # ``fail`` policy would abort the whole request instead of letting
            # vLLM recompute exactly that tail. ``recompute`` truncates the
            # request's computed tokens at the first failed block and recomputes
            # the rest, keeping the loaded prefix reused.
            kv_load_failure_policy="recompute",
        ),
        dtype=dtype,
        gpu_memory_utilization=gpuMemoryUtilization,
        max_model_len=maxModelLen,
        tensor_parallel_size=tensorParallelSize,
        enable_prefix_caching=False,
        enforce_eager=enforceEager,
    )


def Warmup(llm) -> None:
    """Issue a throwaway first request so lmcache's first (corrupt) store is
    discarded.

    On this class of hosts, the *first* store request in a fresh EngineCore
    writes corrupt KV to the CPU cache (nondeterministic — surfaces as EOS or
    wrong content when that KV is later reused). Every subsequent store is
    clean. Issuing one small throwaway generation here, before any real
    Prepare/Run work, makes the real stores always clean.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    # Long enough to survive lmcache's segment splitter (which crashes on streams
    # shorter than the 2-token " # # " separator via ``tensor.unfold``).
    text = (
        "The quick brown fox jumps over the lazy dog while the birds sing "
        "near the quiet river."
    )
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    llm.generate(
        prompts=ids, sampling_params=SamplingParams(temperature=0, max_tokens=1)
    )


# ------------------------------------------------------------- token building
def EncodeText(tokenizer, text: str) -> List[int]:
    """Encode ``text`` without special tokens (the framework builds chat
    prompts from literal text, so the ``<|im_start|>`` etc. live in the string)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # transformers >= 5 dropped add_special_tokens from encode
        return tokenizer(text, add_special_tokens=False)["input_ids"]


def SepTokens(tokenizer) -> List[int]:
    """The literal ``' # # '`` separator ids LMCache's SegmentTokenDatabase
    splits token streams on.

    Uses exactly the library's own convention — ``encode(' # # ')[1:]`` (the
    leading slot is the BOS/special token ``encode`` prepends) — so the segment
    boundaries we build match the ones lmcache looks up, giving equal cache keys.
    """
    try:
        return tokenizer.encode(" # # ")[1:]
    except TypeError:
        return tokenizer(" # # ", add_special_tokens=False)["input_ids"][1:]


def BuildContextTokens(chunks: List[str], tokenizer, sep: List[int]) -> List[int]:
    """Assemble the store/run context at the token level: each chunk encoded
    separately, joined by the literal sep ids.

    Must be built this way (not ``tokenizer.encode("".join(chunks))``) so the
    stream contains the exact ``[sep]`` boundaries LMCache splits segments on,
    and so a reordered/partially-reused run produces the *same* per-chunk
    segment keys (content-hashed) as the stored context.
    """
    ids: List[int] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            ids += sep
        ids += EncodeText(tokenizer, chunk)
    return ids
