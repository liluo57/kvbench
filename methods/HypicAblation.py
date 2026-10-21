"""Quality ablations of HYPIC's recurrent state and attention KV.

Usage (the existing method exports and benchmark list are unchanged)::

    from methods.HypicAblation import HypicAblationMethod

    HypicAblationMethod(mode="A")  # TRR GDN, full-prefill KV
    HypicAblationMethod(mode="B")  # full-prefill GDN, TRR KV
    HypicAblationMethod(mode="C")  # Addition GDN, full-prefill KV
    HypicAblationMethod(mode="D")  # TRR initializer, LinearKV/EPIC repair (20%)
    HypicAblationMethod(mode="D", recompRatio=1.0)  # EPIC repair (100%)

The descriptive mode names are ``gdn_trr_kv_full``, ``gdn_full_kv_trr``,
``gdn_addition_kv_full`` and ``gdn_trr_kv_linearkv``.
For A/B/C, the last PIC segment is the continuation suffix; earlier tokens form
the history whose caches are exchanged. Both donors use the same weights,
token IDs, absolute positions and Triton GDN backend. The suffix is computed
again after the exchange, so the first generated token uses mixed history.
Decode then proceeds normally. These are cache-source interventions: a TRR
or Addition donor's state already contains effects of the other layer family.

C preserves native Addition's query initialization, including its limitation
that only the first miss receives the sum of cached hit states. If query is a
later miss, its initial recurrent state is zero. Convolution history still
comes from the position-preceding segment. Full-prefill KV is written to new
request-private slots, leaving Addition's shared prepared KV untouched.

D keeps HYPIC preparation and captures the unchanged TRR donor's state and
conv history immediately before the final segment. That snapshot replaces
LinearKV's last-block initializer. LinearKV then replays the EPIC-selected
hit tokens and all miss tokens in context order, using its native GDN and
full-attention repair paths. Its repaired state continues into decode; it is
not replaced by the TRR donor's final state. There is no full-prefill donor.
At 100%, every hit token is selected, so GDN also processes every online token
from the TRR initializer; this is not a zero-initialized full prefill.

Only an ablation engine's scheduler process installs the instance wrappers
below. Importing this module does not import the native backend, register
plugins, patch HYPIC, initialize CUDA or change ordinary method timings.
Request-dependent donor forwards and transfers are included in this method's
timings, which are for diagnostics only, not performance comparisons.
"""
from __future__ import annotations

import math
import os
from functools import partial
from types import MethodType
from typing import Sequence
from uuid import uuid4

from core.Method import Method
from helpers.backends.ModelAdapter import _tokenizer
from helpers.backends.Tokenization import TerminalStopIdsFromTokenizer

from .hypic.Method import (
    HypicMethod,
    _BuildHypicEngineKwargs,
    _QWEN35_SYSTEM,
    _WARMUP_TAIL,
)

_TRR = "transition_rope_recompute"
_ADDITION_FULL = "gdn_addition_kv_full"
_TRR_LINEARKV = "gdn_trr_kv_linearkv"
_ONLINE_RID = "kvbench-hypic-ablation:run:"
_MODES = {
    "A": "gdn_trr_kv_full",
    "B": "gdn_full_kv_trr",
    "C": _ADDITION_FULL,
    "D": _TRR_LINEARKV,
    "gdn_trr_kv_full": "gdn_trr_kv_full",
    "gdn_full_kv_trr": "gdn_full_kv_trr",
    _ADDITION_FULL: _ADDITION_FULL,
    _TRR_LINEARKV: _TRR_LINEARKV,
}


class HypicAblationMethod(HypicMethod):
    """HYPIC preparation with an isolated mixed-history prefill path.

    Supports unquantized Qwen3.5/Qwen3.8 GDN checkpoints and ordinary tensor
    parallelism. Like HYPIC, one case owns each prepare/run/reset lifecycle.
    ``fullPrefill=True`` is rejected. A/B/D prepare with TRR; C uses Addition.
    D accepts ``recompRatio`` in [0, 1], defaulting to EPIC's 20% budget.
    """

    name = "hypic_ablation"

    def __init__(self, gpuNums=1, perfWeight=1.0, *, mode, tag=None, **kwargs):
        if not isinstance(mode, str) or mode not in _MODES:
            raise ValueError(f"unknown ablation mode={mode!r}; expected one of {sorted(_MODES)}")
        if kwargs.get("fullPrefill", False):
            raise ValueError("HypicAblationMethod requires fullPrefill=False")
        self.mode = _MODES[mode]
        if self.mode == _TRR_LINEARKV:
            ratio = kwargs.get("recompRatio", 0.2)
            if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                    or not math.isfinite(float(ratio)) or not 0 <= ratio <= 1):
                raise ValueError("recompRatio must be a finite number in [0, 1]")
            kwargs["recompRatio"] = float(ratio)
        pic_mode = "addition" if self.mode == _ADDITION_FULL else _TRR
        if kwargs.get("picMode", pic_mode) != pic_mode:
            raise ValueError(f"HypicAblationMethod requires picMode={pic_mode!r}")
        kwargs.setdefault("picMode", pic_mode)
        super().__init__(gpuNums, perfWeight, tag=self.mode if tag is None else tag, **kwargs)

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        # _PicMethod.Initialize constructs the ordinary engine. Repeat its
        # tokenizer setup here to keep that factory and its callers untouched.
        Method.Initialize(self, gpuIds)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in self.gpuIds)
        if not self.modelPath:
            raise ValueError("ModelPath is empty; set it in config.yaml")
        from .hypic.backend.kvbench_hypic.srt.utils.hf_transformers_utils import get_config

        config = get_config(self.modelPath, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        if (getattr(config, "quantization_config", None)
                or getattr(text_config, "quantization_config", None)):
            raise ValueError("HYPIC ablations require an unquantized FP16/BF16 checkpoint")
        if not any(arch.startswith("Qwen3_5") for arch in (getattr(config, "architectures", None) or [])):
            raise ValueError("HYPIC ablations currently support Qwen3.5/Qwen3.8 GDN models")
        self._tokenizer = _tokenizer(self.modelPath)
        self._terminalStopIds = TerminalStopIdsFromTokenizer(self._tokenizer, self.samplingConfig)
        self._systemIds = self._tokenizer.encode(_QWEN35_SYSTEM, add_special_tokens=False)
        self._tailIds = self._tokenizer.encode(_WARMUP_TAIL, add_special_tokens=False)
        self.engine = _CreateAblationEngine(
            self.modelPath, self.gpuIds, mode=self.mode, dtype=self.dtype,
            maxModelLen=self.maxModelLen, memFractionStatic=self.memFractionStatic,
            picMode=self.picMode, fullPrefill=False, seamTokens=self.seamTokens,
            **({"recompRatio": self.recompRatio} if self.mode == _TRR_LINEARKV else {}),
        )

    def _Warmup(self, chunk):
        previous = getattr(self.engine, "ablation_warmup", False)
        self.engine.ablation_warmup = True
        try:
            super()._Warmup(chunk)
        finally:
            self.engine.ablation_warmup = previous

    def Run(self, data, retainOutput=None, maxNewTokens=None):
        results = super().Run(data, retainOutput=retainOutput, maxNewTokens=maxNewTokens)
        if self.mode == _TRR_LINEARKV:
            for result in results:
                result.metadata.update(
                    ablation_mode=self.mode, quality_only=True,
                    gdn_state_source="trr", gdn_state_scope="repair_initializer",
                    gdn_repair_source="linearkv_selected_replay",
                    attention_kv_source="linearkv_epic", kv_repair_selector="epic",
                    kv_recomp_ratio=self.recompRatio, reuse_ratio_scope="trr_donor",
                    ablation_boundary="last_pic_segment_start",
                )
            return results
        for result in results:
            addition = self.mode == _ADDITION_FULL
            result.metadata.update(
                ablation_mode=self.mode, quality_only=True,
                gdn_state_source=("addition" if addition else
                                  "trr" if self.mode == "gdn_trr_kv_full" else "full_prefill"),
                attention_kv_source="trr" if self.mode == "gdn_full_kv_trr" else "full_prefill",
                reuse_ratio_scope="addition_donor" if addition else "trr_donor",
                ablation_boundary="last_pic_segment_start",
            )
        return results


def _CreateAblationEngine(modelPath, gpuIds, *, mode, **kwargs):
    from .hypic.backend.kvbench_hypic.srt.entrypoints.engine import Engine

    class AblationEngine(Engine):
        # The callable is picklable under multiprocessing spawn. No custom
        # ServerArgs, entry-point registration or parent-process patching.
        run_scheduler_process_func = staticmethod(
            partial(_RunTrrLinearKvScheduler, recompRatio=kwargs["recompRatio"])
            if mode == _TRR_LINEARKV else partial(_RunAblationScheduler, mode=mode)
        )
        ablation_warmup = False

        def generate(self, *args, **options):
            phase = "warmup" if self.ablation_warmup else "run"
            options["rid"] = f"kvbench-hypic-ablation:{phase}:{uuid4().hex}"
            return super().generate(*args, **options)

    options = _BuildHypicEngineKwargs(modelPath, gpuIds, **kwargs)
    # All changes are confined to the experimental engine. Synchronous eager
    # execution makes donor buffers and the active request's state unambiguous.
    options.update(disable_overlap_schedule=True, disable_cuda_graph=True,
                   linear_attn_prefill_backend="triton", linear_attn_decode_backend="triton")
    return AblationEngine(**options)


def _RunTrrLinearKvScheduler(*args, recompRatio, **kwargs):
    from .hypic.backend.kvbench_hypic.srt.managers import scheduler as module

    module.load_plugins()
    original = module.Scheduler

    class TrrLinearKvScheduler(original):
        def __init__(self, *scheduler_args, **scheduler_kwargs):
            super().__init__(*scheduler_args, **scheduler_kwargs)
            self._hypic_ablation = _TrrLinearKvPrefill(self.tp_worker, self.tree_cache, recompRatio)

    module.Scheduler = TrrLinearKvScheduler
    try:
        module.run_scheduler_process(*args, **kwargs)
    finally:
        module.Scheduler = original


def _RunAblationScheduler(*args, mode, **kwargs):
    from .hypic.backend.kvbench_hypic.srt.managers import scheduler as module

    module.load_plugins()
    original = module.Scheduler

    class AblationScheduler(original):
        def __init__(self, *scheduler_args, **scheduler_kwargs):
            super().__init__(*scheduler_args, **scheduler_kwargs)
            self._hypic_ablation = _AblationPrefill(self.tp_worker, mode, self.tree_cache)

    # This assignment exists only in this engine's dedicated scheduler child.
    # The original process entry retains its initialization, errors and cleanup.
    module.Scheduler = AblationScheduler
    try:
        module.run_scheduler_process(*args, **kwargs)
    finally:
        module.Scheduler = original


class _AblationPrefill:
    """Own the experimental worker wrappers and short-lived donor snapshots."""

    def __init__(self, worker, mode, tree_cache=None):
        from .hypic.backend.kvbench_hypic.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from .hypic.backend.kvbench_hypic.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        self.worker = worker
        self.runner = worker.model_runner
        self.mode = mode
        self.pic_mode = "addition" if mode == _ADDITION_FULL else _TRR
        self.backend = self.runner.attn_backend
        if (not isinstance(self.backend, HybridLinearAttnBackend)
                or not isinstance(self.backend.linear_attn_backend, GDNAttnBackend)
                or not self.backend.full_attn_layers):
            raise ValueError("HYPIC ablations require a hybrid GDN/full-attention backend")
        args = self.runner.server_args
        if (args.pp_size != 1 or args.dp_size != 1 or args.enable_dp_attention
                or args.attn_cp_size != 1 or args.moe_dp_size != 1
                or args.speculative_algorithm or args.enable_two_batch_overlap
                or args.enable_pdmux):
            raise ValueError("HYPIC ablations require ordinary TP without PP, DP or speculative decoding")
        self.pool = self.runner.req_to_token_pool
        self.kv = self.runner.token_to_kv_pool
        self.linear = self.backend.linear_attn_backend
        self._capture = None
        self._worker_forward = worker.forward_batch_generation
        gdn_method = ("forward_extend_pic_addition" if mode == _ADDITION_FULL
                      else "forward_extend_pic_transition_recompute")
        self._gdn_forward = getattr(self.linear, gdn_method)

        if mode == _ADDITION_FULL:
            self._install_addition_cleanup(tree_cache)

        def forward_generation(_worker, *forward_args, **forward_kwargs):
            return self._forward_generation(*forward_args, **forward_kwargs)

        def forward_gdn(_linear, layer, forward_batch, mixed_qkv, a, b, **gdn_kwargs):
            if self._capture is not None and self.mode == _ADDITION_FULL:
                seed = _CaptureAdditionSeed(self.linear, layer.layer_id, forward_batch)
            output = self._gdn_forward(layer, forward_batch, mixed_qkv, a, b, **gdn_kwargs)
            if self._capture is not None:
                self._capture[layer.layer_id] = (
                    (seed, _CaptureAdditionConv(self.linear, layer.layer_id))
                    if self.mode == _ADDITION_FULL
                    else _CaptureTrrPrefix(self.linear, layer.layer_id)
                )
            return output

        worker.forward_batch_generation = MethodType(forward_generation, worker)
        setattr(self.linear, gdn_method, MethodType(forward_gdn, self.linear))

    def _forward_generation(self, batch=None, forward_batch=None, pp_proxy_tensors=None,
                            is_verify=False, skip_attn_backend_init=None):
        from .hypic.backend.kvbench_hypic.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        ordinary = dict(forward_batch=forward_batch, pp_proxy_tensors=pp_proxy_tensors,
                        is_verify=is_verify, skip_attn_backend_init=skip_attn_backend_init)
        if (batch is None or batch.forward_mode != ForwardMode.EXTEND
                or not any(req.rid.startswith(_ONLINE_RID) for req in batch.reqs)):
            return self._worker_forward(batch, **ordinary)
        if (len(batch.reqs) != 1 or batch.return_logprob or batch.return_hidden_states
                or is_verify or pp_proxy_tensors is not None or skip_attn_backend_init):
            raise ValueError("HYPIC ablations support one text generation request without logprobs/hidden states")
        req = batch.reqs[0]
        if req.output_ids or req.is_retracted:
            raise RuntimeError("HYPIC ablations cannot resume a retracted prefill")
        if batch.pic_mode != self.pic_mode:
            raise RuntimeError(f"HYPIC ablation requires {self.pic_mode!r} PIC scheduling")
        self.worker.set_hicache_consumer(batch.hicache_consumer_index)
        source = ForwardBatch.init_new(batch, self.runner)
        suffix = (self._prepare_addition_suffix(source, req)
                  if self.mode == _ADDITION_FULL else self._prepare_suffix(source, req))
        return self._worker_forward(None, forward_batch=suffix)

    def _install_addition_cleanup(self, tree_cache):
        # Addition's request map points into shared prepared KV. Full-prefix
        # reference writes therefore need a separate private prompt allocation,
        # retained through decode and released on both completion and abort.
        if tree_cache is None or tree_cache.pic_mode != "addition":
            raise ValueError("Addition ablation requires its scheduler's Addition PICache")
        self.tree_cache = tree_cache
        self.allocator = self.runner.token_to_kv_pool_allocator
        finished = tree_cache.cache_finished_req

        def cache_finished(_cache, req, *args, **kwargs):
            self._release_addition_prompt(req)
            return finished(req, *args, **kwargs)

        tree_cache.cache_finished_req = MethodType(cache_finished, tree_cache)

    def _release_addition_prompt(self, req):
        owned = getattr(req, "_hypic_ablation_prompt_slots", None)
        if owned is None:
            return
        del req._hypic_ablation_prompt_slots
        private, original = owned
        try:
            if req.req_pool_idx is not None:
                self.pool.write((req.req_pool_idx, slice(0, private.numel())), original)
        finally:
            self.allocator.free(private)
            self.tree_cache.remove_inflight(private.numel(), 0)

    def _prepare_addition_suffix(self, source, req):
        segments = req.pic_segments
        if not segments or segments[-1][1] != len(req.origin_input_ids):
            raise RuntimeError("HYPIC ablation segments must cover the original prompt")
        boundary, end = segments[-1]
        if not boundary:
            # No historical KV to replace. Native one-segment Addition already
            # uses zero initial state and ordinary causal attention over all IDs.
            return source
        if not req.pic_miss_segments or tuple(req.pic_miss_segments[-1]) != (boundary, end):
            raise RuntimeError("Addition ablation requires the final PIC segment to be a miss")

        snapshots = {}
        self._capture = snapshots
        try:
            donor = self.runner.forward(source)
            del donor
        finally:
            self._capture = None
        if set(snapshots) != set(self.pool.mamba_map):
            raise RuntimeError("HYPIC ablation did not capture every Addition GDN query state")

        req_index = int(source.req_pool_indices[0].item())
        state_index = int(self.pool.get_mamba_indices(source.req_pool_indices)[0].item())
        original = self.pool.req_to_token[req_index, :end].clone()
        private = self.allocator.alloc(end)
        if private is None:
            raise RuntimeError("Addition ablation needs free KV slots for a private full prompt")
        private = private.clone()
        req._hypic_ablation_prompt_slots = (private, original)
        self.tree_cache.add_inflight(end, 0)
        try:
            self.pool.write((req_index, slice(0, end)), private)
            self.pool.mamba_pool.clear_slots(source.req_pool_indices.new_tensor([state_index]))
            reference = _ContiguousBatch(source, req.origin_input_ids, private, 0, boundary)
            reference.is_prefill_only = True
            donor = self.runner.forward(reference)
            del donor, reference

            # Use exactly the query h0 and convolution history that native
            # Addition selected. The ordinary GDN suffix kernel evolves this
            # seed while full attention reads the reference's historical KV.
            for layer_id, (temporal, conv) in snapshots.items():
                cache = self.pool.mamba2_layer_cache(layer_id)
                cache.temporal[state_index].copy_(temporal)
                cache.conv[0][state_index].copy_(conv)
            return _ContiguousBatch(source, req.origin_input_ids, private, boundary, end)
        except BaseException:
            self._release_addition_prompt(req)
            raise

    def _prepare_suffix(self, source, req):
        import torch

        segments = req.pic_segments
        if not segments or segments[-1][1] != len(req.origin_input_ids):
            raise RuntimeError("HYPIC ablation segments must cover the original prompt")
        boundary, end = segments[-1]
        req_index = int(source.req_pool_indices[0].item())
        state_index = int(self.pool.get_mamba_indices(source.req_pool_indices)[0].item())
        slots = self.pool.req_to_token[req_index, :end].to(torch.long).clone()
        if slots.numel() != end or (slots <= 0).any().item() or slots.unique().numel() != end:
            raise RuntimeError("HYPIC ablations require private KV slots at every absolute token position")

        # A single segment has no historical cache to exchange. Keep the
        # selected GDN continuation algorithm, without inventing a prefix.
        if boundary == 0:
            if self.mode == "gdn_trr_kv_full":
                return source
            self.pool.mamba_pool.clear_slots(source.req_pool_indices.new_tensor([state_index]))
            return _ContiguousBatch(source, req.origin_input_ids, slots, 0, end)

        snapshots = {}
        try:
            if self.mode == "gdn_trr_kv_full":
                self._capture = snapshots
            # Compute the unchanged TRR donor, without invoking the sampler.
            donor = self.runner.forward(source)
            del donor
        finally:
            self._capture = None

        if self.mode == "gdn_trr_kv_full" and set(snapshots) != set(self.pool.mamba_map):
            raise RuntimeError("HYPIC ablation did not capture every GDN prefix state")
        trr_kv = None
        if self.mode == "gdn_full_kv_trr":
            trr_kv = self.kv.get_cpu_copy(slots[:boundary])

        # Reuse only request-private slots. Public prepared KV and the local
        # S/T/conv-tail slots persisted by the TRR donor stay owned by PICache.
        self.pool.mamba_pool.clear_slots(source.req_pool_indices.new_tensor([state_index]))
        reference = _ContiguousBatch(source, req.origin_input_ids, slots, 0, boundary)
        reference.is_prefill_only = True
        donor = self.runner.forward(reference)
        del donor, reference

        suffix = _ContiguousBatch(source, req.origin_input_ids, slots, boundary, end)
        if self.mode == "gdn_full_kv_trr":
            # The active GDN slot already contains full-prefill prefix state.
            self.kv.load_cpu_copy(trr_kv, slots[:boundary])
            return suffix

        # Represent the composed TRR history as one transient cached segment
        # anchored at position zero. Reusing the active slot is safe: compose
        # reads its S/T before writing the final query state into that slot.
        # This retains TRR's query three-pass and final-state compose semantics.
        for layer_id, (temporal, conv) in snapshots.items():
            cache = self.pool.mamba2_layer_cache(layer_id)
            cache.temporal[state_index].copy_(temporal)
            cache.conv[0][state_index].copy_(conv)
            cache.conv_tails[0][state_index].copy_(conv)
            cache.transition[state_index].zero_()
            cache.transition[state_index].diagonal(dim1=-2, dim2=-1).fill_(1)
        _SeedTrrSuffix(suffix, slots, boundary, end, state_index)
        return suffix


class _TrrLinearKvPrefill(_AblationPrefill):
    """Keep native LinearKV repair; replace only its initializer with TRR."""

    def __init__(self, worker, tree_cache, ratio):
        super().__init__(worker, _TRR_LINEARKV, tree_cache)
        self.ratio = ratio
        self._repair_seed = None
        initialize = self.linear._initialize_linearkv_last_block_state
        unfinished = tree_cache.cache_unfinished_req

        def seed(_linear, layer, forward_batch):
            if self._repair_seed is None:
                return initialize(layer, forward_batch)
            temporal, conv = self._repair_seed[layer.layer_id]
            index = int(self.linear.forward_metadata.mamba_cache_indices[0].item())
            cache = self.pool.mamba2_layer_cache(layer.layer_id)
            cache.temporal[index].copy_(temporal)
            cache.conv[0][index].copy_(conv)

        def cache_unfinished(cache, req, **options):
            if not req.rid.startswith(_ONLINE_RID):
                return unfinished(req, **options)
            # Like native LinearKV, publish only independent warmups. Online
            # local misses have TRR donor S/T but repaired FA KV, so they must
            # remain transient. The native rope cleanup frees these slots.
            from .hypic.backend.kvbench_hypic.srt.pic.policy import POLICIES

            req.pic_linearkv_skip_cache = True
            previous = cache.policy
            cache.policy = POLICIES["linearkv"]
            try:
                return unfinished(req, **options)
            finally:
                cache.policy = previous

        self.linear._initialize_linearkv_last_block_state = MethodType(seed, self.linear)
        tree_cache.cache_unfinished_req = MethodType(cache_unfinished, tree_cache)

    def _forward_generation(self, batch=None, forward_batch=None, pp_proxy_tensors=None,
                            is_verify=False, skip_attn_backend_init=None):
        import torch
        from .hypic.backend.kvbench_hypic.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        ordinary = dict(forward_batch=forward_batch, pp_proxy_tensors=pp_proxy_tensors,
                        is_verify=is_verify, skip_attn_backend_init=skip_attn_backend_init)
        if (batch is None or batch.forward_mode != ForwardMode.EXTEND
                or not any(req.rid.startswith(_ONLINE_RID) for req in batch.reqs)):
            return self._worker_forward(batch, **ordinary)
        if (len(batch.reqs) != 1 or batch.return_logprob or batch.return_hidden_states
                or is_verify or pp_proxy_tensors is not None or skip_attn_backend_init):
            raise ValueError("TRR/LinearKV ablation requires one ordinary text generation request")
        req, = batch.reqs
        if req.output_ids or req.is_retracted or batch.pic_mode != _TRR:
            raise RuntimeError("TRR/LinearKV ablation requires an unretracted initial TRR prefill")
        # Also select native LinearKV's transient-slot cleanup if the donor
        # fails and the request is aborted without cache_unfinished_req.
        req.pic_linearkv_skip_cache = True
        self.worker.set_hicache_consumer(batch.hicache_consumer_index)
        source = ForwardBatch.init_new(batch, self.runner)
        if not req.pic_segments or req.pic_segments[-1][1] != len(req.origin_input_ids):
            raise RuntimeError("TRR/LinearKV ablation segments must cover the original prompt")
        # With no matched history there is no KV repair or cached initializer.
        if not req.pic_hit_segments:
            return self._worker_forward(None, forward_batch=source)
        _, end = req.pic_segments[-1]
        index = int(source.req_pool_indices[0].item())
        slots = self.pool.req_to_token[index, :end].to(torch.long).clone()
        if slots.numel() != end or (slots <= 0).any().item() or slots.unique().numel() != end:
            raise RuntimeError("TRR/LinearKV ablation requires request-private KV slots")

        snapshots = {}
        self._capture = snapshots
        try:
            donor = self.runner.forward(source)
            del donor
        finally:
            self._capture = None
        if set(snapshots) != set(self.pool.mamba_map):
            raise RuntimeError("TRR/LinearKV ablation did not capture every TRR prefix state")
        repair = _LinearKvRepairBatch(source, req, slots, self.ratio)
        self._repair_seed = snapshots
        try:
            return self._worker_forward(None, forward_batch=repair)
        finally:
            self._repair_seed = None


def _LinearKvRepairBatch(source, req, slots, ratio):
    """Pack exactly native LinearKV's selected-hit plus miss-token stream."""
    import torch
    from .hypic.backend.kvbench_hypic.srt.pic.selector import (
        merge_online_positions,
        select_epic_positions,
    )

    selected = select_epic_positions(req.pic_hit_segments, ratio=ratio)
    positions = merge_online_positions(req.pic_miss_segments, selected)
    if not positions:
        raise RuntimeError("TRR/LinearKV repair requires a nonempty query")
    end = len(req.origin_input_ids)
    batch = _ContiguousBatch(source, req.origin_input_ids, slots, 0, end)
    indices = torch.tensor(positions, dtype=torch.long, device=source.input_ids.device)
    count = len(positions)
    # Native LinearKV retains every matched token in logical prefix accounting
    # while its physical GDN stream also contains the selected repair rows.
    prefix = sum(end - start for start, end, _ in req.pic_hit_segments)
    batch.input_ids = batch.input_ids.index_select(0, indices)
    batch.positions = indices.to(source.positions.dtype)
    batch.out_cache_loc = slots.index_select(0, indices)
    batch.extend_num_tokens = count
    batch.extend_seq_lens.fill_(count)
    batch.extend_prefix_lens.fill_(prefix)
    batch.extend_seq_lens_cpu = [count]
    batch.extend_prefix_lens_cpu = [prefix]
    batch.num_token_non_padded_cpu = count
    if batch.num_token_non_padded is not None:
        batch.num_token_non_padded.fill_(count)
    if batch.mrope_positions is not None:
        batch.mrope_positions = batch.mrope_positions.index_select(1, indices)
    batch.pic_mode = "linearkv"
    batch.pic_hit_segments = source.pic_hit_segments
    batch.pic_miss_segments = source.pic_miss_segments
    batch.pic_hit_mamba_slots = source.pic_hit_mamba_slots
    batch.pic_miss_mamba_slots = source.pic_miss_mamba_slots
    batch.pic_linearkv_selected_positions = [selected]
    batch.pic_linearkv_independent_prefill = [False]
    batch.pic_rope_meta = [{**meta, "seam": None, "repair": {"positions": selected}}
                           for meta in source.pic_rope_meta]
    return batch


def _CaptureAdditionSeed(linear, layer_id, forward_batch):
    """Snapshot query h0 before Addition's kernel mutates its temporary state.

    Native Addition seeds only its first miss from the hit-slot dictionary
    (including that dictionary's deduplication). Later misses start at zero.
    Do not replace this with a sum of freshly computed miss states or TRR S/T.
    """
    import torch

    misses, = forward_batch.pic_miss_segments
    cache = linear.req_to_token_pool.mamba2_layer_cache(layer_id)
    history = torch.zeros_like(linear._pic_temp_states_buf[-1])
    hits = (forward_batch.pic_hit_mamba_slots[0]
            if forward_batch.pic_hit_mamba_slots else {})
    if len(misses) == 1 and hits:
        indices = torch.tensor(list(hits.values()), dtype=torch.long, device=history.device)
        history.copy_(cache.temporal.index_select(0, indices).sum(dim=0))
    return history.detach().to("cpu", copy=True)


def _CaptureAdditionConv(linear, layer_id):
    # Capture after donor execution, when fresh predecessor tails have been
    # persisted. Reading its active conv slot would include the query itself.
    cache = linear.req_to_token_pool.mamba2_layer_cache(layer_id)
    previous = int(linear._pic_prev_tail_slot[-1].item())
    conv = (cache.conv_tails[0][previous] if previous >= 0
            else cache.conv[0].new_zeros(cache.conv[0].shape[1:]))
    return conv.detach().to("cpu", copy=True)


def _CaptureTrrPrefix(linear, layer_id):
    """Re-compose history using the donor's S/T, stopping before query.

    Pass 3 overwrites its initial-state buffer in place. Replaying the same
    ordered compositions after the layer forward avoids patching a kernel or
    reading that overwritten buffer. Query conv history is its previous
    segment's cached raw input tail, not the active slot after query processing.
    """
    import torch

    plan, = linear._pic_compose_plan
    if plan["seg_order"][-1]["kind"] != "query":
        raise RuntimeError("HYPIC ablation could not locate the TRR continuation row")
    cache = linear.req_to_token_pool.mamba2_layer_cache(layer_id)
    history = torch.zeros_like(linear._pic_h_accum_buf[0])

    def advance_row(state, row):
        return (torch.bmm(state, linear._pic_recompute_temp_transitions[row].float())
                + linear._pic_recompute_temp_states[row].float())

    for entry in plan["seg_order"][:-1]:
        if entry["kind"] != "sys" and entry["sink_row"] is not None:
            history = advance_row(history, entry["sink_row"])
        if entry["use_pool_interior"]:
            slot = entry["mamba_slot"]
            history = torch.bmm(history, cache.transition[slot]) + cache.temporal[slot]
        elif entry["interior_row"] is not None:
            history = advance_row(history, entry["interior_row"])

    # Every nonempty global suffix has the final physical convolution row.
    if int(linear._pic_phys_seg_conv_indices[-1].item()) != plan["req_dst_slot"]:
        raise RuntimeError("HYPIC ablation could not locate the query convolution state")
    previous = int(linear._pic_phys_seg_prev_tail_slot[-1].item())
    conv = (cache.conv_tails[0][previous] if previous >= 0
            else torch.zeros_like(cache.conv[0][plan["req_dst_slot"]]))
    return history.detach().to("cpu", copy=True), conv.detach().to("cpu", copy=True)


def _ContiguousBatch(source, ids, slots, start, end):
    """Build fresh metadata for an ordinary contiguous prefill/continuation."""
    import torch
    from .hypic.backend.kvbench_hypic.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    count = end - start
    if count <= 0:
        raise ValueError("HYPIC ablation forwards must contain at least one token")
    positions = torch.arange(start, end, device=source.input_ids.device,
                             dtype=source.positions.dtype)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=source.input_ids.new_tensor(ids[start:end]),
        req_pool_indices=source.req_pool_indices,
        seq_lens=source.seq_lens.new_tensor([end]),
        seq_lens_cpu=torch.tensor([end], dtype=(source.seq_lens_cpu.dtype
                                               if source.seq_lens_cpu is not None else torch.int64)),
        seq_lens_sum=end, out_cache_loc=slots[start:end],
        orig_seq_lens=(source.orig_seq_lens.new_tensor([end])
                       if source.orig_seq_lens is not None else None),
        sampling_info=source.sampling_info, spec_algorithm=source.spec_algorithm,
        lora_ids=source.lora_ids, rids=source.rids,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        positions=positions, extend_num_tokens=count,
        extend_seq_lens=torch.tensor([count], dtype=torch.int32, device=positions.device),
        extend_prefix_lens=torch.tensor([start], dtype=torch.int32, device=positions.device),
        extend_start_loc=torch.zeros(1, dtype=torch.int32, device=positions.device),
        extend_prefix_lens_cpu=[start], extend_seq_lens_cpu=[count],
        extend_logprob_start_lens_cpu=[0],
        num_token_non_padded=(source.num_token_non_padded.new_tensor(count)
                              if source.num_token_non_padded is not None else None),
        num_token_non_padded_cpu=count, is_extend_in_batch=True,
        all_extend_in_batch=True,
    )
    if source.mrope_positions is not None:
        # Text-only mRoPE advances every positional axis together.
        batch.mrope_positions = positions.to(source.mrope_positions.dtype).unsqueeze(0).expand(
            source.mrope_positions.shape[0], -1,
        ).contiguous()
    return batch


def _SeedTrrSuffix(batch, slots, boundary, end, state_index):
    # No Req/PICache metadata is changed: these two segments are only a
    # forward-batch view, and request cleanup still sees its original segments.
    key = b"kvbench-hypic-ablation-composed-prefix"
    batch.pic_mode = _TRR
    batch.pic_hit_segments = [[(0, boundary, key)]]
    batch.pic_hit_mamba_slots = [{key: state_index}]
    batch.pic_miss_segments = [[(boundary, end)]]
    batch.pic_miss_mamba_slots = [{(boundary, end): state_index}]
    batch.pic_rope_meta = [{
        "local_miss": [],
        # These K are already at absolute positions. The virtual segment
        # starts at zero, so the existing phase-B rotation is the identity.
        "local_hit": [(0, boundary, slots[:boundary], slots[:boundary])],
        "global": (boundary, end, slots[boundary:end]),
        "seam": {"hit_seam": {}}, "repair": {"positions": {}},
    }]
