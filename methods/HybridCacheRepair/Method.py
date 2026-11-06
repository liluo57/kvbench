"""Token-native Qwen3.8 hybrid cache reuse through the existing Hypic runtime."""
from __future__ import annotations

from functools import partial
import math
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from core.Config import Get
from core.Method import Method
from helpers.backends.ModelAdapter import _tokenizer, arch_family
from helpers.backends.Tokenization import TerminalStopIdsFromTokenizer
from methods.hypic.Method import HypicMethod, _BuildHypicEngineKwargs, _QWEN35_SYSTEM, _WARMUP_TAIL

_RID = "kvbench-hybrid-cacherepair:"


class HybridCacheRepair(HypicMethod):
    """Use native TRR history, repaired attention KV, then replay the query.

    Preparation stores both ordinary Hypic warmups and true independent KV.
    All online work, including the TRR donor, fresh-span prefill, TP transfers,
    repair and query replay, is included in inherited end-to-end TTFT.
    """
    name = "hybrid_cacherepair"

    def __init__(self, gpuNums=2, perfWeight=1., *, repairerPath=None,
                 kvTileTokens=256, residualAlpha=None, **kwargs):
        if kwargs.get("fullPrefill", False) or kwargs.get("picMode", "transition_rope_recompute") != "transition_rope_recompute":
            raise ValueError("HybridCacheRepair requires native Hypic TRR; use HypicMethod for controls")
        if kwargs.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError("The current repairer requires a BF16 target")
        if isinstance(kvTileTokens, bool) or not isinstance(kvTileTokens, int) or kvTileTokens < 1:
            raise ValueError("kvTileTokens must be a positive integer")
        config = Get("HybridCacheRepair", {}) or {}
        if not isinstance(config, dict):
            raise TypeError("HybridCacheRepair configuration must be a mapping")
        alpha = config.get("ResidualAlpha", 1.) if residualAlpha is None else residualAlpha
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("residualAlpha must be a finite nonnegative number")
        self.residualAlpha = float(alpha)
        path = repairerPath or config.get("RepairerPath")
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("Set HybridCacheRepair.RepairerPath to a trained local checkpoint")
        path = Path(path).expanduser()
        self.repairerPath = str((Path(__file__).resolve().parents[2] / path if not path.is_absolute() else path).resolve())
        self.kvTileTokens = kvTileTokens
        super().__init__(gpuNums, perfWeight, **kwargs)

    def Initialize(self, gpuIds: Sequence[int]):
        Method.Initialize(self, gpuIds)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpuIds))
        if arch_family(self.modelPath) != "qwen3_5":
            raise ValueError("HybridCacheRepair currently supports Qwen3.8's hybrid Qwen3.5 backbone")
        # Validate the target snapshot before starting expensive scheduler workers.
        from .Runtime import checked_checkpoint
        checked_checkpoint(self.repairerPath, self.modelPath)
        self._tokenizer = _tokenizer(self.modelPath)
        self._terminalStopIds = TerminalStopIdsFromTokenizer(self._tokenizer, self.samplingConfig)
        self._systemIds = self._tokenizer.encode(_QWEN35_SYSTEM, add_special_tokens=False)
        self._tailIds = self._tokenizer.encode(_WARMUP_TAIL, add_special_tokens=False)
        self.engine = _create_engine(self.modelPath, self.gpuIds, self.repairerPath,
            self.kvTileTokens, alpha=self.residualAlpha, dtype=self.dtype, maxModelLen=self.maxModelLen,
            memFractionStatic=self.memFractionStatic, picMode=self.picMode,
            fullPrefill=False, seamTokens=self.seamTokens)

    def _Warmup(self, chunk):
        self.engine.hybrid_phase = "trr_prepare"
        try:
            super()._Warmup(chunk)
            self.engine.hybrid_phase = "independent_prepare"
            # The worker runs an ordinary causal prefill into private slots.
            # A nonempty fresh tail keeps Hypic's final segment a miss.
            ids = [*chunk, *self._tailIds]
            self._Generate(ids, [(0, len(chunk)), (len(chunk), len(ids))], maxNewTokens=1)
        finally:
            self.engine.hybrid_phase = "run"

    def Run(self, data, retainOutput=None, maxNewTokens=None):
        results = super().Run(data, retainOutput=retainOutput, maxNewTokens=maxNewTokens)
        for result in results:
            repaired = result.metadata["num_cached_tokens"] > 0
            result.metadata.update(gdn_state_source="hypic_trr" if repaired else "full_prefill",
                attention_kv_source="cacherepair" if repaired else "full_prefill", repairer_invoked=repaired,
                repairer_path=self.repairerPath, kv_tile_tokens=self.kvTileTokens,
                residual_alpha=self.residualAlpha,
                repairer_input="independent_causal_prefill", implementation="trr_donor_repaired_history_query_replay")
        return results


def _create_engine(modelPath, gpuIds, checkpoint, tile, *, alpha=1., **kwargs):
    from methods.hypic.backend.kvbench_hypic.srt.entrypoints.engine import Engine

    class HybridEngine(Engine):
        run_scheduler_process_func = staticmethod(partial(_run_scheduler, checkpoint=checkpoint, tile=tile, alpha=alpha))
        hybrid_phase = "run"

        def generate(self, *args, **options):
            options["rid"] = f"{_RID}{self.hybrid_phase}:{uuid4().hex}"
            return super().generate(*args, **options)

    options = _BuildHypicEngineKwargs(modelPath, gpuIds, **kwargs)
    options.update(disable_overlap_schedule=True, disable_cuda_graph=True,
                   linear_attn_prefill_backend="triton", linear_attn_decode_backend="triton")
    return HybridEngine(**options)


def _run_scheduler(*args, checkpoint, tile, alpha=1., **kwargs):
    from methods.hypic.backend.kvbench_hypic.srt.managers import scheduler as module
    from .Runtime import HybridPrefill
    module.load_plugins()
    original = module.Scheduler

    class HybridScheduler(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._hybrid_cacherepair = HybridPrefill(self.tp_worker, self.tree_cache, checkpoint, tile, alpha=alpha)

    module.Scheduler = HybridScheduler
    try:
        module.run_scheduler_process(*args, **kwargs)
    finally:
        module.Scheduler = original
