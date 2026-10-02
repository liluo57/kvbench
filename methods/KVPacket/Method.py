

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from core.Config import DefaultConfigPath, Get, ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import IsGreedy, ResolveSamplingConfig
from helpers.backends.Tokenization import OutputFragmentIds, TerminalStopIdsFromTokenizer


@dataclass
class _CaseState:
    """Prepared packet store and source token sequences for one case."""

    chunks: List[List[int]]
    store: Any


def _resolve_config_path(value: object) -> Optional[Path]:
    """Resolve a config path relative to ``config.yaml`` when necessary."""

    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = DefaultConfigPath.parent / path
    return path.resolve()


class KVPacket(Method):
    """Recomputation-free KV Packet serving through ``PacketSession``."""

    name = "kvpacket"
    backend = "kvpacket"
    method_metrics = ("reuse_ratio",)
    # PacketSession owns a mutable prefix/cache and the HF backend uses one model
    # instance.  Keeping each case isolated avoids mixing differently shaped stores.
    maxCaseBatchSize = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        wrapperCheckpointPath: Optional[str] = None,
        wrapperId: Optional[str] = None,
        wrapperModelId: Optional[str] = None,
        headerLen: Optional[int] = None,
        trailerLen: Optional[int] = None,
        match: Optional[str] = None,
        allowUntrainedWrapper: Optional[bool] = None,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)

        cfg = Get("KVPacket", {}) or {}
        if not isinstance(cfg, dict):
            raise TypeError("KVPacket in config.yaml must be a mapping")

        # Explicit constructor arguments win; the aliases make it easy to use
        # checkpoints produced by the KV Packet examples.
        checkpoint = wrapperCheckpointPath
        if checkpoint is None:
            checkpoint = cfg.get("WrapperCheckpointPath")
        if checkpoint is None:
            checkpoint = cfg.get("TrainerPath", cfg.get("CheckpointPath"))
        self.wrapperCheckpointPath = _resolve_config_path(checkpoint)

        self.wrapperId = str(wrapperId or cfg.get("WrapperId", "document"))
        configured_model_id = (
            wrapperModelId
            if wrapperModelId is not None
            else cfg.get("WrapperModelId")
        )
        if configured_model_id is not None and not isinstance(configured_model_id, str):
            raise TypeError("KVPacket.WrapperModelId must be a string or null")
        self.wrapperModelId = configured_model_id.strip() if configured_model_id else None
        self.headerLen = int(headerLen if headerLen is not None else cfg.get("HeaderLen", 8))
        self.trailerLen = int(
            trailerLen if trailerLen is not None else cfg.get("TrailerLen", 8)
        )
        self.match = str(match or cfg.get("Match", "token"))
        if self.match != "token":
            raise ValueError(
                "KVPacket token-native input supports Match='token'; string matching "
                "requires source text and offsets"
            )
        configured_untrained = (
            allowUntrainedWrapper
            if allowUntrainedWrapper is not None
            else cfg.get("AllowUntrainedWrapper", False)
        )
        if not isinstance(configured_untrained, bool):
            raise TypeError("KVPacket.AllowUntrainedWrapper must be a boolean")
        self.allowUntrainedWrapper = configured_untrained
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        self.dtype = dtype

        self._torch = None
        self._model = None
        self._tokenizer = None
        self._terminalStopIds: List[int] = []
        self._backend = None
        self._kv = None
        self._wrapper = None
        self._wrapper_format = "untrained"
        self._states: List[_CaseState] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        if self.wrapperCheckpointPath is None and not self.allowUntrainedWrapper:
            raise RuntimeError(
                "KVPacket requires a model-compatible trained wrapper checkpoint. "
                "Set KVPacket.WrapperCheckpointPath to a checkpoint trained for this "
                "exact model, or explicitly set KVPacket.AllowUntrainedWrapper=true "
                "for a smoke test (results are not representative of trained KVPacket)."
            )
        if self.wrapperCheckpointPath is not None and not self.wrapperCheckpointPath.exists():
            raise FileNotFoundError(
                f"KVPacket wrapper checkpoint not found: {self.wrapperCheckpointPath}"
            )
        from .backend import kvpacket

        self._kv = kvpacket

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        first_gpu = int(self.gpuIds[0]) if self.gpuIds else None
        if first_gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(first_gpu)
            device = f"cuda:{first_gpu}"
        else:
            device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.modelPath)
        self._terminalStopIds = TerminalStopIdsFromTokenizer(
            self._tokenizer, self.samplingConfig
        )
        model_dtype = getattr(torch, self.dtype)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.modelPath,
                dtype=model_dtype,
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.modelPath,
                torch_dtype=model_dtype,
            )
        self._model.to(device)
        self._model.eval()

        self._backend = self._kv.HFBackend(self._model)
        self._wrapper = self._load_wrapper(device)

    def _load_wrapper(self, device: str):
        """Load a trained wrapper or construct a configurable smoke-test wrapper."""

        assert self._kv is not None and self._backend is not None and self._torch is not None
        dtype = next(self._model.parameters()).dtype
        if self.wrapperCheckpointPath is not None:
            if not self.wrapperCheckpointPath.exists():
                raise FileNotFoundError(
                    f"KVPacket wrapper checkpoint not found: {self.wrapperCheckpointPath}"
                )
            state = self._torch.load(
                self.wrapperCheckpointPath,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict) and state.get("format") == "kvpacket.trainer_checkpoint/1":
                trainer = self._kv.PacketTrainer.load(
                    self.wrapperCheckpointPath,
                    backend=self._backend,
                    device=device,
                )
                if self.wrapperId not in trainer.registry:
                    raise KeyError(
                        f"KVPacket wrapper {self.wrapperId!r} is absent from "
                        f"{self.wrapperCheckpointPath}"
                    )
                wrapper = trainer.registry[self.wrapperId].wrapper
                wrapper = wrapper.to(device=device, dtype=dtype)
                wrapper.requires_grad_(False)
                wrapper.eval()
                self._wrapper_format = "trainer_checkpoint"
                return wrapper
            return self._load_legacy_wrapper(state, device=device, dtype=dtype)

        if not self.allowUntrainedWrapper:
            raise RuntimeError(
                "KVPacket requires a model-compatible trained wrapper checkpoint. "
                "Set KVPacket.WrapperCheckpointPath to a checkpoint trained for this "
                "exact model, or explicitly set KVPacket.AllowUntrainedWrapper=true "
                "for a smoke test (results are not representative of trained KVPacket)."
            )
        if self.headerLen < 0 or self.trailerLen < 0:
            raise ValueError("KVPacket HeaderLen and TrailerLen must be non-negative")
        return self._kv.PacketWrapper(
            self.headerLen,
            self.trailerLen,
            self._backend.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    def _load_legacy_wrapper(self, state: object, *, device: str, dtype: Any):
        """Load a paper-era tensor-only wrapper with an explicit model identity."""

        import torch

        if not isinstance(state, dict) or not all(
            key in state for key in ("header", "trailer", "header_len", "trailer_len", "dim")
        ):
            raise ValueError(
                "unsupported KVPacket wrapper checkpoint format; expected a current "
                "PacketTrainer checkpoint or a legacy header/trailer checkpoint"
            )
        if self.wrapperModelId is None:
            raise ValueError(
                "legacy KVPacket wrapper files do not record an exact model identity or "
                "weight revision; set KVPacket.WrapperModelId to the exact model ID "
                "reported by the loaded HF backend after verifying that it names the "
                "checkpoint's source weights"
            )
        runtime_model_id = self._backend.compatibility.model_id
        if self.wrapperModelId != runtime_model_id:
            raise ValueError(
                "legacy KVPacket wrapper model mismatch: configured WrapperModelId "
                f"{self.wrapperModelId!r}, loaded model is {runtime_model_id!r}"
            )

        header, trailer = state["header"], state["trailer"]
        header_len, trailer_len, dim = (
            state["header_len"], state["trailer_len"], state["dim"]
        )
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (
            header_len, trailer_len, dim
        )):
            raise ValueError("legacy KVPacket wrapper dimensions must be integers")
        if dim != self._backend.hidden_size:
            raise ValueError(
                f"legacy KVPacket wrapper dim {dim} does not match model hidden size "
                f"{self._backend.hidden_size}"
            )
        if not isinstance(header, torch.Tensor) or not isinstance(trailer, torch.Tensor):
            raise ValueError("legacy KVPacket header and trailer must be tensors")
        if tuple(header.shape) != (1, header_len, dim):
            raise ValueError(f"legacy KVPacket header shape is invalid: {tuple(header.shape)}")
        if tuple(trailer.shape) != (1, trailer_len, dim):
            raise ValueError(f"legacy KVPacket trailer shape is invalid: {tuple(trailer.shape)}")

        wrapper = self._kv.PacketWrapper(
            header_len,
            trailer_len,
            dim,
            mean=float(state.get("mean", 0.0)),
            std=float(state.get("std", 0.02)),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        with torch.no_grad():
            wrapper.header.copy_(header.to(device=device, dtype=dtype))
            wrapper.trailer.copy_(trailer.to(device=device, dtype=dtype))
        wrapper.eval()
        self._wrapper_format = "legacy_tensor_checkpoint"
        return wrapper

    def Prepare(self, data: List[List[List[int]]]) -> None:
        """Materialize one packet store per case from prepared token chunks."""

        assert self._kv is not None and self._backend is not None and self._tokenizer is not None
        from .backend.kvpacket import (
            PacketPreprocessor,
            PacketStore,
            RawText,
            WrapperEntry,
            WrapperRegistry,
        )

        self._states = []
        for case_index, raw_chunks in enumerate(data):
            # Duplicate texts need not be packetized twice and can make matching
            # ambiguous. Preserve first occurrence order like CacheBlend does.
            raw_chunks = list(raw_chunks or [])

            chunks: list[list[int]] = []
            seen: set[tuple[int, ...]] = set()
            for raw_chunk in raw_chunks or []:
                ids = [int(token) for token in raw_chunk]
                key = tuple(ids)
                if ids and key not in seen:
                    seen.add(key)
                    chunks.append(ids)

            store = PacketStore(backend=self._backend)
            if chunks:
                registry = WrapperRegistry(
                    compatibility=self._backend.compatibility,
                    wrappers={self.wrapperId: WrapperEntry(self._wrapper)},
                )
                preprocessor = PacketPreprocessor(
                    backend=self._backend,
                    wrapper_registry=registry,
                    store=store,
                )
                for chunk_index, ids in enumerate(chunks):
                    digest = hashlib.sha256(
                        b"".join(
                            token.to_bytes(4, "little", signed=True)
                            for token in ids
                        )
                    ).hexdigest()[:16]
                    source_id = f"case{case_index}_chunk{chunk_index}_{digest}"
                    preprocessor.preprocess_pretokenized(
                        RawText("", source_id=source_id),
                        ids,
                        wrapper_id=self.wrapperId,
                    )
            self._states.append(_CaseState(chunks=chunks, store=store))

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        """Serve complete prompts through independent packet sessions."""

        assert self._backend is not None and self._tokenizer is not None
        from .backend.kvpacket import Mode, PacketSession

        generation_limit = (
            self.maxNewTokens
            if maxNewTokens is None
            else ResolveMaxNewTokens(maxNewTokens)
        )
        sampling = self.samplingConfig
        sampling_kwargs: dict[str, Any] = {}
        if not IsGreedy(sampling):
            sampling_kwargs["do_sample"] = True
            for key in ("temperature", "top_p", "top_k", "min_p"):
                if key in sampling:
                    value = sampling[key]
                    # PacketSession uses zero to disable top-k; HF configs also
                    # allow -1 as the equivalent "disabled" value.
                    if key == "top_k" and value < 0:
                        value = 0
                    sampling_kwargs[key] = value
        if "eos_token_id" in sampling:
            sampling_kwargs["eos_token_id"] = sampling["eos_token_id"]

        if len(self._states) != len(data):
            from .backend.kvpacket import PacketStore

            self._states = [
                _CaseState(chunks=[], store=PacketStore(backend=self._backend))
                for _ in data
            ]

        results: list[Result] = []
        for prompt, state in zip(data, self._states, strict=True):
            request_started = time.perf_counter()
            session = PacketSession(
                backend=self._backend,
                mode=Mode.SERVE,
                match="token",
                store=state.store,
                # IDs are authoritative. Omitting the tokenizer here also
                # avoids a hidden re-encode while validating prepared packets;
                # output text is decoded from the native generated IDs below.
                tokenizer=None,
            )
            if state.store.ids():
                session.register_store_chunks()
            input_ids = [int(token) for token in prompt]
            session.push(input_ids)
            generated = session.generate(
                max_new_tokens=generation_limit,
                started_at=request_started,
                **sampling_kwargs,
            )
            if self._backend.device.type == "cuda":
                # Finish any asynchronous cache commits before recording request
                # latency; TTFT was already captured at the first visible token.
                self._torch.cuda.synchronize(self._backend.device)

            n_input = len(input_ids)
            cached_tokens = int(generated.cached_tokens)
            reuse_ratio = cached_tokens / n_input if n_input else 0.0
            output_ids = OutputFragmentIds(
                generated.token_ids,
                self._terminalStopIds,
            )
            results.append(
                Result(
                    output=self._tokenizer.decode(
                        generated.token_ids, skip_special_tokens=True
                    ),
                    output_ids=output_ids,
                    performance={
                        # The start time covers registration, matching, prefill,
                        # and generation through the first visible token.
                        TtftKey: float(generated.ttft),
                        NumOutputTokensKey: len(generated.token_ids),
                        TotalTimeKey: float(time.perf_counter() - request_started),
                    },
                    metadata={
                        "backend": self.backend,
                        "wrapper_status": (
                            self._wrapper_format
                        ),
                        "reuse_ratio": reuse_ratio,
                        "cached_tokens": cached_tokens,
                        "matched_chunk_ids": generated.matched_chunk_ids,
                        "n_input": n_input,
                        "flops": int(generated.flops),
                    },
                )
            )
        return results

    def Reset(self) -> None:
        self._states = []

    def Close(self) -> None:
        self._states = []
        self._wrapper = None
        self._backend = None
        self._tokenizer = None
        self._model = None
        try:
            if self._torch is not None:
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


KVPacketTransformer = KVPacket
KVPacketMethod = KVPacket
KvPacket = KVPacket


__all__ = ["KVPacket"]
