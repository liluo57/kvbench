

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from core.Config import (
    DefaultConfigPath,
    Get,
    MaxModelLen,
    ModelPath as DefaultModelPath,
)
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import IsGreedy, ResolveSamplingConfig
from helpers.backends.Tokenization import OutputFragmentIds, TerminalStopIdsFromTokenizer


_SUPPORTED_SAMPLING_FIELDS = {
    "mode",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "eos_token_id",
}


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
        unsupported = sorted(set(self.samplingConfig) - _SUPPORTED_SAMPLING_FIELDS)
        if unsupported:
            raise ValueError(
                "KVPacket does not support resolved sampling field(s): "
                + ", ".join(unsupported)
            )
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
                if ids and len(ids) + self._wrapper.header_len + self._wrapper.trailer_len > MaxModelLen():
                    raise ValueError(f"Wrapped KVPacket chunk exceeds MaxModelLen={MaxModelLen()}")
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
        """Serve complete prompts through independent packet sessions.

        When ``retainOutput`` is true, the generated native output IDs and their
        completed, unwrapped KV are registered as a reusable packet for later
        workflow prompts. Terminal stop IDs are excluded from both the IDs and KV.
        """

        assert self._backend is not None and self._tokenizer is not None
        from .backend.kvpacket import Mode, PacketSession

        generation_limit = (
            self.maxNewTokens
            if maxNewTokens is None
            else ResolveMaxNewTokens(maxNewTokens)
        )
        if any(len(prompt) >= MaxModelLen() for prompt in data):
            raise ValueError(f"KVPacket prompt must leave room for generation within MaxModelLen={MaxModelLen()}")
        sampling = self.samplingConfig
        do_sample = not IsGreedy(sampling)
        temperature = float(sampling.get("temperature", 1.0))
        # The packet selector validates positive temperatures even in greedy
        # mode, where temperature is intentionally irrelevant.
        if temperature <= 0 and not do_sample:
            temperature = 1.0
        top_k = int(sampling.get("top_k", 0))
        if top_k < 0:
            top_k = 0
        sampling_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": float(sampling.get("top_p", 1.0)),
            "top_k": top_k,
            "min_p": float(sampling.get("min_p", 0.0)),
            "typical_p": float(sampling.get("typical_p", 1.0)),
            "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
            "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
            "frequency_penalty": float(sampling.get("frequency_penalty", 0.0)),
            "seed": sampling.get("seed"),
        }
        eos_ids = sampling.get("eos_token_id")
        if eos_ids is None:
            eos_ids = getattr(
                getattr(self._model, "generation_config", None), "eos_token_id", None
            )
        if eos_ids is None:
            eos_ids = self._tokenizer.eos_token_id
        # The tokenizer is intentionally omitted from PacketSession. Pass the
        # effective EOS explicitly so its first-token selection and the native
        # continuation use the same stopping policy.
        sampling_kwargs["eos_token_id"] = eos_ids if eos_ids is not None else []
        self._terminalStopIds = TerminalStopIdsFromTokenizer(
            self._tokenizer, {"eos_token_id": sampling_kwargs["eos_token_id"]}
        )

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
                # The same native IDs may also exist as prepared, wrapped packets.
                # Prefer their explicitly unwrapped native output packet so matching
                # never aliases generated KV to a different wrapper source.
                native_sequences = {
                    tuple(packet.token_ids)
                    for chunk_id in state.store.ids()
                    for packet in [state.store.get(chunk_id)]
                    if packet.identity.wrapper_id == "native_output"
                    and packet.token_ids is not None
                }
                packet_ids = [
                    chunk_id
                    for chunk_id in state.store.ids()
                    if (
                        state.store.get(chunk_id).identity.wrapper_id == "native_output"
                        or tuple(state.store.get(chunk_id).token_ids or ())
                        not in native_sequences
                    )
                ]
                session.register_store_chunks(packet_ids)
            input_ids = [int(token) for token in prompt]
            session.push(input_ids)
            cover = session.preview_matches()
            matched_tokens = sum(item.end - item.start for item in cover)
            # PacketSession keeps a terminal match fresh to obtain first-token
            # logits. Count only substituted occurrences, including their
            # wrapper positions even when their physical KV is compressed.
            if cover and cover[-1].end == len(input_ids):
                cover = cover[:-1]
            logical_length = len(input_ids) + sum(
                state.store.get(item.chunk_id).wrapped_len - (item.end - item.start)
                for item in cover
            )
            if logical_length >= MaxModelLen():
                raise ValueError(f"Wrapped KVPacket prompt must leave room for generation within MaxModelLen={MaxModelLen()}")
            retain = (
                retainOutput is not None
                and len(results) < len(retainOutput)
                and bool(retainOutput[len(results)])
            )
            generated = session.generate(
                max_new_tokens=min(generation_limit, MaxModelLen() - logical_length),
                started_at=request_started,
                return_output_cache=retain,
                **sampling_kwargs,
            )
            if self._backend.device.type == "cuda":
                # Finish any asynchronous cache commits before recording request
                # latency; TTFT was already captured at the first visible token.
                self._torch.cuda.synchronize(self._backend.device)

            n_input = len(input_ids)
            # Report semantic prompt coverage, including a terminal packet that
            # is deliberately forwarded fresh to provide first-token logits.
            # Wrapper positions and physically stored compressed rows are not
            # input-token coverage and therefore do not affect this metric.
            cached_tokens = matched_tokens
            substituted_tokens = int(generated.cached_tokens)
            reuse_ratio = cached_tokens / n_input if n_input else 0.0
            output_ids = OutputFragmentIds(
                generated.token_ids,
                self._terminalStopIds,
            )
            if retain and output_ids:
                from .backend.kvpacket.fingerprint import (
                    fingerprint_bytes,
                    fingerprint_json,
                    fingerprint_tensor,
                )
                from .backend.kvpacket.session.store import PacketIdentity, StoredPacket

                output_cache = generated.output_cache
                if output_cache is None:
                    raise RuntimeError("KVPacket did not return requested generated output KV")
                token_bytes = b"".join(
                    int(token).to_bytes(8, "little", signed=True)
                    for token in output_ids
                )
                digest = hashlib.sha256(token_bytes).hexdigest()
                source_id = f"native_output_{digest}"
                collision = 0
                while (
                    source_id in state.store
                    and state.store.get(source_id).identity.wrapper_id != "native_output"
                ):
                    collision += 1
                    source_id = f"native_output_{digest}_{collision}"
                token_tensor = self._torch.tensor(
                    [output_ids], dtype=self._torch.long, device=self._backend.device
                )
                with self._torch.no_grad():
                    encoded = self._backend.embed(token_tensor)
                identity = PacketIdentity(
                    source_id=source_id,
                    source_kind="native_token_ids",
                    source_content_fingerprint=fingerprint_bytes(
                        token_bytes, domain="kvpacket.source_content/1"
                    ),
                    backend_compatibility_fingerprint=self._backend.compatibility.fingerprint,
                    wrapper_id="native_output",
                    wrapper_fingerprint=fingerprint_json(
                        {"kind": "native_output", "wrapper": None},
                        domain="kvpacket.wrapper/1",
                    ),
                    encoded_input_fingerprint=fingerprint_tensor(encoded),
                )
                native_packet = StoredPacket(
                    chunk_id=source_id,
                    token_ids=list(output_ids),
                    match_text=None,
                    packet=output_cache,
                    body_len=len(output_ids),
                    wrapped_len=len(output_ids),
                    identity=identity,
                )
                existing = (
                    state.store.get(source_id) if source_id in state.store else None
                )
                if existing is None or existing.identity.wrapper_id == "native_output":
                    state.store.put(
                        native_packet,
                        replace=existing is not None,
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
                        "substituted_tokens": substituted_tokens,
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


from helpers.backends.RecomputeMethod import CacheRecomputeMethod


class RandRecomputeTransformer(CacheRecomputeMethod):
    """Random repair baseline using the same local HF/cache backend as KVPacket."""

    name = "rand_recompute"

    def __init__(self, *args, recomputeRatio=0.2, recompute_ratio=None, seed=None, **kwargs):
        import math

        ratio = recomputeRatio if recompute_ratio is None else recompute_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("recomputeRatio must be a finite number in [0, 1]")
        self.recomputeRatio = float(ratio)
        self.seed = seed
        super().__init__(*args, **kwargs)

    def _select(self, hits):
        import random

        eligible = [position for hit in hits if hit.start > 0
                    for position in range(hit.start, hit.end)]
        return sorted(random.Random(self.seed).sample(
            eligible, int(len(eligible) * self.recomputeRatio),
        ))


class FullRecomputeTransformer(CacheRecomputeMethod):
    """Full recomputation control for the shared KVPacket HF backend."""

    name = "full_recompute"
    fullRecompute = True


__all__ = ["KVPacket", "RandRecomputeTransformer", "FullRecomputeTransformer"]
