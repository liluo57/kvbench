

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from core.Config import DefaultConfigPath, Get, ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import IsGreedy, ResolveSamplingConfig


@dataclass
class _CaseState:
    """Prepared packet store and source texts for one KVBench case."""

    chunks: List[str]
    store: Any


def _resolve_config_path(value: object) -> Optional[Path]:
    """Resolve a config path relative to ``config.yaml`` when necessary."""

    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = DefaultConfigPath.parent / path
    return path.resolve()


def _import_kvpacket(repo_path: Optional[Path]):
    """Import the sibling checkout when KV Packet is not installed.

    Keeping this import lazy is important: KVBench's CPU-only tests and methods
    that do not use KV Packet should not need KV Packet's optional runtime at import
    time.
    """

    candidates: list[Path] = []
    if repo_path is not None:
        if not (repo_path / "kvpacket").is_dir():
            raise RuntimeError(
                f"KVPacket checkout does not contain kvpacket/: {repo_path}"
            )
        candidates.append(repo_path)

    # The repository layout used by this workspace is ``kvreusebench/`` with
    # sibling ``kvbench/`` and ``KVPacket/`` directories.
    candidates.append(Path(__file__).resolve().parents[2] / "KVPacket")
    first_error = None
    if repo_path is None:
        try:
            return importlib.import_module("kvpacket")
        except ModuleNotFoundError as error:
            first_error = error

    for candidate in candidates:
        package_dir = candidate / "kvpacket"
        if not package_dir.is_dir():
            continue
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
        try:
            return importlib.import_module("kvpacket")
        except ModuleNotFoundError as error:
            first_error = first_error or error

    if repo_path is not None:
        try:
            return importlib.import_module("kvpacket")
        except ModuleNotFoundError as error:
            first_error = first_error or error
    raise RuntimeError(
        "KVPacket is not installed and no KVPacket checkout was found; "
        "install kvpacket or set KVPacket.RepoPath in config.yaml"
    ) from first_error


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
        headerLen: Optional[int] = None,
        trailerLen: Optional[int] = None,
        match: Optional[str] = None,
        repoPath: Optional[str] = None,
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
        self.headerLen = int(headerLen if headerLen is not None else cfg.get("HeaderLen", 8))
        self.trailerLen = int(
            trailerLen if trailerLen is not None else cfg.get("TrailerLen", 8)
        )
        self.match = str(match or cfg.get("Match", "token"))
        if self.match not in ("token", "string"):
            raise ValueError("KVPacket match must be 'token' or 'string'")
        self.repoPath = _resolve_config_path(repoPath or cfg.get("RepoPath"))
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        self.dtype = dtype

        self._torch = None
        self._model = None
        self._tokenizer = None
        self._backend = None
        self._kv = None
        self._wrapper = None
        self._states: List[_CaseState] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._kv = _import_kvpacket(self.repoPath)

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
        model_kwargs: dict[str, Any] = {
            "dtype": getattr(torch, self.dtype),
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.modelPath, **model_kwargs)
        except TypeError:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = getattr(torch, self.dtype)
            self._model = AutoModelForCausalLM.from_pretrained(self.modelPath, **model_kwargs)
        self._model.eval()

        from kvpacket import HFBackend

        self._backend = HFBackend(self._model)
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
            from kvpacket.training import PacketTrainer

            trainer = PacketTrainer.load(
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
            return wrapper

        if self.headerLen < 0 or self.trailerLen < 0:
            raise ValueError("KVPacket HeaderLen and TrailerLen must be non-negative")
        from kvpacket.packet import PacketWrapper

        return PacketWrapper(
            self.headerLen,
            self.trailerLen,
            self._backend.hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    def Prepare(self, data: List[List[str]]) -> None:
        """Materialize one packet store per case from the reusable chunks."""

        assert self._kv is not None and self._backend is not None and self._tokenizer is not None
        from kvpacket import PacketPreprocessor, RawText, TextSourceEncoder
        from kvpacket.packet import WrapperEntry, WrapperRegistry
        from kvpacket.session import PacketStore

        self._states = []
        for case_index, raw_chunks in enumerate(data):
            # Duplicate texts need not be packetized twice and can make matching
            # ambiguous. Preserve first occurrence order like CacheBlend does.
            raw_chunks = list(raw_chunks or [])

            chunks: list[str] = []
            seen: set[str] = set()
            for chunk in raw_chunks:
                text = str(chunk)
                if text and text not in seen:
                    seen.add(text)
                    chunks.append(text)

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
                    source_encoder=TextSourceEncoder(self._tokenizer),
                )
                assignments = {}
                for chunk_index, text in enumerate(chunks):
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    source_id = f"case{case_index}_chunk{chunk_index}_{digest}"
                    assignments[RawText(text, source_id=source_id)] = self.wrapperId
                preprocessor.preprocess(assignments)
            self._states.append(_CaseState(chunks=chunks, store=store))

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        """Serve complete prompts through independent packet sessions."""

        assert self._backend is not None and self._tokenizer is not None
        from kvpacket.session import Mode, PacketSession

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
            from kvpacket.session import PacketStore

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
                match=self.match,
                store=state.store,
                tokenizer=self._tokenizer,
            )
            if state.store.ids():
                session.register_store_chunks()
            session.push(prompt)
            request_ready = time.perf_counter()
            generated = session.generate(
                max_new_tokens=generation_limit,
                **sampling_kwargs,
            )

            input_ids = self._tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
            n_input = len(input_ids)
            cached_tokens = int(generated.cached_tokens)
            reuse_ratio = cached_tokens / n_input if n_input else 0.0
            results.append(
                Result(
                    output=generated.text,
                    performance={
                        # Store registration and request buffering are outside the
                        # session's measured generation interval. PacketSession
                        # also performs prompt matching before starting its own
                        # TTFT clock; its public result does not expose that cost.
                        TtftKey: float(request_ready - request_started)
                        + float(generated.ttft),
                        NumOutputTokensKey: len(generated.token_ids),
                        TotalTimeKey: float(time.perf_counter() - request_started),
                    },
                    metadata={
                        "backend": self.backend,
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
