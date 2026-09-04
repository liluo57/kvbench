"""System-vLLM backend helper (used by ``FullPrefillVllm``).

- Qwen2.5-7B-Instruct-1M declares a dual-chunk / sparse-attention config
  (``dual_chunk_attention_config`` + a ``sparse_attention_config.json``).
  Some vLLM builds route that through a ``layer_idx`` hook their
  ``attention.py`` does not implement (package-internal mismatch), so the model
  fails to load. We hand vLLM a *sanitized* copy of the model dir — weights
  symlinked, those config fields removed — which makes it run the same standard
  attention the transformers backend uses. Only applied when the fields are
  present (harmless when the vLLM build supports the sparse config natively).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_VllmProbeCache: Dict[str, bool] = {}

#: Keys removed from config.json for the sanitized model dir.
_StripConfigKeys = ("dual_chunk_attention_config", "sparse_attention_config")


# --------------------------------------------------------------------- probing
def SetCudaVisibleDevices(gpuIds: Union[str, List[int]]) -> None:
    """Set ``CUDA_VISIBLE_DEVICES`` from a ``gpuIds`` argument."""
    if gpuIds is None:
        return
    if isinstance(gpuIds, (list, tuple)):
        gpuIds = ",".join(str(g) for g in gpuIds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuIds)


def VllmAvailable(python: Optional[str] = None, timeout: float = 120.0) -> bool:
    """Whether ``from vllm import LLM`` works in this interpreter.

    Probed once per interpreter and cached. A non-zero exit — including a
    segfault — means vLLM is unusable. Runs with a plain environment (no
    ``LD_PRELOAD``), mirroring the in-process import in :func:`CreateLlm`.
    """
    python = python or sys.executable
    if python in _VllmProbeCache:
        return _VllmProbeCache[python]
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            [python, "-c", "from vllm import LLM  # probe"],
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        ok = proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    _VllmProbeCache[python] = ok
    return ok


# ------------------------------------------------------------ model sanitizing
def SanitizedModelDir(modelPath: Union[str, Path]) -> Path:
    """A model dir vLLM can load, stripping the 1M dual-chunk config if present.

    Returns ``modelPath`` unchanged when the config has nothing to strip.
    Otherwise builds (once, keyed by the stripped config's hash) a sibling
    directory under ``$TMPDIR/kvbench-vllm-models`` with the weights and
    tokenizer files symlinked, so nothing is copied.
    """
    modelPath = Path(modelPath)
    cfgPath = modelPath / "config.json"
    if not cfgPath.exists():
        return modelPath
    cfg = json.loads(cfgPath.read_text())
    stripped = {k: cfg.pop(k, None) for k in _StripConfigKeys}
    if not any(v is not None for v in stripped.values()):
        return modelPath

    digest = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    workDir = Path(
        os.environ.get("KVBENCH_VLLM_MODELS_DIR") or tempfile.gettempdir()
    ) / "kvbench-vllm-models" / f"{modelPath.name}-{digest}"
    if not workDir.is_dir():
        workDir.mkdir(parents=True, exist_ok=True)
        (workDir / "config.json").write_text(json.dumps(cfg, indent=2))
        for child in modelPath.iterdir():
            if child.name == "config.json":
                continue
            target = workDir / child.name
            try:
                if child.is_dir():
                    shutil.copytree(child, target, symlinks=True)
                else:
                    os.symlink(child, target)
            except FileExistsError:
                pass
    return workDir


# ------------------------------------------------------------------ LLM + run
def CreateLlm(
    modelPath: Union[str, Path],
    gpuIds: Union[str, List[int]] = "0",
    *,
    gpuMemoryUtilization: float = 0.7,
    maxModelLen: int = 40960,
    tensorParallelSize: int = 1,
    dtype: str = "bfloat16",
    enforceEager: bool = False,
    chatTemplate: Optional[str] = None,
    languageModelOnly: bool = False,
    maxNumSeqs: Optional[int] = None,
):
    """Build the system vLLM ``LLM`` for ``modelPath`` on ``gpuIds``.

    Points ``CUDA_VISIBLE_DEVICES`` at the task's GPUs and swaps in the
    sanitized model dir when the model declares the dual-chunk config (see
    :func:`SanitizedModelDir`). No ``LD_PRELOAD``: the venv is a consistent
    vLLM/torch/CUDA install.

    ``enforceEager=True`` disables CUDA-graph capture during warmup, which
    trades throughput for a much smaller startup memory peak. Useful when
    the model + KV cache almost fit and the cuda-graph allocation step
    is the one that fails with ``CUDA out of memory``.

    ``chatTemplate`` is forwarded to ``vllm.LLM(chat_template=...)``. When
    ``None`` (default), the function auto-detects ``<modelPath>/chat_template.jinja``
    so the model's own ATEM / Qwen ChatML / etc. template is used by the
    tokenizer's ``apply_chat_template`` rather than a hand-rolled renderer
    in KVBench. Pass an explicit string to override (e.g. a path to a
    custom template file, or the literal template content).
    """
    SetCudaVisibleDevices(gpuIds)
    path = SanitizedModelDir(modelPath)
    from vllm import LLM

    if chatTemplate is None:
        candidate = Path(path) / "chat_template.jinja"
        if candidate.is_file():
            chatTemplate = str(candidate)

    llm_kwargs: Dict[str, Any] = dict(
        model=str(path),
        dtype=dtype,
        gpu_memory_utilization=gpuMemoryUtilization,
        max_model_len=maxModelLen,
        tensor_parallel_size=tensorParallelSize,
        enforce_eager=enforceEager,
        # Skip the vision encoder for multimodal checkpoints (Qwen3.5 /
        # Qwen3.8 ship as ``*ForConditionalGeneration`` and vLLM would
        # otherwise load the vision tower just to ignore it). vLLM 0.28+
        # honours this kwarg via EngineArgs; see vllm/config/model.py.
        language_model_only=languageModelOnly,
    )
    if maxNumSeqs is not None:
        llm_kwargs["max_num_seqs"] = maxNumSeqs
    if chatTemplate is not None:
        llm_kwargs["chat_template"] = chatTemplate
    return LLM(**llm_kwargs)


@dataclass(frozen=True)
class VllmGeneration:
    """One completed vLLM generation, including native stop metadata."""

    text: str
    ttft: float
    numTokens: int
    totalTime: float
    numCached: int
    # These values come directly from vLLM's CompletionOutput. They must not
    # be inferred from the OpenAI endpoint response, which may map a parsed
    # tool call to ``tool_calls`` and otherwise default to ``stop``.
    finishReason: Optional[str]
    stopReason: Optional[Union[int, str]]


def Generate(
    llm, promptText: str, maxNewTokens: int
) -> VllmGeneration:
    """Generate one prompt and return its native vLLM stop metadata.

    TTFT is measured the way vLLM's own offline benchmark
    (``vllm/benchmarks/benchmark_offline.py``) does: drive the V1 engine
    directly (``add_request`` + ``step``) and take the wall-clock time from
    submission to the first generated token. vLLM 0.23's synchronous
    ``LLM.generate`` returns ``RequestOutput.metrics=None`` (it hard-codes
    ``disable_log_stats=True``), so the request-state stats aren't reachable;
    this path uses only public engine API — no vLLM source changes.

    - ``ttft`` — seconds from ``add_request`` to the first decoded token.
    - ``totalTime`` — seconds from ``add_request`` to completion.
    - ``numCached`` — prefix-cache blocks this request reused (0 for full
      prefill).
    - ``finishReason`` / ``stopReason`` — copied from the final vLLM
      :class:`CompletionOutput`, without going through the OpenAI endpoint.
    """
    return GenerateBatch(llm, [promptText], maxNewTokens)[0]


def GenerateBatch(
    llm, promptTexts: List[str], maxNewTokens: int
) -> List[VllmGeneration]:
    """Generate a batch of prompts concurrently on the V1 engine.

    Each prompt is submitted via ``add_request`` and all are stepped together,
    so the V1 scheduler batches them natively (one shared ``step`` loop covers
    every request). Returns one :class:`VllmGeneration` per prompt, in the
    input order.

    - ``ttft`` — per-request wall time to its first decoded token. Latency is
      never divided by batch size; doing that turns a real request latency into
      an artificial service-time estimate.
    - ``totalTime`` — the batch's shared wall-clock (first submission -> last
      completion) *amortized* over the batch (``wallClock / batchSize``), so
      summing per-sample times equals the actual run wall-clock — what
      :class:`~metrics.Throughput.ThroughputMetric` expects.
    """
    from vllm import SamplingParams

    engine = llm.llm_engine
    requestIds: List[str] = []
    t0 = time.perf_counter()
    for i, prompt in enumerate(promptTexts):
        requestId = f"kvbench-{time.time_ns()}-{i}"
        engine.add_request(
            requestId,
            prompt,
            SamplingParams(temperature=0, max_tokens=maxNewTokens),
        )
        requestIds.append(requestId)

    ttfts: Dict[str, float] = {}
    numCached: Dict[str, int] = {}
    texts: Dict[str, str] = {}
    tokenLens: Dict[str, int] = {}
    finishReasons: Dict[str, Optional[str]] = {}
    stopReasons: Dict[str, Optional[Union[int, str]]] = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.request_id not in requestIds:
                continue
            rid = out.request_id
            if rid not in ttfts and out.outputs and out.outputs[0].token_ids:
                ttfts[rid] = time.perf_counter() - t0  # first token: vllm-bench style
            numCached[rid] = max(
                numCached.get(rid, 0),
                int(getattr(out, "num_cached_tokens", 0) or 0),
            )
            if out.finished:
                outputs = getattr(out, "outputs", None) or []
                if outputs:
                    completion = outputs[0]
                    texts[rid] = completion.text
                    tokenLens[rid] = len(completion.token_ids)
                    # CompletionOutput is the authoritative source. Keep
                    # getattr() for compatibility with older vLLM versions
                    # whose output objects may not expose stop_reason.
                    finishReasons[rid] = getattr(
                        completion, "finish_reason", None
                    )
                    stopReasons[rid] = getattr(completion, "stop_reason", None)
    totalTime = time.perf_counter() - t0
    n = len(requestIds) or 1
    amortized = totalTime / n

    return [
        VllmGeneration(
            text=texts.get(rid, ""),
            ttft=float(ttfts.get(rid, 0.0)),
            numTokens=tokenLens.get(rid, 0),
            totalTime=amortized,
            numCached=numCached.get(rid, 0),
            finishReason=finishReasons.get(rid),
            stopReason=stopReasons.get(rid),
        )
        for rid in requestIds
    ]


def EncodeIds(llm, text: str, *, addSpecialTokens: bool = False) -> List[int]:
    """Tokenize ``text`` with the LLM's own tokenizer."""
    tokenizer = llm.get_tokenizer()
    try:
        return tokenizer.encode(text, add_special_tokens=addSpecialTokens)
    except TypeError:  # transformers >= 5 dropped add_special_tokens from encode
        return tokenizer(text, add_special_tokens=addSpecialTokens)["input_ids"]
