"""Qwen38TestMethod — full prefill over HuggingFace ``transformers`` for Qwen3.8-27B.

Qwen3.8-27B is a Qwen3.5-family hybrid linear+full attention model that ships
as a multimodal checkpoint (``Qwen3_5ForConditionalGeneration``) with a
nested ``text_config``. vLLM skips the vision tower via
``language_model_only=True``; the transformers equivalent is to instantiate
``Qwen3_5ForCausalLM`` (text-only, config class ``Qwen3_5TextConfig``) from
the same path — transformers 5.x resolves the multimodal ``config.json`` to
the text sub-config automatically.

This method is the *user-mandated baseline*: recompute the whole prompt
every query, no reuse, no patching. Use it to verify the model loads cleanly
on a single GPU before plugging in any cacheblend / naive variant. Single
GPU only — Qwen3.8-27B has a hybrid layer layout (linear attention layers
have no per-token KV cache) so chunked prefill is not implemented here;
the goal is parity with :class:`FullPrefillTransformer`, not micro-optimisation.
"""

import os
import time
from typing import Any, List, Optional, Sequence, Tuple

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey


class Qwen38Generator:
    """Owns one Qwen3.8-27B text-only model + tokenizer on a single GPU.

    Mirrors the boundary semantics of :class:`TransformersGenerator` (TTFT
    measured at the first generated token, eager prefill, manual greedy
    decode) so a :class:`Qwen38TestMethod` / :class:`FullPrefillTransformer`
    side-by-side is a like-for-like comparison. The only divergence is the
    multimodal-aware model load: transformers' AutoModelForCausalLM does not
    correctly handle the Qwen3.5 multimodal checkpoint on every version, so
    we use the explicit :class:`Qwen3_5ForCausalLM` class.
    """

    def __init__(
        self,
        modelPath: str,
        gpuId: int = 0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
    ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from transformers import AutoTokenizer

        self._torch = torch
        self.device = f"cuda:{int(gpuId)}"
        torch.cuda.set_device(int(gpuId))

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.model = _LoadQwen38LanguageModel(modelPath, self.device, dtype)
        self.model.eval()

        self.eosId = self.tokenizer.eos_token_id
        self.maxNewTokens = int(maxNewTokens)

    # ------------------------------------------------------------------ utils
    def Encode(self, text: str, *, addSpecialTokens: bool = False) -> List[int]:
        """Encode ``text`` to token ids (no special tokens by default — the
        caller has already rendered the full chat prompt via the model's own
        :func:`apply_chat_template`)."""
        try:
            return self.tokenizer.encode(
                text, add_special_tokens=addSpecialTokens
            )
        except TypeError:  # transformers >= 5 may drop ``add_special_tokens`` from encode()
            return self.tokenizer(text, add_special_tokens=addSpecialTokens)["input_ids"]

    def _toTensor(self, ids: List[int]) -> Any:
        return self._torch.tensor([ids], device=self.device, dtype=self._torch.long)

    def _Forward(self, inputIds: List[int], pastKeyValues=None, *, useCache: bool = True):
        """One model step: prefill (past=None) or a single decode token."""
        kw = dict(use_cache=useCache)
        try:
            return self.model(
                input_ids=self._toTensor(inputIds),
                past_key_values=pastKeyValues,
                **kw,
            )
        except TypeError:  # transformers >= 5 may drop use_cache=
            return self.model(
                input_ids=self._toTensor(inputIds),
                past_key_values=pastKeyValues,
            )

    # ------------------------------------------------------------- generation
    def Generate(
        self, inputIds: List[int]
    ) -> Tuple[str, float, float, int]:
        """Greedy-decode ``inputIds`` until EOS or ``maxNewTokens``.

        Returns ``(text, ttft, totalTime, numOutputTokens)``. TTFT is measured
        as wall time from the start of the call to the first generated token
        (which is the full prefill cost — the same boundary as
        :class:`FullPrefillTransformer`).
        """
        generated: List[int] = []
        past = None
        t0 = time.perf_counter()
        ttft: Optional[float] = None

        with self._torch.no_grad():
            for _ in range(max(1, self.maxNewTokens)):
                if past is None:
                    out = self._Forward(inputIds, None, useCache=True)
                else:
                    # Decode one token, reusing the cached prefix.
                    out = self._Forward([inputIds[-1]], past, useCache=True)
                logits = out.logits[0, -1]
                nxt = int(self._torch.argmax(logits).item())
                if ttft is None:
                    ttft = time.perf_counter() - t0
                generated.append(nxt)
                past = out.past_key_values
                if nxt == self.eosId:
                    break
                inputIds = [nxt]

        total = time.perf_counter() - t0
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, float(ttft or 0.0), total, len(generated)


def _LoadQwen38LanguageModel(modelPath: str, device: str, dtype: str):
    """Load the language-only model from a Qwen3.8 multimodal checkpoint.

    Strategy (mirrors vLLM's ``language_model_only=True``):

    1. Try :class:`Qwen3_5ForCausalLM.from_pretrained(modelPath)` first —
       transformers 5.x resolves the nested ``text_config`` from the
       multimodal ``config.json`` automatically.
    2. If that fails (older transformers, custom checkpoint layout, etc.),
       fall back to loading :class:`Qwen3_5ForConditionalGeneration` and
       extracting ``.language_model`` (which is the same ``Qwen3_5Model``
       used by :class:`Qwen3_5ForCausalLM`). The ``lm_head`` is shared
       across both classes (``Qwen3_5ForConditionalGeneration`` sets
       ``self.lm_head`` from ``text_config.vocab_size``), so for
       text-only inference either path gives identical outputs.

    No actual I/O happens here for text-only loading beyond reading the
    weight shards.
    """
    import torch

    torchDtype = getattr(torch, dtype)
    loadKwargs: dict = dict(
        dtype=torchDtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Preferred path: text-only CausalLM class. transformers 5.x extracts
    # ``text_config`` from the multimodal config.json on the fly.
    try:
        from transformers import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM.from_pretrained(modelPath, **loadKwargs)
    except Exception as primaryErr:  # noqa: BLE001  (multiple transformers versions)
        # Fall back: load multimodal, return its language_model. The
        # ``lm_head`` on the conditional class is built from
        # ``text_config.vocab_size`` and matches the CausalLM head weights.
        from transformers import Qwen3_5ForConditionalGeneration

        mm = Qwen3_5ForConditionalGeneration.from_pretrained(modelPath, **loadKwargs)
        # ``Qwen3_5ForConditionalGeneration.__init__`` builds
        # ``self.language_model = AutoModel.from_config(config.text_config)``.
        # For text-only inference we want the CausalLM wrapper (= language
        # model + tied lm_head) so the .generate() / .forward() API matches
        # the preferred path. If the wrapper isn't directly available we
        # fall through to the raw language_model and reuse the mm head.
        try:
            from transformers import Qwen3_5ForCausalLM

            textOnly = Qwen3_5ForCausalLM(mm.config.text_config)
            textOnly.model = mm.language_model
            textOnly.lm_head = mm.lm_head
            return textOnly.to(device)
        except Exception:  # noqa: BLE001
            return mm.language_model.to(device)


class Qwen38TestMethod(Method):
    """Full prefill over transformers, single GPU, for Qwen3.8-27B.

    Equivalent to :class:`FullPrefillTransformer` but with Qwen3.5-family
    multimodal-aware loading (skips the vision tower). ``Prepare`` is a
    no-op (full recompute); ``Run`` is request-sequential because the
    KVCOMM protocol is request-sequential, so ``maxCaseBatchSize = 1``.
    """

    name = "qwen38_test"
    backend = "transformers"
    method_metrics: tuple[str, ...] = ()
    maxCaseBatchSize: Optional[int] = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        # Model path is config-only — switch models via config.yaml.
        self.modelPath = DefaultModelPath()
        self.maxNewTokens = maxNewTokens
        self.dtype = dtype
        self._gen: Optional[Qwen38Generator] = None

    # ---------------------------------------------------------------- Method
    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._gen = Qwen38Generator(
            self.modelPath,
            self.gpuIds[0],
            maxNewTokens=self.maxNewTokens,
            dtype=self.dtype,
        )

    def Prepare(self, data: List[List[str]]) -> None:
        pass  # Full prefill; nothing to warm up

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        # Request-sequential: KVCOMM-style tasks call Run with batchSize=1.
        # Running through self._gen.Generate keeps the manual prefill-then-
        # decode loop with a real TTFT measurement (model.generate() can
        # only approximate it via total time / batchSize).
        results: List[Result] = []
        for prompt in data:
            ids = self._gen.Encode(prompt)
            text, ttft, total, nTokens = self._gen.Generate(ids)
            results.append(
                Result(
                    output=text,
                    performance={
                        TtftKey: ttft,
                        NumOutputTokensKey: nTokens,
                        TotalTimeKey: total,
                    },
                    metadata={
                        "backend": self.backend,
                        "n_input": len(ids),
                    },
                )
            )
        return results

    def Close(self) -> None:
        if hasattr(self, "_gen"):
            self._gen = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
