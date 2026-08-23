"""Plain HuggingFace ``transformers`` backend, used by the transformer methods.

The ``FullPrefillTransformer`` / ``NaiveTransformer`` methods own one
:class:`TransformersGenerator` each. It exposes a manual greedy decode loop
that measures a real TTFT (prefill + first decode step).

GPU placement: multiple transformers methods may share one process (one model
each). ``CUDA_VISIBLE_DEVICES`` only matters at torch's first init and a later
env change cannot move an already-initialized torch — so instead of the env var
we keep all GPUs visible and pin each instance with ``torch.cuda.set_device``
and an explicit ``device_map=f"cuda:<n>"``.
"""

import os
import time
from typing import Any, List, Optional, Tuple


def CacheLayerPairs(cache):
    """Yield ``(K, V)`` per layer from a ``past_key_values`` of any version.

    ``transformers >= 5`` returns a ``DynamicCache`` whose ``layers`` expose
    ``.keys`` / ``.values``; older versions return a tuple of ``(K, V)`` pairs.
    """
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if hasattr(layer, "keys"):
                yield layer.keys, layer.values
            else:
                yield layer[0], layer[1]
    else:
        for layer in cache:
            yield layer[0], layer[1]


def FirstGpu(gpuIds: Any) -> Optional[int]:
    """First GPU index from a ``gpu_ids`` str / int / list (None if unset)."""
    if gpuIds is None:
        return None
    if isinstance(gpuIds, (list, tuple)):
        return int(gpuIds[0]) if gpuIds else None
    return int(gpuIds)


class TransformersGenerator:
    """Owns one HF model + tokenizer and runs a manual greedy decode loop.

    Generation is stepped token by token so the time to the *first* generated
    token (TTFT) can be measured independently of the full decode.
    """

    def __init__(
        self,
        modelPath: str,
        gpuIds: Any = "0",
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.maxNewTokens = maxNewTokens
        #: max tokens per eager prefill step (chunked prefill for long inputs).
        self.prefillChunk = int(os.environ.get("KVBENCH_PREFILL_CHUNK", "2048"))

        first = FirstGpu(gpuIds)
        # ``device_map="cuda"`` resolves to ``cuda:0`` in newer transformers
        # regardless of the current device, so target ``cuda:<n>`` explicitly.
        self.device = f"cuda:{int(first)}" if first is not None else device
        if first is not None:
            torch.cuda.set_device(int(first))

        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        loadKwargs: dict = dict(
            dtype=getattr(torch, dtype), device_map=self.device, low_cpu_mem_usage=True
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(modelPath, **loadKwargs)
        except TypeError:  # older transformers (< 5) uses torch_dtype=
            self.model = AutoModelForCausalLM.from_pretrained(
                modelPath,
                torch_dtype=getattr(torch, dtype),
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
        self.model.eval()
        self.eosId = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ utils
    def Encode(self, text: str, *, addSpecialTokens: bool = False) -> List[int]:
        try:
            return self.tokenizer.encode(
                text, add_special_tokens=addSpecialTokens
            )
        except TypeError:  # transformers >= 5
            return self.tokenizer(
                text, add_special_tokens=addSpecialTokens
            )["input_ids"]

    def _toTensor(self, ids: List[int]) -> Any:
        return self._torch.tensor([ids], device=self.device, dtype=self._torch.long)

    def Forward(self, inputIds: List[int], pastKeyValues=None, *, useCache: bool = True):
        """One model step: prefill (past=None) or a single decode token."""
        inputTensor = self._toTensor(inputIds)
        kw = dict(use_cache=useCache)
        try:
            return self.model(
                input_ids=inputTensor, past_key_values=pastKeyValues, **kw
            )
        except TypeError:  # transformers >= 5 may drop use_cache
            return self.model(
                input_ids=inputTensor, past_key_values=pastKeyValues
            )

    def Prefill(self, inputIds: List[int], pastKeyValues=None):
        """Prefill ``inputIds``, chunking long inputs to bound peak memory.

        ``pastKeyValues`` may contain an already assembled prefix. New tokens are
        prefetched after that prefix and extend the same cache.

        Without a flash-attention backend, one big eager prefill materializes
        the full ``seq x seq`` attention matrix (~tens of GB for a 7.5k prompt).
        Feeding ≤``KVBENCH_PREFILL_CHUNK`` tokens per forward keeps the
        per-step attention matrix small.
        """
        with self._torch.no_grad():
            n = len(inputIds)

            if n <= self.prefillChunk:
                return self.Forward(
                    inputIds,
                    pastKeyValues,
                    useCache=True,
                )

            past = pastKeyValues
            out = None

            for i in range(0, n, self.prefillChunk):
                out = self.Forward(
                    inputIds[i: i + self.prefillChunk],
                    past,
                    useCache=True,
                )
                past = out.past_key_values

            return out

    # ------------------------------------------------------------- generation
    def Generate(
        self,
        inputIds: List[int],
        pastKeyValues=None,
        *,
        maxNewTokens: Optional[int] = None,
        returnCache: bool = False,
    ) -> Tuple[Any, ...]:
        """Greedy-decode ``inputIds``.

        Returns ``(text, ttft, totalTime, numOutputTokens)``.
        When ``pastKeyValues`` is given (naive reuse), only the suffix is
        fed to the model — the cached context KV is never recomputed.
        """
        maxNew = self.maxNewTokens if maxNewTokens is None else maxNewTokens
        generated: List[int] = []
        past = pastKeyValues
        t0 = time.perf_counter()
        ttft: Optional[float] = None

        with self._torch.no_grad():
            for _ in range(max(1, maxNew)):
                if past is None:
                    out = self.Prefill(inputIds)
                else:
                    out = self.Forward(inputIds, past, useCache=True)
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
        result = (text, float(ttft or 0.0), total, len(generated))
        if returnCache:
            return (*result, past, generated)
        return result

    def GenerateBatch(
        self,
        inputIdsList: List[List[int]],
        *,
        maxNewTokens: Optional[int] = None,
    ) -> List[Tuple[str, float, float, int]]:
        """Greedy-decode a batch of prompts in one ``model.generate`` call.

        Inputs are left-padded to a common length so the single GPU call sees a
        real batch (better utilization than N sequential calls). Returns one
        ``(text, ttft, totalTime, numOutputTokens)`` per prompt, in input order.

        ``model.generate`` does not expose per-token timings, so ``ttft`` is
        approximated as ``totalTime / batchSize``. ``totalTime`` is the batch's
        shared wall-clock *amortized* over the batch
        (``wallClock / batchSize``), so summing per-sample times equals the
        actual run wall-clock — what :class:`~metrics.Throughput.ThroughputMetric`
        expects.
        """
        maxNew = self.maxNewTokens if maxNewTokens is None else maxNewTokens
        padId = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        maxLen = max(len(ids) for ids in inputIdsList)
        batch = self._torch.full(
            (len(inputIdsList), maxLen), padId, dtype=self._torch.long, device=self.device
        )
        mask = self._torch.zeros(
            (len(inputIdsList), maxLen), dtype=self._torch.long, device=self.device
        )
        for i, ids in enumerate(inputIdsList):
            batch[i, -len(ids):] = self._torch.tensor(
                ids, dtype=self._torch.long, device=self.device
            )
            mask[i, -len(ids):] = 1

        t0 = time.perf_counter()
        with self._torch.no_grad():
            out = self.model.generate(
                batch,
                attention_mask=mask,
                max_new_tokens=maxNew,
                pad_token_id=padId,
                do_sample=False,
            )
        total = time.perf_counter() - t0
        amortized = total / len(inputIdsList) if inputIdsList else 0.0

        results: List[Tuple[str, float, float, int]] = []
        for i, seq in enumerate(out):
            gen = seq[mask[i].sum():]  # strip the left-padded input
            nTokens = int(gen.numel())
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            results.append((text, amortized, amortized, nTokens))
        return results
