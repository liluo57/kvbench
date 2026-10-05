"""JSON-lines worker for the bundled A^3 (ragkv) implementation.

The worker is deliberately separate from :mod:`methods.A3Repo`: the backend
uses its own FlashInfer forward and model/cache state.
This file only translates KVBench's ``collect``/``reuse``/``full`` requests to
the official loader and decoding functions.  It does not contain a second A^3
scorer or fusion algorithm.  The precompute and runtime module graphs are
kept as separate objects, but their parameters and rotary buffers are aliased
to one checkpoint storage so the worker does not keep two GPU weight copies.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Make direct script launches independent of the caller's working directory.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "methods.A3Repo"


class A3Worker:
    def __init__(self, args):
        self.args = args

        import torch

        self.torch = torch
        try:
            self.sampling_config = json.loads(args.sampling_config or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid --sampling_config JSON") from exc

        self.tokenizer = None
        self.precompute_model = None
        self.model = None
        self.chunk_ids = {}
        self.chunk_kv = {}

        model_args = SimpleNamespace(
            model=args.model,
            reuse=args.reuse_method,
            drop="False",
            drop_config="None",
            rate=args.recomp_ratio,
        )

        from .backend.models.loader import load_model_precompute, model_classes

        print(f"[a3-repo-helper] loading precompute model {args.model} ...", flush=True)
        self.precompute_model, self.tokenizer = load_model_precompute(model_args)
        runtime_cls, _ = model_classes(self.precompute_model.config)

        print("[a3-repo-helper] constructing official runtime graph with shared weights ...", flush=True)
        # Construct only the module graph on meta.  No parameter storage is
        # allocated here; _share_model_tensors() aliases every parameter and
        # buffer to the precompute model's already-loaded storage.
        with self.torch.device("meta"):
            self.model = runtime_cls(self.precompute_model.config)
        self._share_model_tensors(self.precompute_model, self.model)
        self.model.eval()
        self.precompute_model.eval()
        print(_READY_LINE, flush=True)

    @staticmethod
    def _replace_named_tensor(root, name, tensor) -> None:
        """Replace a dotted parameter/buffer attribute without copying data."""
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], tensor)

    @classmethod
    def _share_model_tensors(cls, source, target) -> None:
        """Alias target tensors to source tensors after strict layout checks.

        A3's precompute and runtime classes intentionally have different
        forward implementations, but for the supported dense architectures
        their trainable parameter and rotary-buffer layouts must match.  We
        fail loudly on any mismatch instead of silently falling back to a
        separately initialized or partially shared model.
        """
        # Include every alias: replacing embed_tokens.weight must also replace
        # lm_head.weight for checkpoints with tied embeddings. Otherwise the
        # runtime head still points at its original meta tensor.
        source_params = dict(source.named_parameters(remove_duplicate=False))
        target_params = dict(target.named_parameters(remove_duplicate=False))
        if set(source_params) != set(target_params):
            missing = sorted(set(source_params) - set(target_params))
            extra = sorted(set(target_params) - set(source_params))
            raise RuntimeError(
                "A3 shared-weight parameter layout mismatch: "
                f"missing_in_runtime={missing[:8]}, extra_in_runtime={extra[:8]}"
            )
        for name, source_param in source_params.items():
            target_param = target_params[name]
            if tuple(source_param.shape) != tuple(target_param.shape):
                raise RuntimeError(
                    f"A3 shared-weight shape mismatch for {name}: "
                    f"precompute={tuple(source_param.shape)} "
                    f"runtime={tuple(target_param.shape)}"
                )
            cls._replace_named_tensor(target, name, source_param)

        source_buffers = dict(source.named_buffers(remove_duplicate=False))
        target_buffers = dict(target.named_buffers(remove_duplicate=False))
        if set(source_buffers) != set(target_buffers):
            missing = sorted(set(source_buffers) - set(target_buffers))
            extra = sorted(set(target_buffers) - set(source_buffers))
            raise RuntimeError(
                "A3 shared-weight buffer layout mismatch: "
                f"missing_in_runtime={missing[:8]}, extra_in_runtime={extra[:8]}"
            )
        for name, source_buffer in source_buffers.items():
            target_buffer = target_buffers[name]
            if tuple(source_buffer.shape) != tuple(target_buffer.shape):
                raise RuntimeError(
                    f"A3 shared-weight buffer shape mismatch for {name}: "
                    f"precompute={tuple(source_buffer.shape)} "
                    f"runtime={tuple(target_buffer.shape)}"
                )
            cls._replace_named_tensor(target, name, source_buffer)
        for module in target.modules():
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = module.inv_freq

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _collect_one(self, text: str, ids=None):
        if text in self.chunk_kv:
            return
        ids = list(ids) if ids is not None else self._encode(text)
        if not ids:
            return
        if len(ids) > self.args.max_model_len:
            raise ValueError(f"A3 prepared chunk exceeds max_model_len={self.args.max_model_len}")
        input_ids = self.torch.tensor([ids], device="cuda", dtype=self.torch.long)
        with self.torch.no_grad():
            self.precompute_model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=None,
            )
        layers = []
        for layer in self.precompute_model.model.layers:
            captured = getattr(layer.self_attn, "hack_kv", None)
            if captured is None:
                raise RuntimeError("official precompute model did not expose hack_kv")
            key, value = captured
            layers.append([key[0].detach().clone(), value[0].detach().clone()])
            layer.self_attn.hack_kv = None
        self.chunk_ids[text] = ids
        self.chunk_kv[text] = layers

    def collect(self, chunks, chunk_ids=None):
        chunks = chunks or []
        chunk_ids = chunk_ids or []
        if chunk_ids and len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids must align one-to-one with chunks")
        for index, text in enumerate(chunks):
            ids = chunk_ids[index] if chunk_ids else None
            self._collect_one(str(text), ids=ids)
        # Precompute and the captured KV clones enqueue CUDA work. Finish it
        # at the offline Prepare boundary so Run's TTFT cannot include it.
        self.torch.cuda.synchronize()
        return {"ok": True, "n_chunks": len(self.chunk_kv)}

    def _sampling_args(self):
        # Keep this namespace compatible with ragkv's decode argument surface;
        # KVBench supplies the resolved policy through decode's optional hook.
        return SimpleNamespace(
            model=self.args.model,
            reuse=self.args.reuse_method,
            drop="False",
            drop_config="None",
            rate=self.args.recomp_ratio,
        )

    def _sampling_policy(self):
        from .backend.utils import resolve_sampling_policy
        return resolve_sampling_policy(self.sampling_config)

    def _stop_ids(self):
        from .backend.utils import sampling_stop_ids
        return sampling_stop_ids(self.tokenizer, self.sampling_config)

    def _decode(self, ids, config, max_new_tokens=None, retain_output=False):
        from .backend.utils import decode
        from helpers.backends.Tokenization import OutputFragmentIds

        if not ids:
            raise ValueError("empty prompt")
        if len(ids) >= self.args.max_model_len:
            raise ValueError(f"A3 prompt must leave room for generation within max_model_len={self.args.max_model_len}")
        if retain_output:
            for index, layer in enumerate(self.model.model.layers):
                rotary = getattr(layer.self_attn, "rotary_emb", None)
                rope_type = str(getattr(rotary, "rope_type", "default")).lower()
                if rotary is None or "dynamic" in rope_type:
                    raise RuntimeError(
                        f"retain_output requires static invertible RoPE; layer {index} uses {rope_type!r}"
                    )
        budget = min(
            self.args.max_new_tokens if max_new_tokens is None else max_new_tokens,
            self.args.max_model_len - len(ids),
        )
        decodeCallStart = time.perf_counter()
        args = self._sampling_args()
        sampling = self._sampling_policy()
        device = getattr(self.model, "device", "cuda")
        input_state = {
            "input_ids": self.torch.tensor([ids], device=device, dtype=self.torch.long),
            "past_key_values": None,
            "position_ids": self.torch.arange(len(ids), device=device).unsqueeze(0),
        }
        stop = self._stop_ids()
        # The parent adapter combines this backend boundary with its own
        # Method.Run timestamp.  That includes IPC and all worker-side prompt,
        # cache, and config preparation exactly once.
        generationStart = time.perf_counter()
        # ragkv's decode() prints intermediate/generated text to stdout.
        # Keep that diagnostic stream away from KVBench's JSON-lines protocol;
        # otherwise a numeric-only answer (for example ``2009``) is parsed as
        # a JSON integer by the parent adapter instead of as its response.
        with contextlib.redirect_stdout(sys.stderr):
            decoded = decode(
                args,
                self.model,
                self.tokenizer,
                input_state,
                stop,
                budget,
                config,
                return_token_ids=True,
                sampling=sampling,
                return_cache=retain_output,
            )
        if retain_output:
            text, ttft, _tpot, generatedIds, cache = decoded
        else:
            text, ttft, _tpot, generatedIds = decoded
        total = time.perf_counter() - decodeCallStart
        result = (
            text,
            float(ttft),
            float(total),
            len(generatedIds),
            generationStart,
            OutputFragmentIds(generatedIds, stop),
        )
        return result + ((cache,) if retain_output else ())

    def _retain_output(self, output_ids, cache, prompt_len):
        from .backend.utils import recover_pre_rope_keys
        from helpers.backends.TransformersHelper import CacheLayerPairs

        if not output_ids:
            return
        cache = list(CacheLayerPairs(cache))
        output_len = len(output_ids)
        start, end = prompt_len, prompt_len + output_len
        if cache is None or len(cache) != len(self.model.model.layers):
            raise RuntimeError("A3 runtime cache is unavailable for retain_output")
        layers = []
        positions = self.torch.arange(start, end, device=cache[0][0].device).unsqueeze(0)
        for index, (key, value) in enumerate(cache):
            if key.shape[-2] != end or value.shape[-2] != end:
                raise RuntimeError(
                    "A3 runtime cache length does not match prompt plus retained output: "
                    f"K={key.shape[-2]}, V={value.shape[-2]}, expected={end}"
                )
            attention = self.model.model.layers[index].self_attn
            raw_key = recover_pre_rope_keys(key[:, :, start:end, :], attention.rotary_emb, positions)
            layers.append([raw_key[0].detach().clone(), value[0, :, start:end, :].detach().clone()])
        name = "ids:" + hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in output_ids)
        ).hexdigest()
        self.chunk_ids[name] = list(output_ids)
        self.chunk_kv[name] = layers

    def _full_config(self):
        return {
            "reuse_config": None,
            "drop_config": None,
            "other_config": {"decode": False},
        }

    def full(self, text=None, input_ids=None, max_new_tokens=None, retain_output=False):
        """Run a complete prefill through the same patched A3 runtime.

        This is the worker's fallback for prompts without reusable chunks; it
        is not the paper's selective-reuse path.  ``_full_config`` keeps
        ``reuse_config`` empty while still passing ragkv's config object, so
        the bundled model and FlashInfer kernel remain active.  The
        dedicated ``VanillaWorker`` uses the same runtime for a clean
        baseline process and calls ragkv's ``vanilla()`` entry point.
        """
        ids = list(input_ids) if input_ids is not None else self._encode(text or "")
        decoded = self._decode(
            ids, self._full_config(), max_new_tokens, retain_output
        )
        if retain_output:
            output, ttft, total, n_tokens, generationStart, outputIds, cache = decoded
            retain_start = time.perf_counter()
            self._retain_output(outputIds, cache, len(ids))
            total += time.perf_counter() - retain_start
        else:
            output, ttft, total, n_tokens, generationStart, outputIds = decoded
        return {
            "ok": True,
            "text": output,
            "output_ids": outputIds,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(ids),
            "generation_start": generationStart,
            "reuse_ratio": 0.0,
            "runtime_mode": "ragkv_bundled",
            "algorithm": "full_recompute",
        }

    def reuse(self, chunks, suffix, suffix_ids=None, max_new_tokens=None, retain_output=False):
        if not chunks:
            return self.full(suffix, suffix_ids, max_new_tokens, retain_output)
        for text in chunks:
            if text not in self.chunk_kv:
                raise ValueError(f"chunk was not collected: {text[:80]!r}")

        chunk_ids = [self.chunk_ids[text] for text in chunks]
        suffix_ids = list(suffix_ids) if suffix_ids is not None else self._encode(suffix)
        doc_ids = [token for ids in chunk_ids for token in ids]
        full_ids = doc_ids + suffix_ids
        if not full_ids:
            raise ValueError("empty A^3 prompt")
        if len(full_ids) >= self.args.max_model_len:
            raise ValueError(f"A3 prompt must leave room for generation within max_model_len={self.args.max_model_len}")

        n_layers = len(self.model.model.layers)
        n_doc = len(doc_ids)
        n_query = len(suffix_ids)
        cat_kv = []
        for kind in (0, 1):
            per_layer = []
            for layer_index in range(n_layers):
                pieces = [self.chunk_kv[text][layer_index][kind] for text in chunks]
                if n_query:
                    pieces.append(pieces[0].new_zeros((pieces[0].shape[0], n_query, pieces[0].shape[-1])))
                per_layer.append(self.torch.cat(pieces, dim=1))
            cat_kv.append(self.torch.stack(per_layer))
        cat_kv = self.torch.stack(cat_kv)

        # The official A^3 debug selector is the attention-derived selector
        # used by the repository scripts.  Its ``last_len`` is the fresh
        # query suffix and ``prefix_len`` marks the beginning of the reusable
        # document region.  We use zero because KVBench's prepared chunks are
        # already the reusable region.
        config = {
            "reuse_config": {
                "cat_kv": cat_kv,
                "check": None,
                "fake_q": None,
                "mask": None,
                "recomp_ratio": self.args.recomp_ratio,
                "reuse": self.args.reuse_method,
                "causal": True,
            },
            "drop_config": None,
            "other_config": {
                "decode": False,
                "data_params": {
                    "prefix_len": 0,
                    "last_len": n_query,
                    "doc_start_len": 0,
                },
            },
        }
        decoded = self._decode(full_ids, config, max_new_tokens, retain_output)
        if retain_output:
            output, ttft, total, n_tokens, generationStart, outputIds, cache = decoded
            retain_start = time.perf_counter()
            self._retain_output(outputIds, cache, len(full_ids))
            total += time.perf_counter() - retain_start
        else:
            output, ttft, total, n_tokens, generationStart, outputIds = decoded
        return {
            "ok": True,
            "text": output,
            "output_ids": outputIds,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(full_ids),
            "generation_start": generationStart,
            "reuse_ratio": n_doc / len(full_ids) if full_ids else 0.0,
            "runtime_mode": "ragkv_bundled",
            "algorithm": "a3_reuse",
            "a3_debug": {
                "reuse_method": self.args.reuse_method,
                "doc_tokens": n_doc,
                "query_tokens": n_query,
            },
        }

    def reset(self):
        self.chunk_ids.clear()
        self.chunk_kv.clear()
        return {"ok": True}

    def serve(self):
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                op = request.get("op")
                if op == "collect":
                    response = self.collect(request.get("chunks", []), request.get("chunk_ids"))
                elif op == "reuse":
                    response = self.reuse(
                        request.get("chunks", []),
                        request.get("suffix", ""),
                        request.get("suffix_ids"),
                        request.get("max_new_tokens"),
                        request.get("retain_output", False),
                    )
                elif op == "full":
                    response = self.full(
                        request.get("text", ""), request.get("input_ids"),
                        request.get("max_new_tokens"),
                        request.get("retain_output", False),
                    )
                elif op == "reset":
                    response = self.reset()
                elif op == "close":
                    _write({"ok": True})
                    return
                else:
                    response = {"ok": False, "error": f"unknown op {op!r}"}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            _write(response)


_READY_LINE = "[a3-repo-helper] ready"


def _write(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--recomp_ratio", type=float, default=0.15)
    parser.add_argument("--reuse_method", default="debug")
    parser.add_argument("--sampling_config", default="{}")
    args = parser.parse_args()
    A3Worker(args).serve()


if __name__ == "__main__":
    main()
