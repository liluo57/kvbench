"""JSON-lines worker that calls the official A^3 (ragkv) implementation.

The worker is deliberately separate from :mod:`methods.A3Repo`: ragkv
monkey-patches HuggingFace classes and uses its own flashinfer-backed forward.
This file only translates KVBench's ``collect``/``reuse``/``full`` requests to
the official loader and decoding functions.  It does not contain a second A^3
scorer or fusion algorithm.  The precompute and runtime module graphs are
kept as separate objects, but their parameters and rotary buffers are aliased
to one checkpoint storage so the worker does not keep two GPU weight copies.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


class A3Worker:
    def __init__(self, args):
        self.args = args
        self.repo_root = os.path.abspath(args.repo_root)
        sys.path.insert(0, self.repo_root)

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

        # Qwen3 is not part of the old ragkv checkout or its Transformers
        # environment.  Install the out-of-tree bridge before importing the
        # official loader; Llama/Mistral/Qwen2 continue through ragkv's native
        # functions unchanged.
        arch = self._arch_name(args.model)
        if arch == "Qwen3":
            kvbench_root = str(Path(__file__).resolve().parents[2])
            if kvbench_root not in sys.path:
                sys.path.insert(0, kvbench_root)
            from helpers.a3_repo.Qwen3ForA3Repo import install_qwen3

            _, load_model_precompute = install_qwen3(self.repo_root)
        else:
            from models.loader import load_model_precompute

        # ragkv's precompute loader temporarily replaces
        # transformers.<Arch>ForCausalLM.  Save the original runtime class,
        # then restore it after precompute loading.  The two official model
        # variants have the same parameter layout but different forward
        # graphs; we instantiate the runtime graph on ``meta`` and bind its
        # parameters/buffers to the already-loaded precompute tensors below.
        # This deliberately avoids a second ``from_pretrained`` call (and a
        # second GPU copy of the checkpoint).
        import transformers
        class_name = f"{arch}ForCausalLM"
        runtime_cls = getattr(transformers, class_name)
        print(f"[a3-repo-helper] loading official precompute model {args.model} ...", flush=True)
        self.precompute_model, self.tokenizer = load_model_precompute(model_args)
        setattr(transformers, class_name, runtime_cls)

        # Native ragkv loaders apply these monkey-patches inside
        # ``load_model``.  Since we intentionally skip its second
        # ``from_pretrained`` call, apply the same process-local patch here.
        # Qwen3's out-of-tree bridge already installs its runtime methods.
        if arch != "Qwen3":
            from models.monkeypatch import replace_llama, replace_mistral, replace_qwen

            patch_runtime = {
                "Llama": replace_llama,
                "Mistral": replace_mistral,
                "Qwen2": replace_qwen,
            }.get(arch)
            if patch_runtime is None:
                raise RuntimeError(f"no official runtime patch registered for {arch}")
            patch_runtime()

        print("[a3-repo-helper] constructing official runtime graph with shared weights ...", flush=True)
        # Construct only the module graph on meta.  No parameter storage is
        # allocated here; _share_model_tensors() aliases every parameter and
        # buffer to the precompute model's already-loaded storage.
        with self.torch.device("meta"):
            self.model = runtime_cls(self.precompute_model.config)
        self._share_model_tensors(self.precompute_model, self.model)
        self._install_runtime_compat_shims(arch)
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
        source_params = dict(source.named_parameters())
        target_params = dict(target.named_parameters())
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

        source_buffers = dict(source.named_buffers())
        target_buffers = dict(target.named_buffers())
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

    @staticmethod
    def _install_runtime_compat_shims(arch: str) -> None:
        """Bridge the checked-out ragkv Mistral call to current utilities.

        The official Mistral file calls ``create_flashinfer_mask`` with three
        positional arguments, while the shared utility in this checkout takes
        a fourth causal-mode argument.  Patch only the imported module global
        in this worker process; the ragkv source tree is never modified.
        """
        if arch != "Mistral":
            return
        import models.mistral.mistral as ragkv_mistral

        if getattr(ragkv_mistral.create_flashinfer_mask, "_kvbench_compat", False):
            return
        from models.reuse_utils import create_flashinfer_mask as create_mask

        def compat(query, key, indices, mode=True):
            return create_mask(query, key, indices, mode)

        compat._kvbench_compat = True
        ragkv_mistral.create_flashinfer_mask = compat

    @staticmethod
    def _arch_name(model_path: str) -> str:
        config_path = Path(model_path).expanduser() / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                architectures = config.get("architectures", []) or []
                model_type = str(config.get("model_type", "")).lower()
                if any("Qwen3" in str(name) for name in architectures) or model_type == "qwen3":
                    return "Qwen3"
            except (OSError, json.JSONDecodeError):
                pass
        lower = model_path.lower()
        if "llama" in lower:
            return "Llama"
        if "mistral" in lower:
            return "Mistral"
        if "qwen3" in lower:
            return "Qwen3"
        if "qwen" in lower:
            return "Qwen2"
        raise ValueError(f"official ragkv adapter does not recognize model path: {model_path}")

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _collect_one(self, text: str, ids=None):
        if text in self.chunk_kv:
            return
        ids = list(ids) if ids is not None else self._encode(text)
        if not ids:
            return
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
        return {"ok": True, "n_chunks": len(self.chunk_kv)}

    def _sampling_args(self):
        # Official ragkv's decode helper only reads max token count and does
        # greedy argmax decoding itself.  Keep this namespace compatible with
        # its utils.decode/decode_step functions.
        return SimpleNamespace(
            model=self.args.model,
            reuse=self.args.reuse_method,
            drop="False",
            drop_config="None",
            rate=self.args.recomp_ratio,
        )

    def _decode(self, ids, config):
        from utils import decode

        if not ids:
            raise ValueError("empty prompt")
        decodeCallStart = time.perf_counter()
        args = self._sampling_args()
        input_state = {
            "input_ids": self.torch.tensor([ids], device="cuda", dtype=self.torch.long),
            "past_key_values": None,
            "position_ids": self.torch.arange(len(ids), device="cuda").unsqueeze(0),
        }
        stop = [x for x in (self.tokenizer.eos_token_id, self.tokenizer.bos_token_id) if x is not None]
        # The parent adapter combines this backend boundary with its own
        # Method.Run timestamp.  That includes IPC and all worker-side prompt,
        # cache, and config preparation exactly once.
        generationStart = time.perf_counter()
        # ragkv's decode() prints intermediate/generated text to stdout.
        # Keep that diagnostic stream away from KVBench's JSON-lines protocol;
        # otherwise a numeric-only answer (for example ``2009``) is parsed as
        # a JSON integer by the parent adapter instead of as its response.
        with contextlib.redirect_stdout(sys.stderr):
            text, ttft, _tpot = decode(
                args,
                self.model,
                self.tokenizer,
                input_state,
                stop,
                self.args.max_new_tokens,
                config,
            )
        total = time.perf_counter() - decodeCallStart
        n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        return (
            text,
            float(ttft),
            float(total),
            int(n_tokens),
            generationStart,
        )

    def _full_config(self):
        return {
            "reuse_config": None,
            "drop_config": None,
            "other_config": {"decode": False},
        }

    def full(self, text):
        """Run a complete prefill through the same patched A3 runtime.

        This is the worker's fallback for prompts without reusable chunks; it
        is not the paper's selective-reuse path.  ``_full_config`` keeps
        ``reuse_config`` empty while still passing ragkv's config object, so
        the monkey-patched model and FlashInfer kernel remain active.  The
        dedicated ``OfficialVanillaHelper`` uses the same runtime for a clean
        baseline process and calls ragkv's ``vanilla()`` entry point.
        """
        ids = self._encode(text)
        output, ttft, total, n_tokens, generationStart = self._decode(
            ids, self._full_config()
        )
        return {
            "ok": True,
            "text": output,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(ids),
            "generation_start": generationStart,
            "reuse_ratio": 0.0,
            "runtime_mode": "ragkv_official_patched",
            "algorithm": "full_recompute",
        }

    def reuse(self, chunks, suffix, suffix_ids=None):
        if not chunks:
            return self.full(suffix)
        for text in chunks:
            if text not in self.chunk_kv:
                raise ValueError(f"chunk was not collected: {text[:80]!r}")

        chunk_ids = [self.chunk_ids[text] for text in chunks]
        suffix_ids = list(suffix_ids) if suffix_ids is not None else self._encode(suffix)
        doc_ids = [token for ids in chunk_ids for token in ids]
        full_ids = doc_ids + suffix_ids
        if not full_ids:
            raise ValueError("empty A^3 prompt")

        n_layers = len(self.model.model.layers)
        n_doc = len(doc_ids)
        n_query = len(suffix_ids)
        sample = self.chunk_kv[chunks[0]][0]
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
        output, ttft, total, n_tokens, generationStart = self._decode(
            full_ids, config
        )
        return {
            "ok": True,
            "text": output,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(full_ids),
            "generation_start": generationStart,
            "reuse_ratio": round(n_doc / len(full_ids), 6) if full_ids else 0.0,
            "runtime_mode": "ragkv_official_patched",
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
                    )
                elif op == "full":
                    response = self.full(request.get("text", ""))
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
    parser.add_argument("--repo_root", required=True)
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
