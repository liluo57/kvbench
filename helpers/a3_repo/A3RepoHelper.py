"""JSON-lines worker that calls the official A^3 (ragkv) implementation.

The worker is deliberately separate from :mod:`methods.A3Repo`: ragkv
monkey-patches HuggingFace classes and uses its own flashinfer-backed forward.
This file only translates KVBench's ``collect``/``reuse``/``full`` requests to
the official loader and decoding functions.  It does not contain a second A^3
scorer or fusion algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace


class A3Worker:
    def __init__(self, args):
        self.args = args
        self.repo_root = os.path.abspath(args.repo_root)
        sys.path.insert(0, self.repo_root)

        import torch
        from models.loader import load_model, load_model_precompute

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

        # ragkv's precompute loader temporarily replaces
        # transformers.<Arch>ForCausalLM.  Restore the public class before
        # loading the runtime model so both official model variants coexist.
        import transformers
        arch = self._arch_name(args.model)
        class_name = f"{arch}ForCausalLM"
        original_cls = getattr(transformers, class_name)
        print(f"[a3-repo-helper] loading official precompute model {args.model} ...", flush=True)
        self.precompute_model, self.tokenizer = load_model_precompute(model_args)
        setattr(transformers, class_name, original_cls)
        print(f"[a3-repo-helper] loading official runtime model {args.model} ...", flush=True)
        self.model, runtime_tokenizer = load_model(model_args)
        # The two loaders use the same tokenizer configuration.  Prefer the
        # runtime tokenizer if it differs in special-token metadata.
        self.tokenizer = runtime_tokenizer or self.tokenizer
        self.model.eval()
        self.precompute_model.eval()
        print(_READY_LINE, flush=True)

    @staticmethod
    def _arch_name(model_path: str) -> str:
        lower = model_path.lower()
        if "llama" in lower:
            return "Llama"
        if "mistral" in lower:
            return "Mistral"
        if "qwen" in lower:
            return "Qwen2"
        raise ValueError(f"official ragkv adapter does not recognize model path: {model_path}")

    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _collect_one(self, text: str):
        if text in self.chunk_kv:
            return
        ids = self._encode(text)
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

    def collect(self, chunks):
        for text in chunks or []:
            self._collect_one(str(text))
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
        args = self._sampling_args()
        input_state = {
            "input_ids": self.torch.tensor([ids], device="cuda", dtype=self.torch.long),
            "past_key_values": None,
            "position_ids": self.torch.arange(len(ids), device="cuda").unsqueeze(0),
        }
        stop = [x for x in (self.tokenizer.eos_token_id, self.tokenizer.bos_token_id) if x is not None]
        start = time.perf_counter()
        text, ttft, _tpot = decode(
            args,
            self.model,
            self.tokenizer,
            input_state,
            stop,
            self.args.max_new_tokens,
            config,
        )
        total = time.perf_counter() - start
        n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        return text, float(ttft), float(total), int(n_tokens)

    def _full_config(self):
        return {
            "reuse_config": None,
            "drop_config": None,
            "other_config": {"decode": False},
        }

    def full(self, text):
        ids = self._encode(text)
        output, ttft, total, n_tokens = self._decode(ids, self._full_config())
        return {
            "ok": True,
            "text": output,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(ids),
            "reuse_ratio": 0.0,
        }

    def reuse(self, chunks, suffix):
        if not chunks:
            return self.full(suffix)
        for text in chunks:
            if text not in self.chunk_kv:
                raise ValueError(f"chunk was not collected: {text[:80]!r}")

        chunk_ids = [self.chunk_ids[text] for text in chunks]
        suffix_ids = self._encode(suffix)
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
        output, ttft, total, n_tokens = self._decode(full_ids, config)
        return {
            "ok": True,
            "text": output,
            "ttft": ttft,
            "total_time": total,
            "num_tokens": n_tokens,
            "n_input": len(full_ids),
            "reuse_ratio": round(n_doc / len(full_ids), 6) if full_ids else 0.0,
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
                    response = self.collect(request.get("chunks", []))
                elif op == "reuse":
                    response = self.reuse(request.get("chunks", []), request.get("suffix", ""))
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
