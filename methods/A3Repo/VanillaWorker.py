"""Thin JSON-lines worker for ragkv's official vanilla baseline.

This helper is deliberately separate from :mod:`Worker`.  The A3
worker loads ragkv's reuse/precompute model and calls the official
``utils.decode`` path; this worker loads the same bundled backend and model with the
same bundled runtime mode, but calls ragkv's own ``utils.vanilla`` function
with an empty reuse configuration.  Keeping the workers in separate processes
prevents state from leaking between methods while preserving an identical
runtime, kernel implementation, tokenizer, dtype, and timing boundary.

The protocol is intentionally small and mirrors the CacheBlend helpers::

    {"op": "run", "text": "..."}
    {"op": "close"}

Only protocol responses are written to stdout after the ``ready`` marker.
The official vanilla function prints generated text itself, so that output is
captured and redirected to stderr rather than corrupting the JSON stream.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "methods.A3Repo"


_READY_LINE = "[official-vanilla-helper] ready"


def _write(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class OfficialVanillaWorker:
    """Call the bundled upstream vanilla() implementation in a subprocess."""

    def __init__(self, args: argparse.Namespace):
        # Heavy imports stay inside this isolated worker, just like the
        # CacheBlend and A3 repository helpers.
        self.args = args

        try:
            self.sampling_config = json.loads(args.sampling_config or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid --sampling_config JSON") from exc

        import torch

        from .backend.models.loader import load_model

        self.torch = torch
        model_args = SimpleNamespace(
            model=args.model,
            # The loader always selects the bundled runtime; vanilla() gets
            # reuse_config=None below, preserving a matching FlashInfer path.
            reuse=args.runtime_reuse,
            drop="False",
            drop_config="None",
            rate=0.0,
        )
        print(f"[official-vanilla-helper] loading model {args.model} ...", flush=True)
        # This is the repository's normal loader with the same bundled
        # runtime mode used by Worker.  No KV is reused: the config passed
        # to vanilla() contains ``reuse_config=None``.
        self.model, self.tokenizer = load_model(model_args)
        self.model.eval()
        print(_READY_LINE, flush=True)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _sampling_args(self) -> SimpleNamespace:
        # ragkv.utils.vanilla reads the same small argument surface as decode.
        return SimpleNamespace(
            model=self.args.model,
            reuse=self.args.runtime_reuse,
            drop="False",
            drop_config="None",
            rate=0.0,
        )

    def run(self, text: str, input_ids=None) -> dict:
        from .backend.utils import vanilla

        onlineStart = time.perf_counter()
        ids = list(input_ids) if input_ids is not None else self._encode(text)
        if not ids:
            raise ValueError("empty prompt")
        input_state = {
            "input_ids": self.torch.tensor([ids], device="cuda", dtype=self.torch.long),
            "past_key_values": None,
            "position_ids": self.torch.arange(len(ids), device="cuda").unsqueeze(0),
        }
        stop = [
            token_id
            for token_id in (self.tokenizer.eos_token_id, self.tokenizer.bos_token_id)
            if token_id is not None
        ]
        generationStart = time.perf_counter()
        # vanilla() prints the decoded sequence.  Keep stdout machine-readable
        # and retain the diagnostic in the worker's stderr instead.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            output, ttft, tpot, generated_ids = vanilla(
                self._sampling_args(),
                self.model,
                self.tokenizer,
                input_state,
                stop,
                self.args.max_new_tokens,
                self._full_config(),
                return_token_ids=True,
            )
        onlineOffset = generationStart - onlineStart
        ttft = onlineOffset + float(ttft)
        total = onlineOffset + (time.perf_counter() - generationStart)
        diagnostic = captured.getvalue().strip()
        if diagnostic:
            print(f"[official-vanilla-helper output] {diagnostic}", file=sys.stderr, flush=True)
        # vanilla() keeps terminal stop IDs out of its decoded text, but they
        # were still generated and must be included in the performance count.
        n_tokens = len(generated_ids)
        return {
            "ok": True,
            "text": output,
            "ttft": float(ttft),
            "tpot": float(tpot),
            "total_time": float(total),
            "num_tokens": int(n_tokens),
            "n_input": len(ids),
            "reuse_ratio": 0.0,
            "runtime_mode": "ragkv_bundled",
            "algorithm": "vanilla",
        }

    @staticmethod
    def _full_config() -> dict:
        """Empty-reuse config that still enters ragkv's patched forward.

        ragkv's patched model expects ``extra_config`` to be a dictionary.  A
        literal ``{}`` makes ``decode_step`` omit that argument, while
        ``reuse_config=None`` keeps the official runtime/kernel but performs a
        complete prefill on every request.
        """
        return {
            "reuse_config": None,
            "drop_config": None,
            "other_config": {"decode": False},
        }

    def serve(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                op = request.get("op")
                if op == "run":
                    response = self.run(str(request.get("text", "")), request.get("input_ids"))
                elif op == "close":
                    _write({"ok": True})
                    return
                else:
                    response = {"ok": False, "error": f"unknown op {op!r}"}
            except Exception as exc:  # noqa: BLE001 - keep protocol alive
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            _write(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--sampling_config", default="{}")
    parser.add_argument("--runtime_reuse", "--reuse_method", dest="runtime_reuse", default="debug")
    # Accepted for symmetry with Worker and the comparison script.  The
    # vanilla path intentionally ignores recomputation ratio because no KV is
    # reused or selectively recomputed.
    parser.add_argument("--recomp_ratio", type=float, default=0.15)
    OfficialVanillaWorker(parser.parse_args()).serve()


if __name__ == "__main__":
    main()
