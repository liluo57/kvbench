#!/usr/bin/env python3
"""Shared KVBench runner for the ProphetKV partial Table-1 reproduction.

This runner deliberately changes only the in-memory PREPARE chunks through
``FixedTokenChunkTask``.  KVBench tasks, workflows, core, and existing methods
remain untouched.  The RUN prompt and task evaluator are inherited unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/data1/ly/.cache/huggingface/hub/models--meta-llama--"
    "Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_DATASET_ROOT = str(ROOT / "data")
DEFAULT_CACHEBLEND_ROOT = "/data1/ly/Projects/CacheBlend"

TASK_CHOICES = ("cwe", "vt", "wikimqa", "triviaqa", "hotpotqa", "musique")


def split_by_tokens(text: str, tokenizer: Any, chunk_size: int) -> List[str]:
    """Split at tokenizer offsets while preserving every input character."""
    if chunk_size <= 0:
        return [text]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = list(encoded.get("offset_mapping") or [])
    if len(offsets) <= chunk_size:
        return [text]
    pieces: List[str] = []
    cursor = 0
    for start in range(0, len(offsets), chunk_size):
        end = int(offsets[min(start + chunk_size, len(offsets)) - 1][1])
        if end <= cursor:
            continue
        pieces.append(text[cursor:end])
        cursor = end
    if cursor < len(text):
        if pieces:
            pieces[-1] += text[cursor:]
        else:
            pieces.append(text)
    return pieces or [text]


class FixedTokenChunkTask:
    """Temporary 512-token wrapper; the underlying KVBench Task is unchanged."""

    def __init__(self, inner: Any, model_path: str, chunk_size: int):
        self.inner = inner
        self.model_path = model_path
        self.chunk_size = int(chunk_size)

    @property
    def Label(self) -> str:
        return self.inner.Label

    def Cases(self) -> Iterator[Any]:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        for case in self.inner.Cases():
            original = list(case.input.prepare_input)
            chunks: List[str] = []
            for text in original:
                chunks.extend(split_by_tokens(text, tokenizer, self.chunk_size))
            if "".join(chunks) != "".join(original):
                raise AssertionError("temporary token split changed PREPARE text")
            case.input.prepare_input = chunks
            # RAGWorkflow keeps the same RAGInput privately.
            if hasattr(case.workflow, "_data"):
                case.workflow._data.prepare_input = chunks
            yield case

    def Evaluate(self, result: Any, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self.inner.Evaluate(result, metadata)


def parser_for(mode: str) -> argparse.ArgumentParser:
    smoke = mode == "smoke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cacheblend-root", default=DEFAULT_CACHEBLEND_ROOT)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / ("prophetkv_smoke" if smoke else "prophetkv_full")),
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--ruler-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--recomp-ratio", type=float, default=0.20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2 if smoke else -1,
        help="per task; -1 means every locally available sample",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASK_CHOICES)
    )
    default_methods = ["full", "prophet1", "prophet20"] if smoke else [
        "full", "naive", "cacheblend", "prophet20"
    ]
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "naive", "cacheblend", "prophet1", "prophet20"),
        default=default_methods,
    )
    parser.add_argument("--task-timeout", type=float, default=18000.0)
    parser.add_argument("--initialize-timeout", type=float, default=1800.0)
    parser.add_argument("--pair-retries", type=int, default=0)
    return parser


def configure(args: argparse.Namespace) -> None:
    from core import Config

    config = copy.deepcopy(Config.LoadConfig())
    config["ModelPath"] = str(Path(args.model).expanduser().resolve())
    config["DatasetPath"] = str(Path(args.dataset_root).expanduser().resolve())
    config["ModelConfig"] = {"mode": "greedy"}
    config["Cacheblend"] = {"Repo": {"RepoPath": str(Path(args.cacheblend_root).expanduser().resolve())}}
    engine = dict(config.get("Engine") or {})
    engine.update({
        "AvailableGpuIds": [int(args.gpu_id)],
        "BatchSize": 1,
        "PairRetries": int(args.pair_retries),
        "OutputRoot": str(Path(args.output_root).expanduser().resolve()),
        "TaskTimeoutSec": float(args.task_timeout),
        "InitializeTimeoutSec": float(args.initialize_timeout),
        "Tui": False,
        "Verbose": True,
    })
    config["Engine"] = engine
    Config._ConfigCache[Config.DefaultConfigPath] = config


def build_tasks(args: argparse.Namespace) -> List[Any]:
    from tasks import CWEShuffleTask, MusiqueTask, HotpotQATask, TriviaQATask
    from tasks import VTShuffleTask, WikimQATask

    root = Path(args.dataset_root).expanduser().resolve()
    common = {"maxSamples": args.max_samples, "startIdx": args.start_index}
    specs = {
        "cwe": lambda: CWEShuffleTask(
            dataset="ruler", dataDir=str(root / "ruler"),
            maxSeqLength=args.ruler_length, **common,
        ),
        "vt": lambda: VTShuffleTask(
            dataset="ruler", dataDir=str(root / "ruler"),
            maxSeqLength=args.ruler_length, **common,
        ),
        "wikimqa": lambda: WikimQATask(
            dataset="wikimqa", dataDir=str(root / "wikimqa"), **common
        ),
        "triviaqa": lambda: TriviaQATask(
            dataset="triviaqa", dataDir=str(root / "triviaqa"), **common
        ),
        "hotpotqa": lambda: HotpotQATask(
            dataset="hotpotqa", dataDir=str(root / "hotpotqa"), **common
        ),
        "musique": lambda: MusiqueTask(
            dataset="musique", dataDir=str(root / "musique"), **common
        ),
    }
    tasks: List[Any] = []
    for name in args.tasks:
        task = specs[name]()
        if args.chunk_size > 0:
            task = FixedTokenChunkTask(task, args.model, args.chunk_size)
        tasks.append(task)
    return tasks


def build_methods(args: argparse.Namespace) -> List[Any]:
    from methods import CacheblendRepo, FullPrefillTransformer, NaiveTransformer, ProphetKV

    common = {
        "gpuNums": 1,
        "maxNewTokens": args.max_new_tokens,
        "dtype": args.dtype,
    }
    methods: List[Any] = []
    for name in args.methods:
        if name == "full":
            methods.append(FullPrefillTransformer(**common, tag="full"))
        elif name == "naive":
            methods.append(NaiveTransformer(**common, tag="naive"))
        elif name == "cacheblend":
            methods.append(CacheblendRepo(
                gpuNums=1, maxNewTokens=args.max_new_tokens,
                maxModelLen=args.max_model_len, recompRatio=args.recomp_ratio,
                tag="cacheblend",
            ))
        elif name == "prophet1":
            methods.append(ProphetKV(
                gpuNums=1, maxNewTokens=args.max_new_tokens,
                maxModelLen=args.max_model_len, recomputeRatio=1.0,
                dtype=args.dtype, tag="r1.00",
            ))
        elif name == "prophet20":
            methods.append(ProphetKV(
                gpuNums=1, maxNewTokens=args.max_new_tokens,
                maxModelLen=args.max_model_len, recomputeRatio=args.recomp_ratio,
                dtype=args.dtype, tag=f"r{args.recomp_ratio:.2f}",
            ))
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def preflight(args: argparse.Namespace) -> None:
    model = Path(args.model).expanduser()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model / 'config.json'}")
    root = Path(args.dataset_root).expanduser()
    for task in args.tasks:
        directory = root / ("ruler" if task in {"cwe", "vt"} else task)
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {directory}")
    if "cacheblend" in args.methods:
        cb = Path(args.cacheblend_root).expanduser()
        if not (cb / ".venv" / "bin" / "python").is_file():
            raise FileNotFoundError(f"CacheBlend venv not found: {cb / '.venv/bin/python'}")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or positive")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be non-negative")
    if not 0.0 <= args.recomp_ratio <= 1.0:
        raise ValueError("--recomp-ratio must be in [0, 1]")


def run(args: argparse.Namespace, mode: str) -> None:
    preflight(args)
    configure(args)
    from core.engine import Engine
    from metrics import ThroughputMetric, TTFTMetric

    tasks = build_tasks(args)
    methods = build_methods(args)
    print(json.dumps({
        "protocol": "kvbench_prophetkv_partial_table1_v1",
        "mode": mode,
        "model": str(Path(args.model).expanduser().resolve()),
        "tasks": args.tasks,
        "methods": [method.Label for method in methods],
        "chunk_size_tokens": args.chunk_size,
        "ruler_length": args.ruler_length,
        "recompute_ratio": args.recomp_ratio,
        "max_samples": args.max_samples,
        "dtype": args.dtype,
    }, indent=2, ensure_ascii=False), flush=True)
    report = Engine().Evaluate(
        tasks=tasks,
        methods=methods,
        metrics=[TTFTMetric(), ThroughputMetric()],
    )
    print("=== KVBench report ===", flush=True)
    print(json.dumps(report.get("cores", report), indent=2, ensure_ascii=False), flush=True)


def main(mode: str) -> None:
    parser = parser_for(mode)
    args = parser.parse_args()
    run(args, mode)

