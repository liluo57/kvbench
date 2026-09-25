#!/usr/bin/env python3
"""Run the KVBench-native Mistral A3 adapter comparison.

This is deliberately a KVBench experiment driver, not a reimplementation of
the paper's full LongBench harness.  It uses only the five local Task classes
and datasets currently shipped in this checkout:

    HotpotQA, GovReport, MultiNews, TriviaQA, and SAMSum.

The default method set is the one needed to validate the adapter in context:
Vanilla (full prefill), FullReuse (naive), the existing CacheBlend repository
adapter, and the official A3 repository adapter.  No CacheBlend, core, task,
or workload source is modified by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Engine workers use multiprocessing ``spawn``.  Propagate the command-line
# model override into those fresh interpreters before they render task prompts;
# otherwise ModelAdapter would fall back to the repository's unrelated global
# config.yaml model path.
_spawn_model_path = os.environ.get("KVBENCH_MODEL_PATH")
if _spawn_model_path:
    from core import Config as _SpawnConfig
    _SpawnConfig.LoadConfig()["ModelPath"] = _spawn_model_path

TASK_SPECS = {
    "hotpotqa": "HotpotQATask",
    "govreport": "GovReportTask",
    "multinews": "MultiNewsTask",
    "triviaqa": "TriviaQATask",
    "samsum": "SamsumTask",
}

# These are the metrics used for the corresponding LongBench task families.
# The Engine still records every score returned by each Task (for example both
# F1 and EM on QA); this mapping tells the post-processing which one to use as
# the per-task comparison column.
PRIMARY_METRICS = {
    "hotpotqa": "f1",
    "govreport": "rougeL",
    "multinews": "rougeL",
    "triviaqa": "f1",
    "samsum": "rougeL",
}


def parse_args() -> argparse.Namespace:
    from core import Config

    config = Config.LoadConfig()
    cacheblend = config.get("Cacheblend") or {}
    cacheblend_repo = cacheblend.get("Repo") or {}
    default_model = os.environ.get("KVBENCH_MODEL_PATH") or config.get("ModelPath")
    default_a3_repo = os.environ.get("KVBENCH_A3_REPO_PATH")
    default_a3_python = os.environ.get("KVBENCH_A3_PYTHON")
    default_cacheblend_repo = (
        os.environ.get("KVBENCH_CACHEBLEND_REPO_PATH")
        or cacheblend_repo.get("RepoPath")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model, required=not bool(default_model),
                        help="Mistral-7B-Instruct-v0.2 checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data",
                        help="KVBench data directory")
    parser.add_argument("--a3-repo-root", default=default_a3_repo,
                        help="Official ragkv checkout used by A3Repo")
    parser.add_argument("--a3-python", default=default_a3_python,
                        help="Python executable in the official ragkv env")
    parser.add_argument("--cacheblend-root", default=default_cacheblend_repo,
                        help="Original CacheBlend checkout")
    parser.add_argument("--gpu", type=int, required=True,
                        help="One physical GPU id; methods run sequentially")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / "a3_kvbench_mistral",
                        help="Engine output directory")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Samples per Task (-1 means all local samples)")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--exclude-indices", type=int, nargs="*", default=[],
                        help="Original zero-based dataset indices to skip")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Default generation budget for all methods")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Temporary token chunk size (0 keeps KVBench chunks)")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        default="float16",
                        help="dtype for the plain Vanilla/FullReuse baselines")
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--task-timeout", type=float, default=18000.0)
    parser.add_argument("--initialize-timeout", type=float, default=1800.0)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SPECS),
                        default=list(TASK_SPECS),
                        help="Subset of the five shipped KVBench tasks")
    parser.add_argument("--methods", nargs="+",
                        choices=("vanilla", "fullreuse", "cacheblend", "a3"),
                        default=["vanilla", "fullreuse", "cacheblend", "a3"],
                        help="Comparison methods; default is all four")
    parser.add_argument("--pair-retries", type=int, default=0)
    args = parser.parse_args()
    if "a3" in args.methods:
        if not args.a3_repo_root:
            parser.error("--a3-repo-root or KVBENCH_A3_REPO_PATH is required for A3")
        if not args.a3_python:
            parser.error("--a3-python or KVBENCH_A3_PYTHON is required for A3")
    if "cacheblend" in args.methods and not args.cacheblend_root:
        parser.error(
            "--cacheblend-root or KVBENCH_CACHEBLEND_REPO_PATH is required "
            "for CacheBlend"
        )
    return args


def configure_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    """Override the in-memory KVBench config without editing config.yaml."""
    from core import Config

    config = copy.deepcopy(Config.LoadConfig())
    config["ModelPath"] = str(Path(args.model).expanduser().resolve())
    config["DatasetPath"] = str(args.dataset_root.expanduser().resolve())
    os.environ["KVBENCH_MODEL_PATH"] = config["ModelPath"]
    config["ModelConfig"] = {"mode": "greedy"}
    if "a3" in args.methods:
        config["A3"] = {
            "Repo": {
                "RepoPath": str(Path(args.a3_repo_root).expanduser().resolve()),
                "Python": str(Path(args.a3_python).expanduser().resolve()),
            }
        }
    if args.cacheblend_root:
        config["Cacheblend"] = {
            "Repo": {"RepoPath": str(Path(args.cacheblend_root).expanduser().resolve())}
        }
    engine = dict(config.get("Engine") or {})
    engine.update({
        "AvailableGpuIds": [int(args.gpu)],
        "BatchSize": 1,
        "PairRetries": int(args.pair_retries),
        "OutputRoot": str(args.output_root.expanduser().resolve()),
        "TaskTimeoutSec": float(args.task_timeout),
        "InitializeTimeoutSec": float(args.initialize_timeout),
        "Tui": False,
        "Verbose": True,
    })
    config["Engine"] = engine
    Config._ConfigCache[Config.DefaultConfigPath] = config
    return config


def preflight(args: argparse.Namespace) -> None:
    model = Path(args.model).expanduser()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Mistral model config not found: {model / 'config.json'}")
    dataset_root = args.dataset_root.expanduser()
    missing = [name for name in args.tasks if not (dataset_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"KVBench dataset directories missing under {dataset_root}: {missing}"
        )
    required_paths = []
    if "a3" in args.methods:
        required_paths.extend((
            ("A3 repo", args.a3_repo_root),
            ("A3 Python", args.a3_python),
        ))
    if "cacheblend" in args.methods:
        required_paths.append(("CacheBlend repo", args.cacheblend_root))
    for label, path in required_paths:
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer")
    if any(index < 0 for index in args.exclude_indices):
        raise ValueError("--exclude-indices must contain non-negative indices")
    if not 0.0 <= args.recomp_ratio <= 1.0:
        raise ValueError("--recomp-ratio must be in [0, 1]")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")


def _split_by_tokens(text: str, tokenizer: Any, chunk_size: int) -> List[str]:
    """Split text at tokenizer boundaries while preserving every character."""
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
        end_offset = offsets[min(start + chunk_size, len(offsets)) - 1][1]
        end_offset = int(end_offset)
        if end_offset <= cursor:
            continue
        pieces.append(text[cursor:end_offset])
        cursor = end_offset
    if cursor < len(text):
        if pieces:
            pieces[-1] += text[cursor:]
        else:
            pieces.append(text)
    return pieces or [text]


class FixedTokenChunkTask:
    """Pickle-friendly task wrapper for the temporary 512-token protocol.

    The underlying KVBench Task remains untouched.  Only the PREPARE chunks
    are repartitioned; the already-rendered RUN prompt is unchanged because
    the split function preserves the original text byte-for-byte.
    """

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
                chunks.extend(_split_by_tokens(text, tokenizer, self.chunk_size))
            case.input.prepare_input = chunks
            # RAGWorkflow stores the same RAGInput privately.  Keep both
            # references in sync without changing workflow.py.
            if hasattr(case.workflow, "_data"):
                case.workflow._data.prepare_input = chunks
            yield case

    def Evaluate(self, result: Any, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self.inner.Evaluate(result, metadata)


class ExcludeIndicesTask:
    """Filter selected original dataset rows without changing Task/core code."""

    def __init__(self, inner: Any, excluded: List[int], start_index: int):
        self.inner = inner
        self.excluded = set(int(index) for index in excluded)
        self.start_index = int(start_index)

    @property
    def Label(self) -> str:
        return self.inner.Label

    def Cases(self) -> Iterator[Any]:
        for offset, case in enumerate(self.inner.Cases()):
            if self.start_index + offset in self.excluded:
                continue
            yield case

    def Evaluate(self, result: Any, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self.inner.Evaluate(result, metadata)


def build_tasks(args: argparse.Namespace) -> List[Any]:
    from tasks import GovReportTask, HotpotQATask, MultiNewsTask, SamsumTask, TriviaQATask

    constructors = {
        "hotpotqa": HotpotQATask,
        "govreport": GovReportTask,
        "multinews": MultiNewsTask,
        "triviaqa": TriviaQATask,
        "samsum": SamsumTask,
    }
    tasks: List[Any] = []
    root = args.dataset_root.expanduser().resolve()
    for name in args.tasks:
        task = constructors[name](
            dataset=name,
            dataDir=str(root / name),
            maxSamples=args.max_samples,
            startIdx=args.start_index,
        )
        if name == "govreport" and args.exclude_indices:
            task = ExcludeIndicesTask(task, args.exclude_indices, args.start_index)
        if args.chunk_size:
            task = FixedTokenChunkTask(task, str(args.model), args.chunk_size)
        tasks.append(task)
    return tasks


def build_methods(args: argparse.Namespace) -> List[Any]:
    from methods import A3Repo, CacheblendRepo, FullPrefillTransformer, NaiveTransformer

    common = {
        "gpuNums": 1,
        "maxNewTokens": args.max_new_tokens,
        "dtype": args.dtype,
    }
    methods: List[Any] = []
    if "vanilla" in args.methods:
        methods.append(FullPrefillTransformer(**common, tag="vanilla"))
    if "fullreuse" in args.methods:
        methods.append(NaiveTransformer(**common, tag="fullreuse"))
    if "cacheblend" in args.methods:
        methods.append(CacheblendRepo(
            gpuNums=1,
            maxNewTokens=args.max_new_tokens,
            maxModelLen=args.max_model_len,
            recompRatio=args.recomp_ratio,
            tag="cacheblend",
        ))
    if "a3" in args.methods:
        methods.append(A3Repo(
            gpuNums=1,
            maxNewTokens=args.max_new_tokens,
            maxModelLen=args.max_model_len,
            recompRatio=args.recomp_ratio,
            reuseMethod="debug",
            repoPath=args.a3_repo_root,
            pythonPath=args.a3_python,
            tag="a3",
        ))
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def make_manifest(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocol": "kvbench_a3_mistral_local_tasks_v1",
        "goal": "Validate A3Repo against KVBench-native baselines, not full 11-task Table 1 reproduction",
        "model": str(Path(args.model).expanduser().resolve()),
        "dtype": {
            "transformer_baselines": args.dtype,
            "a3_repo": "official-loader-default (currently bfloat16)",
            "cacheblend_repo": "repository-default",
        },
        "chunk_size_tokens": args.chunk_size,
        "primary_metric_by_task": {
            name: PRIMARY_METRICS[name] for name in args.tasks
        },
        "tasks": list(args.tasks),
        "datasets": {
            name: str((args.dataset_root.expanduser().resolve() / name))
            for name in args.tasks
        },
        "max_samples": args.max_samples,
        "start_index": args.start_index,
        "exclude_indices": list(args.exclude_indices),
        "methods": list(args.methods),
        "recomp_ratio": args.recomp_ratio,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu": args.gpu,
        "a3_repo": (
            str(Path(args.a3_repo_root).expanduser().resolve())
            if args.a3_repo_root else None
        ),
        "cacheblend_repo": (
            str(Path(args.cacheblend_root).expanduser().resolve())
            if args.cacheblend_root else None
        ),
        "engine": config.get("Engine", {}),
    }


def main() -> int:
    args = parse_args()
    preflight(args)
    config = configure_runtime(args)
    tasks = build_tasks(args)
    methods = build_methods(args)

    from core.engine import Engine
    from metrics import ThroughputMetric, TTFTMetric

    print("[a3-table1] tasks:", ", ".join(task.Label for task in tasks), flush=True)
    print("[a3-table1] methods:", ", ".join(method.Label for method in methods), flush=True)
    report = Engine().Evaluate(
        tasks=tasks,
        methods=methods,
        metrics=[TTFTMetric(), ThroughputMetric()],
    )

    args.output_root.expanduser().mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root.expanduser() / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(make_manifest(args, config), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report_path = args.output_root.expanduser() / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ok": True,
        "report": str(report_path),
        "manifest": str(manifest_path),
        "engine_output": report.get("output_dir"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
