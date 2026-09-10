"""Shared setup for the A3 Table 1 reproduction scripts.

The Table 1 experiment is intentionally kept outside ``Main.py`` so the
repository's existing AgentBench entry point is unchanged.  This module
provides the fixed Mistral/LongBench subset protocol and a small task wrapper
that splits prepared text into tokenizer-aligned 512-token chunks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = (
    "/data1/ly/.cache/huggingface/hub/"
    "models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/"
    "63a8b081895390a26e140280378bc85ec8bce07a"
)
DEFAULT_DATASET_ROOT = "/root/kvbench/data"
DEFAULT_CACHEBLEND_ROOT = "/root/cache-blend/CacheBlend"
DATASETS = ("hotpotqa", "govreport", "multinews", "triviaqa", "samsum")
GROUPS = {
    "qa": (("hotpotqa", "triviaqa"), 32, "f1"),
    "summary": (("govreport", "multinews"), 512, "rougeL"),
    "samsum": (("samsum",), 128, "rougeL"),
}


def _split_text(text: str, tokenizer: Any, chunk_size: int = 512) -> List[str]:
    """Split text at tokenizer offsets while preserving the exact text."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise ValueError("tokenizer returned inconsistent ids/offsets")
    if not ids:
        return [text] if text else []

    pieces: List[str] = []
    for begin in range(0, len(ids), chunk_size):
        end = min(begin + chunk_size, len(ids))
        left = offsets[begin][0]
        right = offsets[end - 1][1]
        # Keep inter-token whitespace with the preceding piece.  This makes
        # ``''.join(pieces) == text`` even for BPE tokenizers with gaps.
        if begin > 0:
            previous_right = offsets[begin - 1][1]
            left = previous_right
        piece = text[left:right]
        if piece:
            pieces.append(piece)
    if "".join(pieces) != text:
        # Offset mappings can omit trailing whitespace.  Preserve the original
        # text exactly rather than silently changing the benchmark prompt.
        rebuilt: List[str] = []
        cursor = 0
        for begin in range(0, len(ids), chunk_size):
            end = min(begin + chunk_size, len(ids))
            right = offsets[end - 1][1]
            rebuilt.append(text[cursor:right])
            cursor = right
        rebuilt[-1] += text[cursor:]
        pieces = [piece for piece in rebuilt if piece]
    if "".join(pieces) != text:
        raise ValueError("512-token chunking changed the original text")
    return pieces


class ChunkedTask:
    """Delegate scoring to an existing task and replace only its RAG input."""

    def __init__(self, base: Any, tokenizer: Any, chunk_size: int = 512):
        self.base = base
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.name = base.name
        self.tag = base.tag

    @property
    def Label(self) -> str:
        return self.base.Label

    def Cases(self) -> Iterator[Any]:
        from core.Task import Case
        from workflow.RAGWorkflow import RAGInput, RAGWorkflow

        for case in self.base.Cases():
            original = case.input
            chunks: List[str] = []
            for chunk in list(original.prepare_input or []):
                chunks.extend(_split_text(chunk, self.tokenizer, self.chunk_size))
            run_input = original.run_input
            if "".join(chunks) not in run_input:
                raise ValueError(
                    f"{self.Label} chunked prepared text is not present in run prompt"
                )
            data = RAGInput(prepare_input=chunks, run_input=run_input)
            metadata = dict(case.metadata)
            metadata["n_chunks"] = len(chunks)
            metadata["chunk_size_tokens"] = self.chunk_size
            yield Case(
                input=data,
                workflow=RAGWorkflow(case_id=case.workflow.case_id, data=data),
                metadata=metadata,
            )

    def Evaluate(self, result: Any, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self.base.Evaluate(result, metadata)


def configure_runtime(
    *,
    model: str,
    dataset_root: str,
    cacheblend_root: str,
    gpu_id: int,
    output_root: str,
    task_timeout: float,
) -> None:
    """Override the cached repository config without editing config.yaml."""
    from core import Config

    config = copy.deepcopy(Config.LoadConfig())
    config["ModelPath"] = str(model)
    config["DatasetPath"] = str(dataset_root)
    config["ModelConfig"] = {"mode": "greedy"}
    config["Cacheblend"] = {"Repo": {"RepoPath": str(cacheblend_root)}}
    engine = dict(config.get("Engine") or {})
    engine.update(
        {
            "AvailableGpuIds": [int(gpu_id)],
            "BatchSize": 1,
            "PairRetries": 0,
            "OutputRoot": str(output_root),
            "TaskTimeoutSec": float(task_timeout),
            "InitializeTimeoutSec": max(600.0, min(float(task_timeout), 1800.0)),
            "Tui": False,
            "Verbose": True,
        }
    )
    config["Engine"] = engine
    Config._ConfigCache[Config.DefaultConfigPath] = config


def load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def make_tasks(
    dataset_root: str,
    tokenizer: Any,
    *,
    max_samples: int,
    datasets: Sequence[str],
) -> List[Any]:
    from tasks import GovReportTask, HotpotQATask, MultiNewsTask, SamsumTask, TriviaQATask

    constructors = {
        "hotpotqa": HotpotQATask,
        "govreport": GovReportTask,
        "multinews": MultiNewsTask,
        "triviaqa": TriviaQATask,
        "samsum": SamsumTask,
    }
    result = []
    for name in datasets:
        if name not in constructors:
            raise ValueError(f"unsupported Table 1 subset dataset: {name}")
        task = constructors[name](
            dataset=name,
            dataDir=str(Path(dataset_root) / name),
            maxSamples=max_samples,
            startIdx=0,
        )
        result.append(ChunkedTask(task, tokenizer, 512))
    return result


def make_methods(
    *,
    max_new_tokens: int,
) -> List[Any]:
    """Construct the five required Table 1 methods.

    EPIC is deliberately required.  If its adapter is not yet present, fail
    loudly instead of silently producing a four-method comparison.
    """
    from methods import A3, CacheblendRepo, FullPrefillTransformer, NaiveTransformer

    epic_cls = None
    for module_name, class_name in (
        ("methods.EPIC", "EPIC"),
        ("methods.Epic", "Epic"),
        ("methods.LegoLink", "LegoLink"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            epic_cls = getattr(module, class_name)
            break
        except (ImportError, AttributeError):
            continue
    if epic_cls is None:
        raise RuntimeError(
            "Table 1 requires the EPIC/LegoLink Method adapter; expected "
            "methods.EPIC.EPIC, methods.Epic.Epic, or methods.LegoLink.LegoLink"
        )

    common = dict(gpuNums=1, maxNewTokens=max_new_tokens)
    return [
        FullPrefillTransformer(**common, dtype="float16", tag="vanilla"),
        NaiveTransformer(**common, dtype="float16", tag="fullreuse"),
        CacheblendRepo(
            **common,
            recompRatio=0.15,
            tag="cacheblend",
        ),
        epic_cls(
            gpuNums=1,
            maxNewTokens=max_new_tokens,
            recomputeTokensPerChunk=20,
            dtype="float16",
            tag="legolink_epic",
        ),
        A3(
            **common,
            dtype="float16",
            recompRatio=0.15,
            tag="a3",
        ),
    ]


def environment_manifest(args: Any, *, group: str, datasets: Sequence[str], max_samples: int) -> Dict[str, Any]:
    return {
        "protocol": "a3_table1_subset_v1",
        "group": group,
        "datasets": list(datasets),
        "max_samples": int(max_samples),
        "chunk_size_tokens": 512,
        "chunk_overlap_tokens": 0,
        "dtype": {
            "a3": "float16",
            "full_prefill": "float16",
            "naive": "float16",
            "epic": "float16",
            "cacheblend": "auto (unchanged repository default)",
        },
        "sampling": "greedy",
        "model": str(args.model),
        "dataset_root": str(args.dataset_root),
        "gpu_id": int(args.gpu_id),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": list(sys.argv),
    }


def run_group(args: Any, group: str, *, max_samples: int) -> Dict[str, Any]:
    from core.engine import Engine

    datasets, max_new_tokens, metric = GROUPS[group]
    configure_runtime(
        model=args.model,
        dataset_root=args.dataset_root,
        cacheblend_root=args.cacheblend_root,
        gpu_id=args.gpu_id,
        output_root=str(Path(args.output_root) / group),
        task_timeout=args.task_timeout,
    )
    tokenizer = load_tokenizer(args.model)
    tasks = make_tasks(
        args.dataset_root,
        tokenizer,
        max_samples=max_samples,
        datasets=datasets,
    )
    methods = make_methods(
        max_new_tokens=max_new_tokens,
    )
    report = Engine().Evaluate(tasks=tasks, methods=methods, metrics=[])
    payload = {
        "group": group,
        "metric": metric,
        "datasets": list(datasets),
        "max_new_tokens": max_new_tokens,
        "report": report,
        "manifest": environment_manifest(
            args, group=group, datasets=datasets, max_samples=max_samples
        ),
    }
    out = Path(args.output_root) / group / "table1_group.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def add_common_arguments(parser: Any, *, default_samples: int) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cacheblend-root", default=DEFAULT_CACHEBLEND_ROOT)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-root", default="outputs/table1_mistral_subset")
    parser.add_argument("--max-samples", type=int, default=default_samples)
    parser.add_argument("--task-timeout", type=float, default=18000.0)


def preflight(args: Any) -> None:
    model = Path(args.model)
    if not model.is_dir():
        raise FileNotFoundError(f"model directory not found: {model}")
    if not (model / "config.json").exists():
        raise FileNotFoundError(f"model config.json not found: {model}")
    root = Path(args.dataset_root)
    missing = [name for name in DATASETS if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing Table 1 dataset directories: {missing}")
    if not Path(args.cacheblend_root).is_dir():
        raise FileNotFoundError(f"CacheBlend repo not found: {args.cacheblend_root}")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
