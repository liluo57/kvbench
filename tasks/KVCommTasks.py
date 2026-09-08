"""Text-only reproductions of KVCOMM's four benchmark entry points."""

import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from core.Config import DatasetDir
from core.Result import Result
from core.Task import Case, Task
from workflow import AgentSpec, MultiAgentFullConnectionInput, MultiAgentFullConnectionWorkflow


_NUMBER = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _extract_choice(text: Any) -> str | None:
    """Extract an MMLU choice without matching letters inside prose."""
    output = str(text or "").strip().upper()
    output = re.sub(r"<THINK>.*?</THINK>", "", output, flags=re.S)
    lines = [
        re.sub(r"[*_`]", "", line).strip()
        for line in output.splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        explicit = re.search(
            r"(?:FINAL\s+)?ANSWER\s*(?:IS|:)?\s*(?:OPTION\s*)?([A-D])\b",
            line,
        )
        if explicit:
            return explicit.group(1)
        match = re.fullmatch(
            r"(?:THE\s+)?(?:FINAL\s+)?ANSWER\s*(?:IS|:)?\s*\(?([A-D])\)?[.!]?",
            line,
        )
        if match:
            return match.group(1)
        match = re.fullmatch(r"\(?([A-D])\)?[.!]?", line)
        if match:
            return match.group(1)
    return None


def _extract_number(text: Any, *, target: bool = False) -> Decimal | None:
    """Extract the final GSM8K answer, preferring explicit answer markers."""
    output = str(text or "")
    patterns = [r"####\s*(" + _NUMBER + r")"] if target else [
        r"(?:THE\s+)?(?:FINAL\s+)?ANSWER\s*(?:IS|:|=)\s*(" + _NUMBER + r")"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.I)
        if matches:
            raw = matches[-1]
            break
    else:
        matches = re.findall(_NUMBER, output)
        if not matches:
            return None
        raw = matches[-1]
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _extract_python(text: Any) -> str | None:
    output = str(text or "").strip()
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", output, flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    # Accept raw code, as the official evaluator does, but reject explanatory
    # prose that would otherwise turn every candidate into a SyntaxError.
    if re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\s*\()", output):
        return output
    return None


def _limit_samples(rows: List[Any], max_samples: Optional[int]) -> List[Any]:
    """Apply the common sample limit without treating zero as unlimited."""
    if max_samples is None:
        return rows
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("maxSamples must be an integer or None")
    if max_samples < 0:
        raise ValueError("maxSamples must be non-negative")
    return rows[:max_samples]


def _cases(rows, specs, decision, metadata=None):
    for i, row in enumerate(rows):
        text, meta = row if isinstance(row, tuple) else (row, {})
        yield Case(
            text,
            MultiAgentFullConnectionWorkflow(
                i, MultiAgentFullConnectionInput(text, specs, decision)
            ),
            meta if metadata is None else metadata(meta),
        )


def _cycle(names, count, template):
    return [
        AgentSpec(
            n,
            template.replace("{role}", n),
            systemPrompt=f"You are the {n}.",
        )
        for n in (names * ((count + len(names) - 1) // len(names)))[:count]
    ]


class KVCommMMLUTask(Task):
    name = "kvcomm_mmlu"
    def __init__(self, maxSamples=None, agentCount=5, tag: Optional[str] = None):
        super().__init__(tag=tag)
        self.maxSamples, self.agentCount = maxSamples, agentCount
        self._rows = None
    def Cases(self) -> Iterator[Case]:
        if self._rows is None:
            rows = []
            root = DatasetDir("mmlu")
            files = sorted((root / "val").glob("*.csv"))
            for path in files:
                with path.open(newline="", encoding="utf-8") as f:
                    for r in csv.reader(f):
                        if len(r) >= 6: rows.append(("{}\nOption A: {}\nOption B: {}\nOption C: {}\nOption D: {}".format(r[0], r[1], r[2], r[3], r[4]), {"answer": r[5]}))
            random.Random(888).shuffle(rows)
            self._rows = _limit_samples(rows, self.maxSamples)
        roles = ["Knowledgeable Expert", "Wiki Searcher", "Critic", "Mathematician", "Psychologist", "Historian", "Doctor", "Lawyer", "Economist", "Programmer"]
        specs = _cycle(roles, self.agentCount, "You are a {role}. Analyze the question and choose one of A, B, C, or D.\n\nQ: {task}")
        decision = AgentSpec("FinalRefer", "You are the top decision-maker. Choose exactly one of A, B, C, or D using the analyses.\n\nQ: {task}")
        yield from _cases(self._rows, specs, decision)
    def Evaluate(self, result, metadata):
        answer = str(metadata.get("answer", "")).strip().upper()
        return {"accuracy": float(bool(answer) and _extract_choice(result.output) == answer)}


class KVCommGSM8KTask(Task):
    name = "kvcomm_gsm8k"
    def __init__(self, maxSamples=None, agentCount=3, tag: Optional[str] = None):
        super().__init__(tag=tag)
        self.maxSamples, self.agentCount = maxSamples, agentCount
        self._rows = None
    def Cases(self):
        if self._rows is None:
            root = DatasetDir("gsm8k")
            path = root / "gsm8k.jsonl"
            if not path.exists():
                path = root / "gsm8k.json"
            rows = []
            with path.open(encoding="utf-8") as f:
                records = json.load(f) if path.suffix == ".json" else (json.loads(line) for line in f)
                for r in records:
                    rows.append((r.get("question", ""), {"answer": r.get("answer", r.get("target", ""))}))
            self._rows = _limit_samples(rows, self.maxSamples)
        specs = _cycle(["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"], self.agentCount, "You are a {role}. Solve the problem step by step. Include few-shot reasoning where useful.\n\nQ:{task}\n")
        decision = AgentSpec("FinalRefer", "You are the top decision-maker. Select the most reliable mathematical answer from the following work.\n\nQ:{task}\n")
        yield from _cases(self._rows, specs, decision)
    def Evaluate(self, result, metadata):
        predicted = _extract_number(result.output)
        target = _extract_number(metadata.get("answer", ""), target=True)
        return {"accuracy": float(predicted is not None and predicted == target)}


class KVCommHumanEvalTask(Task):
    name = "kvcomm_humaneval"
    def __init__(self, maxSamples=None, agentCount=5, tag: Optional[str] = None):
        super().__init__(tag=tag)
        self.maxSamples, self.agentCount = maxSamples, agentCount
        self._rows = None
    def Cases(self):
        if self._rows is None:
            rows = []
            with (DatasetDir("humaneval") / "humaneval-py.jsonl").open(encoding="utf-8") as f:
                for line in f: rows.append(json.loads(line))
            self._rows = _limit_samples(rows, self.maxSamples)
        names = ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Bug Fixer"]
        specs = _cycle(names, self.agentCount, "You are the {role}. Produce Python code for the task.\n\n{task}")
        decision = AgentSpec("Final Decision", "Return only the best Python implementation in a ```python block.\n\n{task}")
        for i, r in enumerate(self._rows):
            text = r.get("prompt", ""); meta = {"test": r.get("test", ""), "entry_point": r.get("entry_point", "")}
            yield Case(text, MultiAgentFullConnectionWorkflow(i, MultiAgentFullConnectionInput(text, specs, decision)), meta)
    def Evaluate(self, result, metadata):
        code = _extract_python(result.output)
        if code is None:
            return {"accuracy": 0.0}

        command = [
            sys.executable,
            "-I",
            "-c",
            code + "\n" + metadata.get("test", ""),
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="kvcomm-humaneval-") as cwd:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env={"PATH": os.environ.get("PATH", "")},
                    start_new_session=(os.name == "posix"),
                    preexec_fn=_limit_human_eval_process if os.name == "posix" else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.communicate()
                    return {"accuracy": 0.0}
                return {"accuracy": float(process.returncode == 0)}
        except (OSError, subprocess.SubprocessError):
            return {"accuracy": 0.0}


class KVCommCopyTask(Task):
    name = "kvcomm_copy"
    def __init__(self, nCases=100, agentCount=5, seed=42, tag: Optional[str] = None):
        super().__init__(tag=tag)
        self.nCases, self.agentCount, self.seed = nCases, agentCount, seed
    def Cases(self):
        rng = random.Random(self.seed)
        specs = [AgentSpec("Copy Machine", " Ω" * 512 + "\nRandomly output Ω or Δ 512 times.\n\n{task}") for _ in range(self.agentCount)]
        for i in range(self.nCases):
            text = " ".join(rng.choices(["Δ", "Ω"], k=1000))
            yield Case(text, MultiAgentFullConnectionWorkflow(i, MultiAgentFullConnectionInput(text, specs)), {})
    def Evaluate(self, result, metadata): return {}


def _limit_human_eval_process() -> None:
    """Apply best-effort POSIX limits before executing generated code."""
    import resource

    limits = {
        resource.RLIMIT_CPU: (10, 10),
        resource.RLIMIT_FSIZE: (16 * 1024 * 1024, 16 * 1024 * 1024),
        resource.RLIMIT_NOFILE: (64, 64),
        resource.RLIMIT_AS: (1024 * 1024 * 1024, 1024 * 1024 * 1024),
    }
    for resource_kind, limit in limits.items():
        try:
            resource.setrlimit(resource_kind, limit)
        except (ValueError, OSError):
            continue
