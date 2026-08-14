"""Shared machinery for the CacheBlend knowledge-base tasks (musique / wikimqa / samsum).

These are the knowledge-base workloads the original CacheBlend repo evaluates
on (``example/blend_musique.py``, ``blend_wikimqa.py``, ``blend_samsum.py``).
Each resolves its data by *name* against ``DatasetPath`` (see ``core.Config``):
``MusiqueTask()`` reads ``<DatasetPath>/musique`` etc.

The prompt text mirrors what the original scripts build (same instruction
prefixes, chunk format and query text) — minus the model-specific chat special
tokens (``[INST]``/``[/INST]``), since the KVBench backends encode the Qwen chat
format built by ``TemplateHelper``.

Data shape (the original ``inputs/*.json``):
    musique / wikimqa:  ``{"ctxs": [{"title", "text"}], "question", "answers"}``
    samsum:             ``{"ctxs": [{"title", "text"}], "question", "answers", ...}``
    (wikimqa's ``answers`` is nested: ``[["answer"]]``)

Case payload contract (consumed by every Method)
-------------------------------------------------
``prepare_input = chunks``      (List[str]; the first chunk carries the
                                 instruction prefix and the user-turn header)
``run_input     = fullPrompt``  (``"".join(wrappedChunks) + suffix``)
``metadata``    = ``{"answers", "question", "n_chunks", ...}``

The ``suffix`` (the fresh question fused against the cached knowledge base) is
recovered by the reuse methods via ``SplitReuseParts``.

:class:`_KBBase` owns the data loading, the chat-prompt building and the scoring
shared by the three task families; each family lives in its own module
(:mod:`tasks.Musique`, :mod:`tasks.Wikimqa`, :mod:`tasks.Samsum`).
"""

import collections
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.Config import DatasetDir
from core.Result import Result
from core.Task import Case, Task

from .TemplateHelper import AssistantSuffix, UserContext

# ---------------------------------------------------------------------------
# Metric helpers (mirror ``example/utils.py`` without the extra deps)
# ---------------------------------------------------------------------------


def NormalizeAnswer(s) -> str:
    """SQuAD-style normalisation (the original ``utils.normalize_answer``)."""

    def RemoveArticles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def WhiteSpaceFix(text: str) -> str:
        return " ".join(text.split())

    def RemovePunc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    return WhiteSpaceFix(RemoveArticles(RemovePunc(str(s).lower())))


def _tokens(s: str) -> List[str]:
    return NormalizeAnswer(s).split()


def ParseGeneration(s) -> str:
    """First-line + Yes/No collapse (the original ``utils.parse_generation``)."""
    s = (s or "").lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        return "Yes"
    words = s.split()
    if words and (words[0].startswith("No") or words[0].startswith("no")):
        return "No"
    return s


def TokenF1(pred: str, gold: str) -> float:
    """Token-overlap F1 (``utils.compute_f1``, over word tokens)."""
    goldToks, predToks = _tokens(gold), _tokens(pred)
    if not goldToks or not predToks:
        return float(goldToks == predToks)
    common = collections.Counter(goldToks) & collections.Counter(predToks)
    numSame = sum(common.values())
    if numSame == 0:
        return 0.0
    precision = numSame / len(predToks)
    recall = numSame / len(goldToks)
    return 2.0 * precision * recall / (precision + recall)


def TokenEm(pred: str, gold: str) -> float:
    """Exact match over normalised text."""
    return float(NormalizeAnswer(pred) == NormalizeAnswer(gold))


def RougeL(pred: str, gold: str) -> float:
    """ROUGE-L F-measure over word tokens (``utils.compute_rl``, LCS-based)."""
    goldToks, predToks = _tokens(gold), _tokens(pred)
    if not goldToks or not predToks:
        return float(goldToks == predToks)
    n, m = len(goldToks), len(predToks)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if goldToks[i] == predToks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2.0 * precision * recall / (precision + recall)


def _FlattenAnswers(answers) -> List[str]:
    """Accept both ``["a"]`` and ``[["a"]]`` (wikimqa nests its answers)."""
    flat: List[str] = []
    for a in answers or []:
        if isinstance(a, (list, tuple)):
            flat.extend(str(x) for x in a if x)
        else:
            flat.append(str(a))
    return flat


# ---------------------------------------------------------------------------
# Knowledge-base base task
# ---------------------------------------------------------------------------


class _KBBase(Task):
    """Shared knowledge-base task logic (case generation + eval).

    Subclasses implement :meth:`_DefaultDataset` (data dir name) and
    :meth:`_Build` (chunks / suffix for one sample).
    """

    #: dataset name resolved against ``DatasetPath``; overridden per task.
    defaultDataset = ""

    def __init__(
        self,
        dataset: Optional[str] = None,
        maxSamples: int = -1,
        startIdx: int = 0,
        dataDir: Optional[str] = None,
    ):
        self.dataset = dataset or self.defaultDataset
        self.maxSamples = maxSamples
        self.startIdx = startIdx
        self.dataDir = Path(dataDir) if dataDir else DatasetDir(self.dataset)

    # ---------------------------------------------------------------- data
    def _LoadSamples(self) -> List[Dict[str, Any]]:
        files = sorted(self.dataDir.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"no *.json files under {self.dataDir} "
                f"(dataset={self.dataset!r})"
            )
        data = json.load(open(files[0], encoding="utf-8"))
        samples = data[self.startIdx:]
        if self.maxSamples != -1:
            samples = samples[: self.maxSamples]
        return samples

    # ---------------------------------------------------------------- cases
    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            chunks, suffix = self._Build(s)
            if not chunks or not suffix:
                continue
            # The whole prompt is one user turn + assistant header (chat format):
            # the user-turn header goes at the start of the cached context and
            # the assistant header is fused with the fresh query.
            chunks = [UserContext(chunks[0])] + chunks[1:]
            suffix = AssistantSuffix(suffix)
            fullPrompt = "".join(chunks) + suffix
            yield Case(
                prepare_input=chunks,          # isolated knowledge chunks
                run_input=fullPrompt,          # whole prompt to generate
                metadata={
                    "case_id": i,
                    "question": s.get("question"),
                    "answers": _FlattenAnswers(s.get("answers")),
                    "n_chunks": len(chunks),
                    "dataset": self.dataset,
                },
            )

    # ------------------------------------------------------------- evaluate
    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError

    def _Score(self, result: Result, metadata: Dict[str, Any]) -> Tuple[float, float]:
        """Return (bestF1, bestEm) over all reference answers."""
        pred = ParseGeneration(result.output)
        answers = metadata["answers"]
        if not answers:
            return 0.0, 0.0
        return max(TokenF1(pred, a) for a in answers), max(
            TokenEm(pred, a) for a in answers
        )

    # --------------------------------------------------------------- prompts
    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# QA over a knowledge base (musique / wikimqa)
# ---------------------------------------------------------------------------


class _QABase(_KBBase):
    """Question-answering over isolated passages (musique / wikimqa).

    Prompt structure copied from ``example/blend_musique.py`` /
    ``blend_wikimqa.py``: instruction prefix, then ``title\n\ntext\n\n``
    passages as separate chunks, then the question prompt as the suffix.
    """

    #: prefix fed to the model before the passages (per-dataset).
    prefixPrompt = ""
    #: query prompt that precedes the question (per-dataset).
    queryPrompt = ""

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        q = self._normalizeQuestion(sample["question"])
        chunks = [f"{c['title']}\n\n{c['text']}\n\n" for c in sample["ctxs"]]
        chunks = [self.prefixPrompt + chunks[0]] + chunks[1:] if chunks else []
        suffix = f"{self.queryPrompt}{q}\nAnswer:"
        return chunks, suffix

    @staticmethod
    def _normalizeQuestion(question: str) -> str:
        q = question if question.endswith("?") else question + "?"
        return q[0].lower() + q[1:]

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        f1, em = self._Score(result, metadata)
        return {"f1": f1, "accuracy": em}
