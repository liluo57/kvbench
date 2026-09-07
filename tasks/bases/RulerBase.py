"""Shared machinery for the RULER long-context tasks (NIAH / VT / CWE).

The datasets are the genuine RULER benchmark (Hsieh et al., COLM 2024),
resolved by *name* against ``DatasetPath`` (see ``core.Config``): each task
reads ``<DatasetPath>/ruler/<task>_len*.jsonl``, e.g.
``niah_len8192.jsonl`` / ``vt_len8192.jsonl`` / ``cwe_len8192.jsonl``.
Each line is a RULER record::

    {index, input, outputs, length, length_w_model_temp, answer_prefix,
     token_position_answer, needle?, depth_percent?}

``input`` is the (possibly ``[INST]``-wrapped) prompt text; ``answer_prefix``
is the start of the expected answer the model should continue from. The tasks
rebuild it as a chat prompt: the whole ``input`` is the user turn and the
``answer_prefix`` opens the assistant turn (see ``helpers.ModelAdapter`` for the
arch-aware boundary strings).

:class:`RulerBase` owns the data loading, the chat-prompt building and the
string-match scoring shared by the three task families; each family lives in
its own module (:mod:`tasks.Niah`, :mod:`tasks.Vt`, :mod:`tasks.Cwe`) together
with its shuffle variant.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.Config import DatasetDir, ModelPath
from core.Result import Result
from core.Task import Case, Task

from helpers.backends.ModelAdapter import assistant_turn_suffix, user_turn_prefix


def _References(refs) -> List[Any]:
    """Normalize and validate the non-empty RULER reference list."""
    if isinstance(refs, str):
        refs = [refs]
    elif refs is None:
        refs = []
    else:
        refs = list(refs)
    if not refs:
        raise ValueError("RULER metrics require at least one reference answer")
    if any(ref is None or not str(ref).strip() for ref in refs):
        raise ValueError("RULER reference answers must not be null or blank")
    return refs


def StringMatchAll(pred: str, refs) -> float:
    """RULER's string-match metric: fraction of ``refs`` found in ``pred``."""
    refs = _References(refs)
    if not pred:
        return 0.0
    predLower = str(pred).lower()
    return sum(1.0 if str(a).lower() in predLower else 0.0 for a in refs) / len(refs)


def ExactMatch(pred: str, refs) -> float:
    """1.0 if the stripped lower-cased prediction equals any of ``refs``."""
    refs = _References(refs)
    if not pred:
        return 0.0
    predNorm = str(pred).strip().lower()
    return float(any(str(a).strip().lower() == predNorm for a in refs))


def _LengthFromName(path: Path) -> Optional[int]:
    """Extract the token length from a filename like ``niah_len8192.jsonl``."""
    m = re.search(r"len(\d+)", path.name)
    return int(m.group(1)) if m else None


def _findNeedleSentence(body: str, value: str) -> str:
    """The sentence of ``body`` containing the needle ``value``.

    Used when a record has no ``needle`` field (official RULER downloads): the
    needle value is unique in the haystack, so the sentence around it is the
    needle sentence.
    """
    pos = body.find(str(value))
    if pos < 0:
        raise RuntimeError(f"needle value {value!r} not found in input")
    start = body.rfind(". ", 0, pos)
    start = start + 2 if start >= 0 else 0
    end = body.find(". ", pos)
    end = end + 1 if end >= 0 else len(body)  # keep the trailing '.'
    return body[start:end].strip()


class RulerBase(Task):
    """Shared RULER data loading + chat-prompt building + scoring.

    Subclasses pick how a sample's prompt is split for the shuffle variant:
    :class:`~tasks.Niah.NIAHTask` splits around the needle (A/B/C);
    :class:`~tasks.Vt.VTTask` and :class:`~tasks.Cwe.CWETask` keep the whole
    prompt as one piece and their ``*ShuffleTask`` split it around the test
    example's informative units (variable assignments / numbered list items).
    """

    #: Dataset directory name resolved against DatasetPath.
    defaultDataset = "ruler"
    #: JSONL filename stem, e.g. ``"niah"`` -> ``niah_len*.jsonl``.
    taskName = "niah"

    def __init__(
        self,
        dataset: Optional[str] = None,
        maxSeqLength: Optional[int] = None,
        maxSamples: int = -1,
        startIdx: int = 0,
        dataDir: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        super().__init__(tag=tag)
        self.dataset = dataset or self.defaultDataset
        self.maxSeqLength = maxSeqLength
        self.maxSamples = maxSamples
        self.startIdx = startIdx
        self.dataDir = Path(dataDir) if dataDir else DatasetDir(self.dataset)

    # ---------------------------------------------------------------- data
    def _LoadSamples(self) -> List[Dict[str, Any]]:
        files = sorted(self.dataDir.glob(f"{self.taskName}_len*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"no {self.taskName}_len*.jsonl under {self.dataDir} "
                f"(dataset={self.dataset!r})"
            )

        samples: List[Dict[str, Any]] = []
        for path in files:
            lenHint = _LengthFromName(path)
            for lineNumber, line in enumerate(path.open(encoding="utf-8"), 1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                outputs = sample.get("outputs")
                if (
                    not isinstance(outputs, list)
                    or not outputs
                    or any(
                        output is None or not str(output).strip()
                        for output in outputs
                    )
                ):
                    raise ValueError(
                        f"RULER sample requires a non-empty outputs list: "
                        f"{path}:{lineNumber}"
                    )
                sample["file"] = path.name
                seqLen = sample.get("max_seq_length") or lenHint
                if self.maxSeqLength is not None and seqLen != self.maxSeqLength:
                    continue
                samples.append(sample)

        samples = samples[self.startIdx:]
        if self.maxSamples != -1:
            samples = samples[: self.maxSamples]
        return samples

    # ---------------------------------------------------------------- chat
    def _StripTemplate(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """``(inputText, answerPrefix)`` with any ``[INST]`` wrapper removed."""
        body = sample["input"]
        for pre in ("[INST] ", "[INST]"):
            if body.startswith(pre):
                body = body[len(pre):]
                break
        for post in (" [/INST]", "[/INST]"):
            if body.endswith(post):
                body = body[: -len(post)]
                break
        return body, sample.get("answer_prefix", "")

    def _FullChat(self, body: str, answerPrefix: str) -> str:
        """The complete chat prompt for ``input = body`` (arch-aware via ModelAdapter)."""
        modelPath = ModelPath()
        return user_turn_prefix(modelPath) + body + assistant_turn_suffix(modelPath) + answerPrefix

    def _Needle(self, sample: Dict[str, Any], body: str) -> str:
        """The needle sentence of ``body``.

        Prefers the record's ``needle`` field; official RULER downloads have
        none, so fall back to the sentence around the (unique) needle value.
        """
        needle = sample.get("needle") or ""
        if needle and needle not in body:
            needle = ""
        if not needle:
            needle = _findNeedleSentence(body, sample["outputs"][0])
        return needle

    def _BuildChatParts(
        self, sample: Dict[str, Any], *, splitNeedle: bool
    ) -> Tuple[Optional[List[str]], str]:
        """``([A, B, C], fullChatPrompt)`` with the needle split.

        A/B/C reassemble exactly to ``fullChatPrompt``:
            A = "<|im_start|>user\n" + text before the needle (instruction +
                essay-before-needle)
            B = the needle sentence
            C = text after the needle + question + assistant header + prefix

        When ``splitNeedle`` is False (VT / CWE) the parts are ``None`` and the
        whole prompt is one piece.
        """
        body, answerPrefix = self._StripTemplate(sample)
        if not splitNeedle:
            return None, self._FullChat(body, answerPrefix)

        needle = self._Needle(sample, body)
        pos = body.find(needle)
        modelPath = ModelPath()
        a = user_turn_prefix(modelPath) + body[:pos]
        b = needle
        c = body[pos + len(needle):] + assistant_turn_suffix(modelPath) + answerPrefix
        return [a, b, c], a + b + c

    # -------------------------------------------------------------- shuffle
    def _Shuffled(self, parts: List[str], seed: int) -> List[str]:
        """Permute ``parts`` so the *joined text* differs from the original.

        The loop guards against two degenerate cases the user-facing guarantee
        ("must not equal the original order") must cover:
        - a permutation that is not the identity order but still concatenates
          to the same text (e.g. a pool containing identical segments);
        - a pool that cannot be reordered at all (< 2 distinct segments).

        Returns the segments in the new order (the caller joins them).
        """
        if len(parts) < 2 or len(set(parts)) < 2:
            return list(parts)
        original = "".join(parts)
        rng = random.Random(seed)
        order = list(range(len(parts)))
        while True:
            rng.shuffle(order)
            if "".join(parts[i] for i in order) != original:
                return [parts[i] for i in order]

    # ------------------------------------------------------------- metadata
    def _Metadata(self, i: int, s: Dict[str, Any], fullPrompt: str) -> Dict[str, Any]:
        return {
            "case_id": i,
            "outputs": s.get("outputs", []),
            "needle": s.get("needle"),
            "depth_percent": s.get("depth_percent"),
            "length": s.get("length"),
            "length_w_model_temp": s.get("length_w_model_temp"),
            "file": s.get("file"),
        }

    # ------------------------------------------------------------- evaluate
    def _Score(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        pred = result.output or ""
        refs = metadata["outputs"]
        return {
            "accuracy": StringMatchAll(pred, refs),
            "exact_match": ExactMatch(pred, refs),
        }
