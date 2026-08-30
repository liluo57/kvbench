"""RULER Common Words Extraction: the base task and its shuffle variant.

Find the 10 most frequent words in a numbered list; ``outputs`` is those 10
words, all of which must appear in the prediction. Data come from
``<DatasetPath>/ruler/cwe_len*.jsonl`` (see :mod:`tasks.bases.RulerBase` for
the record layout and chat-prompt wrapping).

Dataset structure (what the shuffle keys on)
--------------------------------------------
RULER CWE is few-shot: a record's ``input`` holds *two* examples concatenated —
a filled example, then the test example. Each example is an instruction, a
numbered list of words (``N. word``, the common words repeated 30x, the
uncommon ones 3x), and a question. The last example is the test one: the
record's ``answer_prefix`` completes its question and ``outputs`` is its list's
10 most frequent words.

The informative units are the numbered list items. Their *frequencies* are the
answer, and a frequency is invariant under any permutation of the items — so
shuffling the test list's items changes the prompt text while leaving the
correct answer untouched. That is exactly what makes a shuffle a fair test here
— see :class:`CWEShuffleTask`.

Case payload contract
---------------------
:class:`CWETask`:
    ``input = RAGInput(prepare_input=[], run_input=fullChatPrompt)``
    ``workload = RAGWorkload`` (skip Prepare, just Run)

:class:`CWEShuffleTask`:
    ``input = RAGInput(prepare_input=segments, run_input=shuffled_prompt)``
    ``workload = RAGWorkload`` (Prepare → Run)

``chunks`` are fixed-size blocks of the test list's numbered items. ``head``
(few-shot example + test instruction) and ``tail`` (test question + assistant
header + answer prefix) are kept in place so the shuffled prompt stays a
coherent, answerable prompt; only the item blocks reorder (the out-of-order
item numbers are labels, not part of the answer).
"""

import re
from typing import Any, Dict, Iterator, List

from core.Config import ModelPath
from core.Result import Result
from core.Task import Case
from workload.RAGWorkload import RAGInput, RAGWorkload

from .bases.RulerBase import RulerBase
from helpers.backends.ModelAdapter import assistant_turn_suffix, user_turn_prefix

#: The CWE instruction line, used to locate the test example's list.
_HeaderLine = "Below is a numbered list"
#: A numbered list item start, e.g. ``"1. "`` / ``"945. "``.
_ItemPattern = re.compile(r"\d+\.\s")
#: Number of item blocks the test list is split into (the shuffle pool size).
_NChunks = 10


class CWETask(RulerBase):
    """Common words extraction: list the most frequent words in the haystack.

    ``outputs`` is the 10 most common words; all of them must appear in the
    prediction.
    """

    name = "cwe"
    taskName = "cwe"

    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            body, answerPrefix = self._StripTemplate(s)
            fullPrompt = self._FullChat(body, answerPrefix)
            yield Case(
                input=RAGInput(prepare_input=[], run_input=fullPrompt),
                workload=RAGWorkload(case_id=i, data=RAGInput(prepare_input=[], run_input=fullPrompt)),
                metadata=self._Metadata(i, s, fullPrompt),
            )

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self._Score(result, metadata)


class CWEShuffleTask(CWETask):
    """The same content, but the test list's item blocks are shuffled.

    ``prepare_input`` is ``[head, *chunks, tail]`` — the original-order segments
    handed to reuse methods as a chunk-isolated context; ``run_input`` keeps
    ``head``/``tail`` in place and permutes ``chunks``, the test list's
    numbered items in fixed-size blocks. Reordering the items preserves their
    frequencies, so a cacheblend method that re-detects the segments and reuses
    the stored per-chunk KV can still name the same top-10 words; the naive
    control that blindly serves the stale stitched KV cannot. ``_Shuffled``
    guarantees the shuffled join differs from the original text.
    """

    name = "cwe_shuffle"

    def _SplitSegments(
        self, sample: Dict[str, Any], body: str, answerPrefix: str
    ) -> List[str]:
        """``[head, *chunks, tail]`` around the test example's list.

        ``head`` = user opener + few-shot example + test instruction;
        ``chunks`` = the test list's items split into ~10 fixed-size blocks
        (each block starts at a numbered-item boundary); ``tail`` = question +
        assistant header + answer prefix. ``join(segments)`` reconstructs the
        full chat prompt exactly.
        """
        testStart = body.rfind(_HeaderLine)
        if testStart < 0:
            raise RuntimeError("CWE instruction not found in input")
        qpos = body.rfind("Question: What are the 10 most common")
        if qpos < 0:
            raise RuntimeError("CWE question not found in input")
        # End of the test instruction line; the list begins right after it.
        nl = body.find("\n", testStart)
        hdrEnd = nl + 1 if nl >= 0 else testStart
        listText = body[hdrEnd:qpos]
        itemStarts = [m.start() for m in _ItemPattern.finditer(listText)]
        if not itemStarts:
            raise RuntimeError(f"no numbered items after the CWE instruction: {sample['file']}")

        nChunks = min(_NChunks, len(itemStarts))
        bounds = [0] + [
            itemStarts[k * len(itemStarts) // nChunks] for k in range(1, nChunks)
        ] + [len(listText)]
        modelPath = ModelPath()
        head = user_turn_prefix(modelPath) + body[:hdrEnd]
        chunks = [listText[bounds[k]:bounds[k + 1]] for k in range(nChunks)]
        tail = body[qpos:] + assistant_turn_suffix(modelPath) + answerPrefix
        return [head, *chunks, tail]

    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            body, answerPrefix = self._StripTemplate(s)
            segments = self._SplitSegments(s, body, answerPrefix)
            fullPrompt = "".join(segments)
            chunks = self._Shuffled(segments[1:-1], seed=int(s.get("index", i)))
            run_input = segments[0] + "".join(chunks) + segments[-1]
            yield Case(
                input=RAGInput(prepare_input=segments, run_input=run_input),
                workload=RAGWorkload(case_id=i, data=RAGInput(prepare_input=segments, run_input=run_input)),
                metadata=self._Metadata(i, s, fullPrompt),
            )
