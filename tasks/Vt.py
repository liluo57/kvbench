"""RULER Variable Tracking: the base task and its shuffle variant.

Track the chain of ``VAR x = y`` assignments to a value; ``outputs`` is the
list of variable names assigned the queried value, all of which must appear in
the prediction. Data come from ``<DatasetPath>/ruler/vt_len*.jsonl`` (see
:mod:`tasks.bases.RulerBase` for the record layout and chat-prompt wrapping).

Dataset structure (what the shuffle keys on)
--------------------------------------------
RULER VT is few-shot: a record's ``input`` holds *two* examples concatenated —
a filled example, then the test example. Each example is an instruction, a
haystack of repeated noise sentences with the ``VAR x = ...`` assignments of a
single chain scattered through it, and a question. The last example is the test
one: the record's ``answer_prefix`` completes its question and ``outputs`` is
its chain's variable list.

The informative units are the chain assignments. Each is a self-contained fact
(``VAR x = VAR y`` / ``VAR x = <value>``), so the chain stays resolvable in any
order: the query value is found, then every ``VAR x = VAR y`` whose right side
was just resolved. That is exactly what makes a shuffle a fair test here — see
:class:`VTShuffleTask`.

Case payload contract
---------------------
:class:`VTTask`:
    ``input = RAGInput(prepare_input=[], run_input=fullChatPrompt)``
    ``workload = RAGWorkload`` (skip Prepare, just Run)

:class:`VTShuffleTask`:
    ``input = RAGInput(prepare_input=segments, run_input=shuffled_prompt)``
    ``workload = RAGWorkload`` (Prepare → Run)

``chain`` is the test example's assignments, each paired with the noise that
follows it. ``head`` (few-shot example + test instruction) and ``tail`` (test
question + assistant header + answer prefix) are kept in place so the shuffled
prompt stays a coherent, answerable prompt; only the assignments reorder.
"""

import re
from typing import Any, Dict, Iterator, List

from core.Result import Result
from core.Task import Case
from core.Workload import RAGInput, RAGWorkload

from .TemplateHelper import AssistantSuffix, UserContext
from .bases.RulerBase import RulerBase

#: A ``VAR x = y`` assignment: ``VAR ABCDE = 12345`` or ``VAR FGHIJ = VAR ABCDE``.
_ChainPattern = re.compile(r"VAR [A-Z]+ = (?:VAR [A-Z]+|\d+)")


class VTTask(RulerBase):
    """Variable tracking: track the chain of variable assignments to a value.

    ``outputs`` is the list of variable names assigned the queried value; all
    of them must appear in the prediction.
    """

    name = "vt"
    taskName = "vt"

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


class VTShuffleTask(VTTask):
    """The same content, but the test chain's assignments are shuffled.

    ``prepare_input`` is ``[head, *chain, tail]`` — the original-order segments
    handed to reuse methods as a chunk-isolated context; ``run_input`` keeps
    ``head``/``tail`` in place and permutes ``chain``, the test example's
    assignments (each with its trailing noise). The reorder changes the prompt
    text but leaves every assignment readable, so a cacheblend method that
    re-detects the segments and reuses the stored per-chunk KV can still resolve
    the chain; the naive control that blindly serves the stale stitched KV
    cannot. ``_Shuffled`` guarantees the shuffled join differs from the original
    text.
    """

    name = "vt_shuffle"

    def _SplitSegments(
        self, sample: Dict[str, Any], body: str, answerPrefix: str
    ) -> List[str]:
        """``[head, *chain, tail]`` around the test example's assignments.

        ``head`` = user opener + few-shot example + test instruction + the
        leading noise; ``chain`` = one segment per assignment of the test
        chain, each carrying the noise up to the next assignment; ``tail`` =
        trailing noise + question + assistant header + answer prefix.
        ``join(segments)`` reconstructs the full chat prompt exactly.
        """
        testStart = body.rfind("Memorize and track the chain")
        if testStart < 0:
            raise RuntimeError("VT instruction not found in input")
        qpos = body.rfind("Question: Find all variables")
        if qpos < 0:
            raise RuntimeError("VT question not found in input")
        # The test example is the last one; its assignments are after its header.
        starts = [m.start() for m in _ChainPattern.finditer(body) if m.start() >= testStart]
        if not starts:
            raise RuntimeError(f"no VAR assignments after the test instruction: {sample['file']}")

        head = UserContext(body[:starts[0]])
        chain = [
            body[starts[k]:starts[k + 1]] if k + 1 < len(starts)
            else body[starts[k]:qpos]
            for k in range(len(starts))
        ]
        tail = body[qpos:] + AssistantSuffix("") + answerPrefix
        return [head, *chain, tail]

    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            body, answerPrefix = self._StripTemplate(s)
            segments = self._SplitSegments(s, body, answerPrefix)
            fullPrompt = "".join(segments)
            chain = self._Shuffled(segments[1:-1], seed=int(s.get("index", i)))
            run_input = segments[0] + "".join(chain) + segments[-1]
            yield Case(
                input=RAGInput(prepare_input=segments, run_input=run_input),
                workload=RAGWorkload(case_id=i, data=RAGInput(prepare_input=segments, run_input=run_input)),
                metadata=self._Metadata(i, s, fullPrompt),
            )
