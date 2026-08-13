"""RULER Needle-in-a-Haystack: the base task and its shuffle variant.

A magic number (the needle) is hidden mid-essay; the question asks for the
value tied to a word key. Data come from ``<DatasetPath>/ruler/niah_len*.jsonl``
(see :mod:`tasks._Ruler` for the record layout and chat-prompt wrapping).

Case payload contract
---------------------
:class:`NIAHTask`:
    ``prepare_input = []``             (no warm-up)
    ``run_input     = fullChatPrompt`` (the complete prompt, original order)

:class:`NIAHShuffleTask`:
    ``prepare_input = [prefix, *essays, needle, suffix]``  (original order)
    ``run_input     = prefix + shuffle([*essays, needle]).join("") + suffix``

``prefix`` (user opener + instruction) and ``suffix`` (question + assistant
header + answer prefix) are kept in place, so the shuffled prompt stays a
coherent, answerable chat — only the haystack chunks (``essays``) and the
needle sentence reorder. The needle value remains findable anywhere in the
reordered haystack, so a cacheblend method that re-detects the segments and
reuses the stored per-chunk KV still answers, while a naive one that blindly
serves the stale stitched KV fails.
"""

from typing import Any, Dict, Iterator, List, Tuple

from core.Result import Result
from core.Task import Case

from .TemplateHelper import AssistantSuffix, UserContext
from ._Ruler import _RulerBase


class NIAHTask(_RulerBase):
    """Needle-in-a-haystack: the needle is embedded mid-essay; ask for its value.

    The whole prompt is sent in original order with no warm-up — the recompute
    baseline; KV-reuse behaviour is exercised by NIAHShuffleTask.
    """

    name = "niah"
    taskName = "niah"

    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            _, fullPrompt = self._BuildChatParts(s, splitNeedle=True)
            yield Case(
                prepare_input=[],
                run_input=fullPrompt,
                metadata=self._Metadata(i, s, fullPrompt),
            )

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        return self._Score(result, metadata)


class NIAHShuffleTask(NIAHTask):
    """The same content, but the haystack chunks and needle are shuffled.

    ``prepare_input`` is ``[prefix, *essays, needle, suffix]`` — the original
    order segments handed to reuse methods as a chunk-isolated context. The
    ``prefix`` (user opener + instruction) and ``suffix`` (question + assistant
    header + answer prefix) stay in place, and ``run_input`` permutes only the
    middle: the haystack chunks and the needle sentence, so the prompt remains a
    coherent, answerable chat that no longer starts with the cached context.

    The point is to test whether reuse survives re-ordering: a cacheblend method
    re-detects the segments in their run order and reuses the stored per-chunk
    KV, while the naive control blindly serves the stale stitched KV and its
    accuracy shows exactly what the KV recombination breaks. ``_Shuffled``
    guarantees the shuffled pool's join differs from the original order (and
    from the original text).
    """

    name = "niah_shuffle"

    #: Number of haystack chunks the essay is split into (the shuffle pool is
    #: ``_NEssays`` essay chunks + the needle sentence).
    _NEssays = 4

    def Cases(self) -> Iterator[Case]:
        for i, s in enumerate(self._LoadSamples()):
            parts, fullPrompt = self._SplitSegments(s)
            shuffled = self._Shuffled(parts[1:-1], seed=int(s.get("index", i)))
            yield Case(
                prepare_input=parts,
                run_input=parts[0] + "".join(shuffled) + parts[-1],
                metadata=self._Metadata(i, s, fullPrompt),
            )

    # -------------------------------------------------------------- splitting
    def _SplitSegments(
        self, sample: Dict[str, Any]
    ) -> Tuple[List[str], str]:
        """``([prefix, *essays, needle, suffix], fullChatPrompt)``.

        ``prefix`` = user opener + the RULER instruction line; ``essays`` = the
        haystack text on both sides of the needle, split into ``_NEssays``
        chunks at sentence boundaries; ``needle`` = the needle sentence;
        ``suffix`` = question + assistant header + answer prefix.
        ``"".join(segments)`` reconstructs the full chat prompt exactly, so the
        shuffle pool ``segments[1:-1]`` (essays + needle) can be permuted freely
        without breaking the chat structure — the needle value stays locatable
        anywhere in the reordered haystack.
        """
        body, answerPrefix = self._StripTemplate(sample)

        instructionEnd = body.find("\n")
        if instructionEnd < 0:
            instructionEnd = len(body)
        questionStart = body.rfind("What is the special magic")
        if questionStart < 0:
            raise RuntimeError(f"NIAH question not found in input: {sample['file']}")

        prefix = UserContext(body[:instructionEnd])
        essay = body[instructionEnd:questionStart]
        value = str(sample["outputs"][0])
        vpos = essay.find(value)
        if vpos < 0:
            raise RuntimeError(f"needle value {value!r} not found in essay")
        # The needle sentence around the value, bounded to the essay: the span
        # runs from the last sentence-end before the value to the next one. The
        # value is unique in the haystack, so its sentence is the needle.
        start = essay.rfind(". ", 0, vpos)
        start = start + 2 if start >= 0 else 0
        end = essay.find(". ", vpos)
        end = end + 1 if end >= 0 else len(essay)
        needle = essay[start:end]

        before, after = essay[:start], essay[end:]
        nBefore = max(1, self._NEssays // 2)
        nAfter = max(1, self._NEssays - nBefore)
        essays = self._SplitInto(before, nBefore) + self._SplitInto(after, nAfter)
        suffix = body[questionStart:] + AssistantSuffix("") + answerPrefix

        parts = [prefix, *essays, needle, suffix]
        return parts, "".join(parts)

    @staticmethod
    def _NearestSentenceBoundary(text: str, target: int) -> int:
        """Index of a sentence boundary (``'. '`` / ``'.\\n'``) nearest ``target``."""
        if target <= 0:
            return 0
        if target >= len(text):
            return len(text)
        lo, hi = max(0, target - 200), min(len(text), target + 200)
        best, bestDist = None, None
        i = lo
        while i < hi - 1:
            if text[i] == "." and text[i + 1] in " \n":
                d = abs(i + 1 - target)
                if best is None or d < bestDist:
                    best, bestDist = i + 1, d
            i += 1
        return best if best is not None else target

    @classmethod
    def _SplitInto(cls, text: str, k: int) -> List[str]:
        """Split ``text`` into ``k`` contiguous chunks at sentence boundaries."""
        if not text:
            return []
        if k <= 1:
            return [text]
        bounds = [0]
        for j in range(1, k):
            bounds.append(cls._NearestSentenceBoundary(text, len(text) * j // k))
        bounds.append(len(text))
        return [text[bounds[i]:bounds[i + 1]] for i in range(k)]
