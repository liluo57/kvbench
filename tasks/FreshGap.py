"""Synthetic sanity test for reuse across an intermediate fresh span.

Structure:

    Prepare: [A, C]
    Run:      A + [fresh B] + C + Q

``Q`` is the fresh assistant-generation suffix.  It is intentionally not part
of C: HYPIC reserves the final PIC segment for the query whose logits seed
decode. Keeping Q separate lets the benchmark test reuse of the *final
prepared segment* C without bypassing that runtime contract.

A and C are both long and roughly equal in size. B is only a few tokens.

Expected behaviour:

- Intermediate-fresh reuse works:
    A and C are both served from cached KV.
    reuse_ratio should be > 0.9 (normally close to 1.0).
    TTFT should be much lower than full prefill; ~4x is a reasonable
    empirical expectation for the current CacheBlend setup.

- Only prefix reuse works:
    only A is reused; B + C are recomputed.
    reuse_ratio should be around 0.5.
    TTFT should be roughly ~2x faster than full prefill.

B contains the answer while the question lives at the end of C. Therefore a
method that reuses C without correctly blending the fresh B can also be caught
by the accuracy metric.
"""

from typing import Any, Dict, Iterator

from core.Config import ModelPath
from core.Result import Result
from core.Task import Case, Task
from workload.RAGWorkload import RAGInput, RAGWorkload

from helpers.backends.ModelAdapter import assistant_turn_suffix, user_turn_prefix


class FreshGapTask(Task):
    name = "fresh_gap"

    def __init__(
        self,
        nCases: int = 4,
        linesPerChunk: int = 192,
    ):
        self.nCases = nCases
        self.linesPerChunk = linesPerChunk
        modelPath = ModelPath()

        # Keep A / C identical across cases. Only the tiny fresh B changes.
        # This makes the performance comparison less noisy.
        # This is an exact-answer microbenchmark; disable the model's optional
        # chain-of-thought preamble so the answer fits in the smoke budget.
        thinking = False
        self._a = user_turn_prefix(modelPath, thinking=thinking) + (
            "Read the following context and answer the final question. "
            "Return only the requested numeric code.\n\n"
            + self._Filler("A")
        )

        self._c = (
            self._Filler("C")
            + "\nWhat is the transient code? "
            "Answer with the numeric code only.\n"
        )
        # HYPIC's last segment is the query segment and is deliberately not
        # cached. Keep the model's assistant-generation suffix outside C so C
        # can be a true final prepared hit while Q supplies the final forward
        # row used for next-token sampling.
        self._query_tail = assistant_turn_suffix(modelPath, thinking=thinking)

    def Cases(self) -> Iterator[Case]:
        for i in range(self.nCases):
            # Intentionally tiny compared with A and C.
            answer = str(731946 + i)
            fresh = f"\nThe transient code is {answer}.\n"

            data = RAGInput(
                prepare_input=[self._a, self._c],
                run_input=self._a + fresh + self._c + self._query_tail,
            )

            yield Case(
                input=data,
                workload=RAGWorkload(case_id=i, data=data),
                metadata={
                    "answer": answer,
                    "fresh": fresh,
                },
            )

    def Evaluate(
        self,
        result: Result,
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        answer = metadata["answer"]
        output = result.output or ""

        return {
            "accuracy": float(answer in output),
        }

    def _Filler(self, label: str) -> str:
        # Each line is intentionally simple and tokenizer-friendly.
        # 192 lines gives each side a few thousand tokens on common 7B models:
        # large enough that 50% vs ~100% reuse is unmistakable in both
        # reuse_ratio and TTFT.
        return "".join(
            f"{label} record {i:04d}: "
            "amber cedar orbit marble river copper lantern winter.\n"
            for i in range(self.linesPerChunk)
        )
