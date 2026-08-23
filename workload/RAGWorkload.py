"""RAG workload: Prepare → Run execution policy."""

from dataclasses import dataclass, field
from typing import List, Optional

from core.Result import Result
from core.Workload import Action, ActionKind, ActionResult, Workload


@dataclass
class RAGInput:
    """Input data for RAG workload.

    This preserves backward compatibility with existing RAG benchmarks.
    The Task computes prepare_input and run_input during Cases() construction.

    Attributes:
        prepare_input: Text segments fed to Method.Prepare as warmup
            (e.g. the document whose KV cache is prefilled).
            Empty list means no warmup.
        run_input: The complete prompt fed to Method.Run.
    """
    prepare_input: List[str] = field(default_factory=list)
    run_input: str = ""


class RAGWorkload(Workload):
    """Simplest execution policy: Prepare → Run, two steps max.

    This perfectly expresses existing RAG benchmark semantics:
    - prepare(documents) → warm up KV cache
    - run(question_prompt) → execute inference

    If prepare_input is empty, skips Prepare and goes directly to Run.
    """

    def __init__(self, case_id: int, data: RAGInput):
        self.case_id = case_id
        self._data = data
        self._step = 0  # 0: not started, 1: prepare done, 2: run done
        self._final_result: Optional[Result] = None

    def next(self) -> Optional[List[Action]]:
        if self._step == 0:
            if self._data.prepare_input:
                # Has prepare phase
                self._step = 1
                return [Action(
                    kind=ActionKind.PREPARE,
                    case_id=self.case_id,
                    data=self._data.prepare_input,
                )]
            else:
                # No prepare phase (e.g. NIAHTask, FullPrefill), skip to Run
                self._step = 1
                return self.next()

        if self._step == 1:
            self._step = 2
            return [Action(
                kind=ActionKind.RUN,
                case_id=self.case_id,
                data=self._data.run_input,
            )]

        # Step >= 2: finished
        return None

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError("RAGWorkload expects exactly one ActionResult per step")
        # ``next`` advances _step before the action is executed. Step 1 is the
        # optional PREPARE placeholder; step 2 is the real RUN result, whose
        # output may legitimately be None while diagnostics remain useful.
        if self._step == 2:
            self._final_result = results[0].result

    @property
    def finished(self) -> bool:
        return self._step >= 2

    @property
    def final_result(self) -> Optional[Result]:
        """The result from the Run step."""
        return self._final_result
