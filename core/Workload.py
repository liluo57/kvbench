"""Workload: stateful execution policy that drives the Engine–Method loop.

A Workload describes execution semantics and decides what to execute next based
on previous execution results. It supports:
- Sequential execution (A → B → C)
- Fixed topology (A → (B, C) → D)
- Dynamic agent routing

Key constraint:
- Workload does NOT directly call Method
- All actions in one step must have the same kind (all PREPARE or all RUN)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .Result import Result


# ── ActionKind: distinguishes prepare vs run ────────────────────────────────

class ActionKind(Enum):
    """Type of action the Engine should execute."""
    PREPARE = "prepare"
    RUN = "run"


# ── Action: unified execution request from Workload to Engine ───────────────

@dataclass
class Action:
    """A single execution request.

    Action does NOT contain Method — Workload and Method are fully decoupled.
    Engine decides whether to call Method.Prepare or Method.Run based on kind.

    Attributes:
        kind: PREPARE or RUN
        case_id: Identifies which Case this Action belongs to (for batch distinction)
        data: PREPARE → List[str] (warmup segments)
              RUN → str (complete prompt)
        tag: Optional step label, e.g. "agent_A", "agent_B"
    """
    kind: ActionKind
    case_id: int
    data: Any  # List[str] for PREPARE, str for RUN
    tag: str = ""


# ── ActionResult: result returned by Engine to Workload ─────────────────────

@dataclass
class ActionResult:
    """Result of executing an Action, returned to Workload.

    Attributes:
        case_id: Corresponds to Action.case_id
        result: Raw Result produced by Method
        tag: Corresponds to Action.tag
    """
    case_id: int
    result: Result
    tag: str = ""


# ── Workload: stateful execution policy ────────────────────────────────────

class Workload(ABC):
    """Stateful execution protocol.

    A workload describes execution semantics, decides the next step, and
    determines subsequent behavior based on previous execution results.

    Key constraints:
    - Workload must NOT directly call Method
    - All Actions produced by one next() call must have the same kind
      (either all PREPARE or all RUN), no mixing allowed
    """

    #: Case ID this workload belongs to. Set by Task during Case construction.
    case_id: int = 0

    @abstractmethod
    def next(self) -> Optional[List[Action]]:
        """Return the next step's Actions to execute.

        All Actions in the returned list must have the same kind.
        Returns None if this Workload has no more steps.

        Returns:
            List of Actions for this step, or None if finished.
        """

    @abstractmethod
    def observe(self, results: List[ActionResult]) -> None:
        """Receive execution results for the current step's Actions.

        Workload updates internal state here for subsequent next() decisions.

        Args:
            results: Results for each Action from the current step.
        """

    @property
    @abstractmethod
    def finished(self) -> bool:
        """True if and only if the Workload has completed all execution."""


# ── RAG workload support ───────────────────────────────────────────────────

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
        # RAG doesn't need to make decisions based on results
        # But we store the last result for potential use
        if results and results[0].result.output is not None:
            self._final_result = results[0].result

    @property
    def finished(self) -> bool:
        return self._step >= 2

    @property
    def final_result(self) -> Optional[Result]:
        """The result from the Run step."""
        return self._final_result