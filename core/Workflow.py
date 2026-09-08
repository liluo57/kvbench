"""Workflow: stateful execution policy that drives the Engine–Method loop.

A Workflow describes execution semantics and decides what to execute next based
on previous execution results. It supports:
- Sequential execution (A → B → C)
- Fixed topology (A → (B, C) → D)
- Dynamic agent routing

Key constraint:
- Workflow does NOT directly call Method
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


# ── Action: unified execution request from Workflow to Engine ───────────────

@dataclass
class Action:
    """A single execution request.

    Action does NOT contain Method — Workflow and Method are fully decoupled.
    Engine decides whether to call Method.Prepare or Method.Run based on kind.

    Attributes:
        kind: PREPARE or RUN
        case_id: Identifies which Case this Action belongs to (for batch distinction)
        data: PREPARE → List[str] (warmup segments)
              RUN → str (complete prompt)
        tag: Optional step label, e.g. "agent_A", "agent_B"
        retainOutput: For RUN actions, whether generated output may be reused
            by later requests of the same case. This is a hint, not a guarantee.
    """
    kind: ActionKind
    case_id: int
    data: Any  # List[str] for PREPARE, str for RUN
    tag: str = ""
    retainOutput: bool = False


# ── ActionResult: result returned by Engine to Workflow ─────────────────────

@dataclass
class ActionResult:
    """Result of executing an Action, returned to Workflow.

    Attributes:
        case_id: Corresponds to Action.case_id
        result: Raw Result produced by Method
        tag: Corresponds to Action.tag
    """
    case_id: int
    result: Result
    tag: str = ""


# ── Workflow: stateful execution policy ────────────────────────────────────

class Workflow(ABC):
    """Stateful execution protocol.

    A workflow describes execution semantics, decides the next step, and
    determines subsequent behavior based on previous execution results.

    Key constraints:
    - Workflow must NOT directly call Method
    - All Actions produced by one next() call must have the same kind
      (either all PREPARE or all RUN), no mixing allowed
    """

    #: Case ID this workflow belongs to. Set by Task during Case construction.
    case_id: int = 0

    @abstractmethod
    def next(self) -> Optional[List[Action]]:
        """Return the next step's Actions to execute.

        All Actions in the returned list must have the same kind.
        Returns None if this Workflow has no more steps.

        Returns:
            List of Actions for this step, or None if finished.
        """

    @abstractmethod
    def observe(self, results: List[ActionResult]) -> None:
        """Receive execution results for the current step's Actions.

        Workflow updates internal state here for subsequent next() decisions.

        Args:
            results: Results for each Action from the current step.
        """

    @property
    @abstractmethod
    def finished(self) -> bool:
        """True if and only if the Workflow has completed all execution."""
