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