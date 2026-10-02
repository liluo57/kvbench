"""Single-round sequential complete-DAG multi-agent workflow."""

from dataclasses import dataclass, field
from typing import List, Optional

from core.Config import ModelPath
from core.Result import Result
from core.Workflow import Action, ActionKind, ActionResult, Workflow
from helpers.backends.Tokenization import (
    EncodeChunks,
    EncodeText,
)


@dataclass(frozen=True)
class AgentSpec:
    role: str
    promptTemplate: str
    systemPrompt: str = ""


@dataclass
class MultiAgentFullConnectionInput:
    task: str
    agents: List[AgentSpec]
    decisionAgent: Optional[AgentSpec] = None
    prepareSharedTask: bool = True
    chatTemplate: bool = True
    #: Path to the model whose chat template wraps each agent's prompt.
    #: Used only when ``chatTemplate=True``; :func:`render_chat` loads the
    #: tokenizer from this path. Defaults to the framework-wide
    #: ``ModelPath()`` so existing callers (the four KVComm tasks) keep
    #: working without having to thread the path through their constructors.
    modelPath: str = field(default_factory=ModelPath)


class MultiAgentFullConnectionWorkflow(Workflow):
    """Execute workers in creation order, then an optional decision agent."""

    def __init__(self, case_id: int, data: MultiAgentFullConnectionInput):
        self.case_id = case_id
        self._data = data
        # PREPARE and each RUN occurrence use these exact independently
        # tokenized shared-task IDs.
        self._taskIds = EncodeChunks([data.task], data.modelPath)[0]
        self._prepared = not data.prepareSharedTask
        self._agentIndex = 0
        self._outputs: List[tuple[str, Optional[List[int]]]] = []
        self._decisionDone = False
        self._finalResult: Optional[Result] = None

    def _BuildPrompt(self, spec: AgentSpec) -> List[int]:
        contentParts: List[tuple[str, Optional[List[int]]]] = []

        def appendText(text: str) -> None:
            if text:
                contentParts.append((text, None))

        def appendIds(displayText: str, ids: List[int]) -> None:
            contentParts.append((displayText, ids))

        templateParts = spec.promptTemplate.split("{task}")
        for index, text in enumerate(templateParts):
            appendText(text)
            if index < len(templateParts) - 1:
                appendIds(self._data.task, self._taskIds)
        if self._outputs:
            appendText(
                "\n\nAt the same time, the outputs of other agents are as follows:\n\n"
            )
            for i, (output, outputIds) in enumerate(self._outputs):
                role = self._data.agents[i].role
                appendText(
                    f"Agent {i}, role is {role}, output is:\n\n"
                )
                if outputIds is None:
                    raise RuntimeError(
                        "reused agent output is missing its native token IDs"
                    )
                appendIds(output, outputIds)
                appendText("\n\n")

        if self._data.chatTemplate:
            systemPrompt = spec.systemPrompt or f"You are the {spec.role}."
            from helpers.backends.ModelAdapter import _chat_boundary_parts

            prefix, suffix = _chat_boundary_parts(
                self._data.modelPath, False, systemPrompt
            )
        else:
            prefix, suffix = "", ""

        pieces: List[List[int]] = []
        freshParts: List[str] = [prefix] if prefix else []
        for text, ids in contentParts:
            if ids is None:
                freshParts.append(text)
            else:
                freshText = "".join(freshParts)
                if freshText:
                    pieces.append(EncodeText(freshText, self._data.modelPath))
                freshParts.clear()
                pieces.append(ids)
        # Include both template boundaries in their adjacent fresh spans.
        if suffix:
            freshParts.append(suffix)
        freshText = "".join(freshParts)
        if freshText:
            pieces.append(EncodeText(freshText, self._data.modelPath))
        return [token for piece in pieces for token in piece]

    def next(self) -> Optional[List[Action]]:
        if not self._prepared:
            self._prepared = True
            # The user task is present verbatim in every agent request. Make
            # it an explicit reusable unit instead of counting only previous
            # agent outputs as reusable communication.
            return [Action(
                ActionKind.PREPARE,
                self.case_id,
                [self._taskIds],
                "prepare_shared_task",
            )]
        if self._agentIndex < len(self._data.agents):
            i = self._agentIndex
            self._agentIndex += 1
            retain = self._data.decisionAgent is not None or i < len(self._data.agents) - 1
            promptIds = self._BuildPrompt(self._data.agents[i])
            return [Action(ActionKind.RUN, self.case_id, promptIds, f"agent_{i}", retain)]
        if self._data.decisionAgent is not None and not self._decisionDone:
            self._decisionDone = True
            promptIds = self._BuildPrompt(self._data.decisionAgent)
            return [Action(ActionKind.RUN, self.case_id, promptIds, "decision", False)]
        return None

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError("FullConnection expects exactly one ActionResult per step")
        if results[0].tag == "prepare_shared_task":
            return
        result = results[0].result
        self._finalResult = result
        if results[0].tag.startswith("agent_"):
            output = "" if result.output is None else str(result.output)
            # Result.output_ids is the authoritative Method fragment. Methods
            # exclude only the terminal stop token; preserve every returned ID
            # exactly so retained KV and the next prompt use identical input.
            outputIds = result.output_ids
            self._outputs.append((output, outputIds))

    @property
    def finished(self) -> bool:
        return self._prepared and self._agentIndex >= len(self._data.agents) and (
            self._data.decisionAgent is None or self._decisionDone
        )

    @property
    def final_result(self) -> Optional[Result]:
        return self._finalResult
