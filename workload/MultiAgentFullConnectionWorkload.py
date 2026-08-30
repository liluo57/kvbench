"""Single-round sequential complete-DAG multi-agent workload."""

from dataclasses import dataclass, field
from typing import List, Optional

from core.Config import ModelPath
from core.Result import Result
from core.Workload import Action, ActionKind, ActionResult, Workload


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


class MultiAgentFullConnectionWorkload(Workload):
    """Execute workers in creation order, then an optional decision agent."""

    def __init__(self, case_id: int, data: MultiAgentFullConnectionInput):
        self.case_id = case_id
        self._data = data
        self._prepared = not data.prepareSharedTask
        self._agentIndex = 0
        self._outputs: List[str] = []
        self._decisionDone = False
        self._finalResult: Optional[Result] = None

    def _BuildPrompt(self, spec: AgentSpec) -> str:
        userPrompt = spec.promptTemplate.replace("{task}", self._data.task)
        if self._outputs:
            userPrompt += (
                "\n\nAt the same time, the outputs of other agents are as follows:\n\n"
            )
            for i, output in enumerate(self._outputs):
                role = self._data.agents[i].role
                userPrompt += f"Agent {i}, role is {role}, output is:\n\n{output}\n\n"
        if not self._data.chatTemplate:
            return userPrompt

        # The agent always sees real system/user turns (and stops at the
        # model's chat-template close token); helpers.backends.ModelAdapter
        # .render_chat owns the per-arch kwargs (``enable_thinking``, etc.)
        # and the boundary strings, so the same workload renders correctly
        # for Qwen3 / Muse Glimmer / Mistral / any future arch. ``thinking``
        # is forced off: the multi-agent decision flow does not benefit
        # from a per-agent <think> trace and KV-reuse tasks would otherwise
        # eat the budget on reasoning.
        from helpers.backends.ModelAdapter import render_chat

        systemPrompt = spec.systemPrompt or f"You are the {spec.role}."
        return render_chat(
            [{"role": "user", "content": userPrompt}],
            modelPath=self._data.modelPath,
            system_prefix=systemPrompt,
            thinking=False,
        )

    def next(self) -> Optional[List[Action]]:
        if not self._prepared:
            self._prepared = True
            # The user task is present verbatim in every agent request. Make
            # it an explicit reusable unit instead of counting only previous
            # agent outputs as reusable communication.
            return [Action(
                ActionKind.PREPARE,
                self.case_id,
                [self._data.task],
                "prepare_shared_task",
            )]
        if self._agentIndex < len(self._data.agents):
            i = self._agentIndex
            self._agentIndex += 1
            retain = self._data.decisionAgent is not None or i < len(self._data.agents) - 1
            return [Action(ActionKind.RUN, self.case_id, self._BuildPrompt(self._data.agents[i]), f"agent_{i}", retain)]
        if self._data.decisionAgent is not None and not self._decisionDone:
            self._decisionDone = True
            return [Action(ActionKind.RUN, self.case_id, self._BuildPrompt(self._data.decisionAgent), "decision", False)]
        return None

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError("FullConnection expects exactly one ActionResult per step")
        if results[0].tag == "prepare_shared_task":
            return
        result = results[0].result
        self._finalResult = result
        if results[0].tag.startswith("agent_"):
            self._outputs.append("" if result.output is None else str(result.output))

    @property
    def finished(self) -> bool:
        return self._prepared and self._agentIndex >= len(self._data.agents) and (
            self._data.decisionAgent is None or self._decisionDone
        )

    @property
    def final_result(self) -> Optional[Result]:
        return self._finalResult