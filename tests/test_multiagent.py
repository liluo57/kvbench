import unittest
from decimal import Decimal

from core.Config import ModelPath
from core.Result import Result
from core.Workload import ActionKind, ActionResult
from helpers.backends import ModelAdapter
from tasks.KVCommTasks import _extract_choice, _extract_number, _extract_python
from workload import AgentSpec, MultiAgentFullConnectionInput, MultiAgentFullConnectionWorkload


def _BuildExpectedPrompt(spec, task, priorOutputs, priorAgents):
    """Mirror :meth:`MultiAgentFullConnectionWorkload._BuildPrompt`'s
    user-prompt construction so the test can predict the input to
    :func:`ModelAdapter.render_chat`.
    """
    userPrompt = spec.promptTemplate.replace("{task}", task)
    if priorOutputs:
        userPrompt += (
            "\n\nAt the same time, the outputs of other agents are as follows:\n\n"
        )
        for i, output in enumerate(priorOutputs):
            role = priorAgents[i].role
            userPrompt += f"Agent {i}, role is {role}, output is:\n\n{output}\n\n"
    return userPrompt


class MultiAgentWorkloadTest(unittest.TestCase):
    def _run(self, decision):
        modelPath = ModelPath()
        agents = [
            AgentSpec("A", "a {task}"),
            AgentSpec("B", "b {task}"),
            AgentSpec("C", "c {task}"),
        ]
        w = MultiAgentFullConnectionWorkload(
            1, MultiAgentFullConnectionInput(
                "TASK", agents, decision,
                modelPath=modelPath,
            ),
        )
        prompts, retains, agentOutputs = [], [], []
        while not w.finished:
            action = w.next()[0]
            if action.kind == ActionKind.PREPARE:
                self.assertEqual(action.data, ["TASK"])
                w.observe([ActionResult(1, Result(), action.tag)])
                continue

            tag = action.tag
            if tag.startswith("agent_"):
                specIndex = int(tag.split("_")[1])
                spec = agents[specIndex]
                priorOutputs = agentOutputs[:specIndex]
            else:
                spec = decision
                priorOutputs = list(agentOutputs)
            expectedPrompt = ModelAdapter.render_chat(
                [{"role": "user", "content": _BuildExpectedPrompt(
                    spec, "TASK", priorOutputs, agents,
                )}],
                modelPath=modelPath,
                system_prefix=f"You are the {spec.role}.",
                thinking=False,
            )
            self.assertEqual(action.data, expectedPrompt)

            agentOutputs.append("OUT" + str(len(agentOutputs) + 1))
            prompts.append(action.data)
            retains.append(action.retainOutput)
            w.observe([ActionResult(1, Result(agentOutputs[-1]), action.tag)])
        return prompts, retains, w.final_result.output

    def test_decision_and_retain(self):
        prompts, retains, final = self._run(AgentSpec("D", "d {task}"))
        self.assertEqual(retains, [True, True, True, False])
        self.assertEqual(final, "OUT4")

    def test_no_decision_last_not_retained(self):
        prompts, retains, final = self._run(None)
        self.assertEqual(retains, [True, True, False]); self.assertEqual(final, "OUT3")

    def test_mmlu_does_not_match_letters_inside_prose(self):
        self.assertIsNone(_extract_choice("A detailed analysis without a final choice"))
        self.assertEqual(_extract_choice("analysis\nThe final answer is C"), "C")
        self.assertEqual(_extract_choice("**Final Answer**: D"), "D")
        self.assertEqual(_extract_choice("**Final Answer: Option B: text**"), "B")

    def test_gsm8k_prefers_explicit_final_answer(self):
        self.assertEqual(_extract_number("1 + 2 = 3\nThe answer is 3"), Decimal("3"))
        self.assertEqual(_extract_number("work\n#### 1,250", target=True), Decimal("1250"))

    def test_humaneval_accepts_fenced_and_raw_code(self):
        self.assertEqual(
            _extract_python("```python\ndef f():\n return 1\n```"),
            "def f():\n return 1",
        )
        self.assertEqual(_extract_python("def f():\n return 1"), "def f():\n return 1")


if __name__ == "__main__": unittest.main()