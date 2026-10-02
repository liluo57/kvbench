import unittest
from decimal import Decimal
from unittest.mock import patch

from core.Result import Result
from core.Workflow import ActionKind, ActionResult
from helpers.backends import ModelAdapter
from tasks.KVCommTasks import _extract_choice, _extract_number, _extract_python
from workflow import AgentSpec, MultiAgentFullConnectionInput, MultiAgentFullConnectionWorkflow


def _BuildExpectedPrompt(spec, task, priorOutputs, priorAgents):
    """Mirror the raw user-prompt construction used by the workflow."""
    userPrompt = spec.promptTemplate.replace("{task}", task)
    if priorOutputs:
        userPrompt += (
            "\n\nAt the same time, the outputs of other agents are as follows:\n\n"
        )
        for i, output in enumerate(priorOutputs):
            role = priorAgents[i].role
            userPrompt += f"Agent {i}, role is {role}, output is:\n\n{output}\n\n"
    return userPrompt


class MultiAgentWorkflowTest(unittest.TestCase):
    def setUp(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(character) for character in text]

        self.tokenizerPatch = patch.object(
            ModelAdapter, "_tokenizer", lambda _path: FakeTokenizer()
        )
        self.tokenizerPatch.start()
        self.addCleanup(self.tokenizerPatch.stop)

    def _run(self, decision):
        agents = [
            AgentSpec("A", "a {task}"),
            AgentSpec("B", "b {task}"),
            AgentSpec("C", "c {task}"),
        ]
        w = MultiAgentFullConnectionWorkflow(
            1, MultiAgentFullConnectionInput(
                "TASK", agents, decision,
                chatTemplate=False,
                modelPath="synthetic",
            ),
        )
        prompts, retains, agentOutputs = [], [], []
        while not w.finished:
            action = w.next()[0]
            if action.kind == ActionKind.PREPARE:
                self.assertEqual(action.data, [[ord(character) for character in "TASK"]])
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
            expectedPrompt = _BuildExpectedPrompt(
                spec, "TASK", priorOutputs, agents,
            )
            self.assertEqual(
                action.data, [ord(character) for character in expectedPrompt]
            )

            agentOutputs.append("OUT" + str(len(agentOutputs) + 1))
            prompts.append(expectedPrompt)
            retains.append(action.retainOutput)
            w.observe([ActionResult(
                1,
                Result(
                    agentOutputs[-1],
                    output_ids=[ord(character) for character in agentOutputs[-1]],
                ),
                action.tag,
            )])
        return prompts, retains, w.final_result.output

    def test_decision_and_retain(self):
        prompts, retains, final = self._run(AgentSpec("D", "d {task}"))
        self.assertEqual(retains, [True, True, True, False])
        self.assertEqual(final, "OUT4")

    def test_no_decision_last_not_retained(self):
        prompts, retains, final = self._run(None)
        self.assertEqual(retains, [True, True, False]); self.assertEqual(final, "OUT3")

    def test_native_output_pieces_do_not_match_by_display_text(self):
        agents = [
            AgentSpec("A", "A {task}"),
            AgentSpec("B", "B {task}"),
            AgentSpec("C", "C {task}"),
        ]
        workflow = MultiAgentFullConnectionWorkflow(
            7,
            MultiAgentFullConnectionInput(
                "TASK", agents, AgentSpec("D", "D {task}"),
                chatTemplate=False, modelPath="synthetic",
            ),
        )
        prepare = workflow.next()[0]
        preparedTask = prepare.data[0]
        workflow.observe([ActionResult(7, Result(), prepare.tag)])

        fragments = ([900, 999], [901, 999], [902, 999])
        displays = ("same", "same", "")
        for index, (display, fragment) in enumerate(zip(displays, fragments)):
            action = workflow.next()[0]
            self.assertEqual(action.data.count(preparedTask[0]), 1)
            workflow.observe([
                ActionResult(
                    7,
                    Result(output=display, output_ids=list(fragment)),
                    action.tag,
                )
            ])

        decision = workflow.next()[0]
        for fragment in fragments:
            self.assertTrue(any(
                decision.data[index : index + len(fragment)] == list(fragment)
                for index in range(len(decision.data) - len(fragment) + 1)
            ))
        # Distinct native IDs for equal display text remain distinct, and the
        # empty display fragment still occupies its native token position.
        self.assertEqual(
            [decision.data.index(fragment[0]) for fragment in fragments],
            sorted(decision.data.index(fragment[0]) for fragment in fragments),
        )

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
