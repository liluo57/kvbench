import unittest
from decimal import Decimal

from core.Result import Result
from core.Workload import ActionKind, ActionResult
from tasks.KVCommTasks import _extract_choice, _extract_number, _extract_python
from workload import AgentSpec, MultiAgentFullConnectionInput, MultiAgentFullConnectionWorkload


class MultiAgentWorkloadTest(unittest.TestCase):
    def _run(self, decision):
        w = MultiAgentFullConnectionWorkload(1, MultiAgentFullConnectionInput(
            "TASK", [AgentSpec("A", "a {task}"), AgentSpec("B", "b {task}"), AgentSpec("C", "c {task}")], decision))
        prompts, retains = [], []
        while not w.finished:
            action = w.next()[0]
            if action.kind == ActionKind.PREPARE:
                self.assertEqual(action.data, ["TASK"])
                w.observe([ActionResult(1, Result(), action.tag)])
                continue
            self.assertTrue(action.data.startswith("<|im_start|>system\n"))
            self.assertTrue(action.data.endswith("<think>\n\n</think>\n\n"))
            prompts.append(action.data); retains.append(action.retainOutput)
            w.observe([ActionResult(1, Result("OUT" + str(len(prompts))), action.tag)])
        return prompts, retains, w.final_result.output

    def test_decision_and_retain(self):
        prompts, retains, final = self._run(AgentSpec("D", "d {task}"))
        self.assertEqual(retains, [True, True, True, False])
        self.assertIn("OUT1", prompts[1]); self.assertIn("OUT1", prompts[3]); self.assertEqual(final, "OUT4")

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
