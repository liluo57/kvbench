"""Tests for kvcomm."""

from decimal import Decimal

import pytest

from core.Result import Result
from tasks.KVCommTasks import (
    KVCommHumanEvalTask,
    _extract_choice,
    _extract_number,
    _extract_python,
)


@pytest.mark.parametrize("text,expected", [
    ("A detailed analysis without a final choice", None),
    ("analysis\nThe final answer is C", "C"),
    ("**Final Answer**: D", "D"),
    ("**Final Answer: Option B: text**", "B"),
])
def test_mmlu_extraction_requires_an_answer_marker(text, expected):
    assert _extract_choice(text) == expected


def test_gsm8k_prefers_explicit_final_answer():
    assert _extract_number("1 + 2 = 3\nThe answer is 3") == Decimal("3")
    assert _extract_number("work\n#### 1,250", target=True) == Decimal("1250")


@pytest.mark.parametrize("text", ["```python\ndef f():\n return 1\n```", "def f():\n return 1"])
def test_humaneval_accepts_fenced_and_raw_code(text):
    assert _extract_python(text) == "def f():\n return 1"


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def add(a, b):\n    return a + b", 1.0),
        ("def add(a, b):\n    return -999", 0.0),
        ("def unrelated(a, b):\n    return a + b", 0.0),
        ("def add(a, b):\n    raise RuntimeError('broken')", 0.0),
    ],
)
def test_humaneval_executes_check_against_entry_point(code, expected):
    score = KVCommHumanEvalTask().Evaluate(
        Result(output=code),
        {
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
        },
    )
    assert score == {"accuracy": expected}


@pytest.mark.parametrize("metadata", [{}, {"test": "def check(candidate): pass"}])
def test_humaneval_missing_test_metadata_fails(metadata):
    assert KVCommHumanEvalTask().Evaluate(
        Result(output="def add(a, b): return a + b"), metadata
    ) == {"accuracy": 0.0}
