import pytest

from core.Result import Result
from tasks.KVCommTasks import KVCommHumanEvalTask


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
