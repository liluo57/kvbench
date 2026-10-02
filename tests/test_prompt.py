import pytest

from helpers.backends.Prompt import (
    ComposeInterleavedReuse,
    ComposeInterleavedReuseText,
    ComposeReuse,
)


@pytest.mark.parametrize(
    ("prepare", "run", "expected"),
    [
        ([], [7], (None, [7])),
        ([[1], [2]], [1, 2, 9], ([[1], [2]], [9])),
        ([[1], [2], [3]], [2, 1, 3, 9], ([[2], [1], [3]], [9])),
        ([[4], [4]], [4, 4, 5], ([[4], [4]], [5])),
        ([[], []], [8, 9], (None, [8, 9])),
    ],
)
def test_compose_reuse_finds_longest_prefix(prepare, run, expected):
    assert ComposeReuse(prepare, run) == expected


@pytest.mark.parametrize(
    ("prepare", "run", "expected"),
    [
        ([], [7, 8], [(None, [7, 8])]),
        ([[1, 2], [3]], [9, 3, 8, 1, 2, 7, 3], [
            (None, [9]), (1, [3]), (None, [8]), (0, [1, 2]),
            (None, [7]), (1, [3]),
        ]),
        ([[1, 2]], [], []),
        ([[], []], [8, 9], [(None, [8, 9])]),
        ([[1, 2]], [1, 2, 1, 2], [(0, [1, 2]), (0, [1, 2])]),
    ],
)
def test_token_and_text_interleaved_matcher(prepare, run, expected):
    assert ComposeInterleavedReuse(prepare, run) == expected


def test_text_matcher_preserves_repeated_chunk_selection():
    assert ComposeInterleavedReuseText(["ab", "cd"], "XcdYabZcd") == [
        (None, "X"), (1, "cd"), (None, "Y"),
        (0, "ab"), (None, "Z"), (1, "cd"),
    ]
