"""Tests for prompt."""

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


def test_prefix_matching_considers_alternatives_to_the_first_prepared_chunk():
    assert ComposeReuse([[1, 2], [1], [2, 3]], [1, 2, 3, 9]) == ([[1], [2, 3]], [9])


def test_prefix_matching_keeps_prepared_order_when_coverage_ties():
    first, second = [1], [1, 1]
    selected, suffix = ComposeReuse([first, second], [1, 1, 1, 9])
    assert selected[0] is first and selected[1] is second
    assert suffix == [9]


def test_interleaved_matcher_prefers_coverage_then_fewer_chunks():
    prepare = [[1, 2], [2, 3, 4], [1]]
    prompt = [1, 2, 3, 4, 9]
    assert ComposeInterleavedReuse(prepare, prompt) == [(2, [1]), (1, [2, 3, 4]), (None, [9])]
    assert ComposeInterleavedReuse([[1], [2], [1, 2]], [1, 2]) == [(2, [1, 2])]
    assert prepare == [[1, 2], [2, 3, 4], [1]] and prompt == [1, 2, 3, 4, 9]


def test_long_chunk_anchor_is_only_a_prefilter():
    prefix = list(range(64))
    assert ComposeInterleavedReuse([prefix + [100]], prefix + [101]) == [(None, prefix + [101])]
