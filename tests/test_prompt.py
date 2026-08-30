import pytest

from helpers.backends.Prompt import ComposeInterleavedReuse, ComposeReuse, ContextText


def test_context_text_handles_empty_and_preserves_segments():
    assert ContextText([]) == ""
    assert ContextText(["a", "", "b"]) == "ab"


@pytest.mark.parametrize(
    ("prepare", "run", "expected"),
    [
        ([], "prompt", (None, "prompt")),
        (["A", "B"], "ABtail", (["A", "B"], "tail")),
        (["A", "B", "C"], "BACtail", (["B", "A", "C"], "tail")),
        (["same", "same"], "samesame!", (["same", "same"], "!")),
        (["A"], "unmatched", (None, "unmatched")),
    ],
)
def test_compose_reuse_finds_longest_prefix(prepare, run, expected):
    assert ComposeReuse(prepare, run) == expected


def test_interleaved_reuse_keeps_fresh_spans_and_repeats_chunks():
    parts = ComposeInterleavedReuse(["ab", "cd"], "XcdYabZcd")

    assert parts == [
        (None, "X"),
        (1, "cd"),
        (None, "Y"),
        (0, "ab"),
        (None, "Z"),
        (1, "cd"),
    ]
    assert "".join(text for _, text in parts) == "XcdYabZcd"


@pytest.mark.parametrize(
    ("prepare", "run", "expected"),
    [
        (["abc"], "", []),
        ([], "abc", [(None, "abc")]),
        (["", ""], "abc", [(None, "abc")]),
        (["abc", "bc", "c"], "abc", [(0, "abc")]),
        (["aba", "bab"], "abab", [(0, "aba"), (None, "b")]),
        (["xyz"], "abc", [(None, "abc")]),
    ],
)
def test_interleaved_reuse_edge_cases(prepare, run, expected):
    assert ComposeInterleavedReuse(prepare, run) == expected


def test_interleaved_anchor_is_only_a_prefilter():
    common = "x" * 64
    prepare = [common + "A", common + "B"]
    run = "fresh" + common + "B" + "tail"

    assert ComposeInterleavedReuse(prepare, run) == [
        (None, "fresh"),
        (1, common + "B"),
        (None, "tail"),
    ]

