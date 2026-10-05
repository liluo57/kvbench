"""Tests for tokenization."""

from types import SimpleNamespace

from helpers.backends import ModelAdapter
from helpers.backends.Tokenization import EncodePrompt, OutputFragmentIds


def test_terminal_stop_override_replaces_default_and_keeps_explicit_stops():
    from helpers.backends.Tokenization import (
        OutputFragmentIds,
        TerminalStopIdsFromTokenizer,
    )
    tokenizer = SimpleNamespace(eos_token_id=2)
    stops = TerminalStopIdsFromTokenizer(tokenizer, {"eos_token_id": 9}, [8])
    assert stops == [8, 9]
    assert OutputFragmentIds([10, 2], stops) == [10, 2]
    assert OutputFragmentIds([2, 10, 9], stops) == [2, 10]
    assert TerminalStopIdsFromTokenizer(tokenizer, {"eos_token_id": []}) == []
    assert TerminalStopIdsFromTokenizer(tokenizer) == [2]


def test_task_tokenization_encodes_reusable_and_fresh_pieces_independently(
    monkeypatch,
):
    class BoundaryTokenizer:
        all_special_ids = []

        def encode(self, text, add_special_tokens=False):
            if text == "ab":
                return [900]
            return [ord(character) for character in text]

    monkeypatch.setattr(ModelAdapter, "_tokenizer", lambda _path: BoundaryTokenizer())
    prepareIds, runIds = EncodePrompt(["a"], "ab", "synthetic")

    assert prepareIds == [[ord("a")]]
    assert runIds == [ord("a"), ord("b")]
    assert runIds != [900]


def test_output_fragment_keeps_interior_specials_and_removes_terminal_stops():
    assert OutputFragmentIds([1, 99, 2, 99, 99], [99]) == [1, 99, 2]
    assert OutputFragmentIds([1, 99, 2]) == [1, 99, 2]
