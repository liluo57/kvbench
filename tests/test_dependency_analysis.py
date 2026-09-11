import pytest

from methods.DependencyAnalysis import DependencyAnalysisMethod


class _BoundaryTokenizer:
    """Tiny tokenizer fixture with a token merged across a chunk boundary."""

    prompt = "A.B C"

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert text == self.prompt
        return {
            "input_ids": [10, 20, 30],
            # Token 10 contains the final character of chunk 0 and the first
            # character of chunk 1.
            "offset_mapping": [(0, 3), (3, 4), (4, 5)],
        }

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {
            "A.": [11],
            "B C": [20, 30],
        }[text]


class _Generator:
    def __init__(self):
        self.tokenizer = _BoundaryTokenizer()

    def Encode(self, text, *, addSpecialTokens=False):
        if text == _BoundaryTokenizer.prompt:
            return [10, 20, 30]
        return self.tokenizer.encode(
            text,
            add_special_tokens=addSpecialTokens,
        )


def test_occurrences_tolerate_bpe_merges_at_reuse_boundaries():
    method = DependencyAnalysisMethod(skipK=1)
    method._gen = _Generator()

    occurrences, token_info = method._Occurrences(
        _BoundaryTokenizer.prompt,
        [10, 20, 30],
        ["A.", "B C"],
    )

    assert len(occurrences) == 2
    # The merged boundary token has no exact independent counterpart and is
    # omitted; the remaining two rows are safely aligned.
    assert sum(len(item.tokenPairs) for item in occurrences) == 2
    assert token_info == {
        1: (1, 0),
        2: (1, 1),
    }


def test_dependency_method_rejects_negative_skip_k():
    with pytest.raises(ValueError, match="skipK"):
        DependencyAnalysisMethod(skipK=-1)
