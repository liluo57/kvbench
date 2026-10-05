"""Tests for dependency analysis."""

from types import SimpleNamespace

import pytest
import torch

from methods.DependencyAnalysis import DependencyAnalysisMethod, _DependencyStats


def test_occurrences_align_exact_token_chunks_without_character_offsets():
    method = DependencyAnalysisMethod(skipK=1)

    occurrences, token_info = method._Occurrences(
        [10, 11, 20, 30],
        [10, 11, 20, 30],
        [[10, 11], [20, 30]],
    )

    assert len(occurrences) == 2
    assert [item.tokenPairs for item in occurrences] == [
        ((0, 0), (1, 1)),
        ((2, 0), (3, 1)),
    ]
    assert token_info == {
        0: (0, 0),
        1: (0, 1),
        2: (2, 0),
        3: (2, 1),
    }


def test_dependency_method_rejects_negative_skip_k():
    with pytest.raises(ValueError, match="skipK"):
        DependencyAnalysisMethod(skipK=-1)


class _UniformAttentionModule:
    num_key_value_groups = 1


class _UniformQwenModule:
    @staticmethod
    def repeat_kv(states, groups):
        assert groups == 1
        return states


def _UniformDependencyStats():
    token_info = {
        0: (0, 0),
        1: (0, 1),
        2: (2, 0),
        3: (2, 1),
    }
    stats = _DependencyStats(
        token_info,
        skipK=0,
        cciChunks=[(1, (0, 1)), (2, (2, 3))],
    )
    stats.SetQueryRows(0, 4)
    query = torch.zeros((1, 1, 4, 1))
    key = torch.zeros((1, 1, 4, 1))
    value = torch.zeros((1, 1, 4, 1))
    qwen = _UniformQwenModule()
    stats.Record(
        torch,
        qwen,
        _UniformAttentionModule(),
        query,
        key,
        value,
        1.0,
    )
    # A second module represents a second transformer layer.  The result must
    # be the same layer average, without storing either layer's attention.
    stats.Record(
        torch,
        qwen,
        _UniformAttentionModule(),
        query,
        key,
        value,
        1.0,
    )
    return stats.Values()


def test_cci_uses_causal_inter_intra_mass_and_token_weighted_output():
    values = _UniformDependencyStats()

    # With uniform causal attention, current chunk queries at positions 2 and
    # 3 put 2/3 and 1/2 mass on the prefix, while only position 3 puts 1/4
    # mass on an earlier token in its own chunk.
    expectedA = (2.0 / 3.0 + 1.0 / 2.0) / (2.0 * 2.0)
    expectedB = (1.0 / 4.0) / (2.0 * 2.0)
    expectedRatio = expectedA / expectedB
    expectedCci = 1.0 / (1.0 + torch.exp(torch.tensor(-expectedRatio)))

    assert values["missing_attention"] == pytest.approx(7.0 / 24.0)
    assert values["value_weighted_dependency"] == 0.0
    assert values["missing_attention_skip_k"] == pytest.approx(7.0 / 24.0)
    assert values["value_weighted_dependency_skip_k"] == 0.0
    assert values["cci"] == pytest.approx(float(expectedCci))
    assert values["cci_mean"] == pytest.approx(float(expectedCci))
    assert values["cci_max"] == pytest.approx(float(expectedCci))
    assert values["cci_raw_ratio_weighted"] == pytest.approx(expectedRatio)
    assert values["cci_chunks"] == [
        {
            "chunk_index": 2,
            "token_count": 2,
            "a_bar": pytest.approx(expectedA),
            "b_bar": pytest.approx(expectedB),
            "raw_ratio": pytest.approx(expectedRatio),
            "cci": pytest.approx(float(expectedCci)),
            "b_near_zero": False,
        }
    ]


def test_cci_is_unavailable_for_a_single_reusable_chunk():
    stats = _DependencyStats(
        {0: (0, 0), 1: (0, 1)},
        skipK=0,
        cciChunks=[(1, (0, 1))],
    )

    values = stats.CciValues()

    assert values["cci"] is None
    assert values["cci_chunks"] == []


def test_cci_exposes_and_clamps_a_near_zero_intra_denominator():
    stats = _DependencyStats(
        {0: (0, 0), 1: (1, 0)},
        skipK=0,
        cciChunks=[(1, (0,)), (2, (1,))],
    )
    stats.SetQueryRows(0, 2)
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.zeros((1, 1, 2, 1))
    stats.Record(
        torch,
        _UniformQwenModule(),
        _UniformAttentionModule(),
        query,
        key,
        value,
        1.0,
    )

    values = stats.Values()
    chunk = values["cci_chunks"][0]

    assert chunk["b_bar"] == 0.0
    assert chunk["b_near_zero"] is True
    assert values["cci_b_near_zero_chunks"] == [2]
    assert torch.isfinite(torch.tensor(values["cci"]))
    assert torch.isfinite(torch.tensor(chunk["raw_ratio"]))


def test_generation_consumes_native_ids_and_post_generation_analysis_counts_only_toward_total(clock):
    method = object.__new__(DependencyAnalysisMethod)
    def generate(ids, **kwargs):
        assert ids == [41, 42]
        clock.advance(0.04)
        return "answer", 0.03, 0.04, 1, None, [2]

    method._gen = SimpleNamespace(
        Generate=generate,
        OutputTokenIds=lambda token_ids: list(token_ids),
    )
    method._chunks = [[]]
    def analyze(*args):
        clock.advance(5)
        return {}

    method._Analyze = analyze

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] == pytest.approx(0.03)
    assert result.performance["total_time"] == pytest.approx(5.04)
    assert result.output_ids == [2]
