import pytest
import torch

from core.Sampling import ResolveSamplingConfig
from helpers.backends.TransformersHelper import TransformersGenerator


@pytest.mark.parametrize("penalty", [
    {"repetition_penalty": 2.0},
    {"presence_penalty": 2.0},
    {"frequency_penalty": 1.0},
])
def test_hf_argmax_applies_resolved_history_penalties_without_mutating_logits(penalty):
    sampler = object.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.samplingConfig = ResolveSamplingConfig(config={
        "ModelConfig": {"mode": "override", "do_sample": False, **penalty},
    })
    logits = torch.tensor([7.0, 6.0, -1.0])
    original = logits.clone()

    assert sampler._SampleNextToken(logits, [0, 0], None) == 1
    torch.testing.assert_close(logits, original)


def test_canonical_greedy_policy_remains_unmodified_argmax():
    sampler = object.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.samplingConfig = ResolveSamplingConfig(config={
        "ModelConfig": {"mode": "greedy", "presence_penalty": 100.0},
    })

    assert sampler._SampleNextToken(torch.tensor([7.0, 6.0, -1.0]), [0, 0], None) == 0
