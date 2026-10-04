"""Shared cache-backend token selection (KV Packet MIT-licensed runtime)."""

from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor

@dataclass(frozen=True)
class SamplingConfig:
    """Greedy or stochastic first-token and continuation selection.

    Attributes:
        do_sample: Whether to sample instead of selecting the maximum logit.
        temperature: Positive sampling temperature.
        top_p: Cumulative-probability threshold in ``[0, 1]``; zero disables.
        top_k: Maximum number of candidates; zero disables this filter.
        min_p: Minimum probability relative to the most likely token.
        typical_p: Locally typical probability mass to retain.
        repetition_penalty: Multiplicative penalty for tokens already in history.
        presence_penalty: Additive penalty for tokens already in history.
        frequency_penalty: Additive penalty scaled by each token's count.
        seed: Optional per-request sampling seed.
    """

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate sampling bounds before any model work begins."""
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if not 0 <= self.top_p <= 1 or not math.isfinite(self.top_p):
            raise ValueError("top_p must be finite and in [0, 1]")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0 <= self.min_p <= 1 or not math.isfinite(self.min_p):
            raise ValueError("min_p must be finite and in [0, 1]")
        if not 0 <= self.typical_p <= 1 or not math.isfinite(self.typical_p):
            raise ValueError("typical_p must be finite and in [0, 1]")
        if self.repetition_penalty <= 0 or not math.isfinite(self.repetition_penalty):
            raise ValueError("repetition_penalty must be finite and positive")
        for name, value in (
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


def make_generator(sampling: SamplingConfig, device: torch.device | str) -> torch.Generator | None:
    """Create a request-local random generator when a seed was configured."""
    if sampling.seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling.seed)
    return generator


def select_next_token(
    logits: Tensor,
    sampling: SamplingConfig,
    history: Sequence[int] = (),
    generator: torch.Generator | None = None,
) -> int:
    """Select one token, applying history penalties for both greedy and sampling."""
    has_history_penalty = (
        sampling.repetition_penalty != 1.0
        or sampling.presence_penalty != 0.0
        or sampling.frequency_penalty != 0.0
    )
    if not sampling.do_sample and not has_history_penalty:
        return int(logits.argmax(dim=-1))

    scores = logits.float().clone()
    if history and has_history_penalty:
        counts: dict[int, int] = {}
        for token in history:
            counts[int(token)] = counts.get(int(token), 0) + 1
        if sampling.presence_penalty or sampling.frequency_penalty:
            for token, count in counts.items():
                scores[..., token] -= (
                    sampling.presence_penalty + sampling.frequency_penalty * count
                )
        if sampling.repetition_penalty != 1.0:
            token_ids = torch.tensor(
                list(counts), dtype=torch.long, device=scores.device
            )
            selected = scores.index_select(-1, token_ids)
            penalized = torch.where(
                selected < 0,
                selected * sampling.repetition_penalty,
                selected / sampling.repetition_penalty,
            )
            scores.scatter_(-1, token_ids.expand(*scores.shape[:-1], -1), penalized)

    if not sampling.do_sample:
        return int(scores.argmax(dim=-1))

    scores = scores / sampling.temperature
    vocab_size = int(scores.shape[-1])
    if 0 < sampling.top_k < vocab_size:
        threshold = torch.topk(scores, sampling.top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -float("inf"))
    if 0.0 < sampling.typical_p < 1.0:
        probabilities = torch.softmax(scores, dim=-1)
        log_probs = torch.log(probabilities.clamp_min(1e-20))
        entropy = -(probabilities * log_probs).sum(dim=-1, keepdim=True)
        shifted = (-log_probs - entropy).abs()
        _, sorted_indices = torch.sort(shifted, descending=False, dim=-1)
        sorted_probs = probabilities.gather(-1, sorted_indices)
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_remove = cumulative > sampling.typical_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, -float("inf"))
    if 0.0 < sampling.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative > sampling.top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, -float("inf"))
    if sampling.min_p > 0.0:
        threshold = scores.amax(dim=-1, keepdim=True) + math.log(sampling.min_p)
        scores = scores.masked_fill(scores < threshold, -float("inf"))

    probabilities = torch.softmax(scores, dim=-1)
    return int(
        torch.multinomial(probabilities, num_samples=1, generator=generator).item()
    )


__all__ = ["SamplingConfig", "make_generator", "select_next_token"]
