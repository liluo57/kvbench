"""Independent EPIC/LegoLink policy over KVBench's shared cache backend.

EPIC repairs the attention-sink prefix of each relocated chunk. The first
chunk, when it already occupies position zero, needs no boundary repair.
Fresh tokens are always forwarded by the shared execution layer. If a task
prepares the terminal query suffix too, EPIC repairs that full suffix.
"""

from helpers.backends.RecomputeMethod import CacheRecomputeMethod


class EPIC(CacheRecomputeMethod):
    name = "epic"

    def __init__(self, *args, recomputeTokens=32, recompute_tokens=None, **kwargs):
        budget = recomputeTokens if recompute_tokens is None else recompute_tokens
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("recomputeTokens must be a non-negative integer")
        self.recomputeTokens = budget
        super().__init__(*args, **kwargs)

    def _select(self, hits):
        return [position for hit in hits if hit.start > 0
                for position in range(hit.start, min(hit.end, hit.start + self.recomputeTokens))]

    def _select_for_prompt(self, hits, prompt_length):
        selected = self._select(hits)
        # EPIC recomputes the full free-form query suffix. Some KVBench shuffle
        # tasks include that suffix in Prepare, so it can arrive here as a hit
        # instead of as fresh input. Preserve the official query policy in that
        # case, even when the suffix is longer than the document repair budget.
        terminal_hits = [
            hit for hit in hits
            if hit.start > 0 and hit.end == prompt_length
        ]
        if terminal_hits:
            terminal = terminal_hits[-1]
            selected.extend(range(terminal.start, terminal.end))
        return selected


__all__ = ["EPIC"]
