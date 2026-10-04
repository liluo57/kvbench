"""Independent EPIC/LegoLink policy over KVBench's shared cache backend.

EPIC repairs the attention-sink prefix of each relocated chunk. The first
chunk, when it already occupies position zero, needs no boundary repair.
Fresh tokens are always forwarded by the shared execution layer.
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


__all__ = ["EPIC"]
