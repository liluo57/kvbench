"""CacheBlend dialogue summarisation (``example/blend_samsum.py``).

Dataset: ``<DatasetPath>/samsum``. See :mod:`tasks._Cacheblend` for the record
layout, chat-prompt wrapping and scoring.
"""

from typing import Any, Dict, List, Tuple

from core.Result import Result

from ._Cacheblend import _KBBase, ParseGeneration, RougeL


class SamsumTask(_KBBase):
    """Dialogue summarisation (cacheblend ``blend_samsum.py``)."""

    name = "samsum"
    defaultDataset = "samsum"
    prefixPrompt = (
        "Summarize the dialogue into a few short sentences. The following are "
        "some examples.\n\n"
    )

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        chunks = [c["text"] for c in sample.get("ctxs", [])]
        suffix = "\n\n" + sample["question"]
        chunks = [self.prefixPrompt + chunks[0]] + chunks[1:] if chunks else []
        return chunks, suffix

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        pred = ParseGeneration(result.output)
        answers = metadata["answers"]
        best = max((RougeL(pred, a) for a in answers), default=0.0)
        return {"rougeL": best}
