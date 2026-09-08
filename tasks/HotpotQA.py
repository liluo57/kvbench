"""LongBench HotpotQA task.

The checked-in snapshot lives under ``data/hotpotqa`` and is loaded locally;
this task has no dependency on the HYPIC checkout.  HotpotQA is scored with
the standard normalized token F1 and exact-match accuracy used by KVBench's
other QA tasks.
"""

from typing import Any, Dict, List, Tuple

from .bases.KBBase import PassageChunks, QABase


class HotpotQATask(QABase):
    """Answer a multi-document question from the supplied passages."""

    name = "hotpotqa"
    defaultDataset = "hotpotqa"
    prefixPrompt = (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n"
    )
    queryPrompt = (
        "\n\nAnswer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nQuestion: "
    )

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        context = str(sample.get("context") or "")
        question = str(sample.get("input") or "").strip()
        if not context or not question:
            return [], ""
        chunks = PassageChunks(context, prefix=self.prefixPrompt)
        return chunks, f"{self.queryPrompt}{question}\nAnswer:"
