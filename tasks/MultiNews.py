"""LongBench MultiNews summarization task with local data only."""

from typing import Any, Dict, List, Tuple

from core.Result import Result

from .bases.KBBase import KBBase, PassageChunks, RougeL


class MultiNewsTask(KBBase):
    """Summarize multiple news passages and score with ROUGE-L."""

    name = "multinews"
    defaultDataset = "multinews"
    prefixPrompt = (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n"
    )
    suffixPrompt = "\n\nNow, write a one-page summary of all the news.\n\nSummary:"

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        context = str(sample.get("context") or "")
        if not context:
            return [], ""
        return PassageChunks(context, prefix=self.prefixPrompt), self.suffixPrompt

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        prediction = str(result.output or "").lstrip()
        answers = metadata["answers"]
        best = max((RougeL(prediction, answer) for answer in answers), default=0.0)
        return {"rougeL": best}
