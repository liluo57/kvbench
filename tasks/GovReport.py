"""LongBench GovReport summarization task with local data only."""

from typing import Any, Dict, List, Tuple

from core.Result import Result

from .bases.KBBase import KBBase, RougeL


class GovReportTask(KBBase):
    """Summarize a government report and score with ROUGE-L."""

    name = "govreport"
    defaultDataset = "govreport"
    prefixPrompt = (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n"
    )
    suffixPrompt = (
        "\n\nNow, write a one-page summary of the report.\n\nSummary:"
    )

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        context = str(sample.get("context") or "")
        if not context:
            return [], ""
        return [self.prefixPrompt + context], self.suffixPrompt

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        prediction = str(result.output or "").lstrip()
        answers = metadata["answers"]
        best = max((RougeL(prediction, answer) for answer in answers), default=0.0)
        return {"rougeL": best}
