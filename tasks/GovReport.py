"""LongBench GovReport summarization task with local data only."""

from typing import Any, Dict, List, Tuple

from core.Result import Result

from .bases.KBBase import KBBase, ParagraphChunks, RougeL


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

    def __init__(
        self,
        dataset=None,
        maxSamples=-1,
        startIdx=0,
        dataDir=None,
        tag=None,
        nChunks=1,
        maxSampleLength=0,
    ):
        """Create the task, optionally splitting each report into chunks.

        ``nChunks=1`` retains the original single-chunk prompt exactly.
        Larger values split at paragraph boundaries when available, with a
        sentence-boundary fallback for the flattened GovReport snapshot.

        ``maxSampleLength`` is measured in tokens after rendering the complete
        model-specific chat prompt.  ``0`` disables filtering; a positive
        value excludes samples longer than that limit.
        """
        if not isinstance(nChunks, int) or isinstance(nChunks, bool) or nChunks < 1:
            raise ValueError("nChunks must be a positive integer")
        if (
            not isinstance(maxSampleLength, int)
            or isinstance(maxSampleLength, bool)
            or maxSampleLength < 0
        ):
            raise ValueError("maxSampleLength must be a non-negative integer")
        super().__init__(
            dataset=dataset,
            maxSamples=maxSamples,
            startIdx=startIdx,
            dataDir=dataDir,
            tag=tag,
        )
        self.nChunks = nChunks
        self.maxSampleLength = maxSampleLength

    def _ShouldSkipPrompt(self, fullPrompt: str, modelPath: str) -> bool:
        if self.maxSampleLength == 0:
            return False
        # Reuse the adapter's cached tokenizer so the count uses the same
        # tokenizer as render_user_prompt, rather than a character heuristic.
        from helpers.backends.ModelAdapter import _tokenizer

        tokenCount = len(
            _tokenizer(modelPath).encode(fullPrompt, add_special_tokens=False)
        )
        return tokenCount > self.maxSampleLength

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        context = str(sample.get("context") or "")
        if not context:
            return [], ""
        return ParagraphChunks(
            context, self.nChunks, prefix=self.prefixPrompt
        ), self.suffixPrompt

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        prediction = str(result.output or "").lstrip()
        answers = metadata["answers"]
        best = max((RougeL(prediction, answer) for answer in answers), default=0.0)
        return {"rougeL": best}
