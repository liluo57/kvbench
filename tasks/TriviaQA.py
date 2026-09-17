"""LongBench TriviaQA task backed by a local data snapshot."""

from typing import Any, Dict, List, Optional, Tuple

from .bases.KBBase import PassageChunks, QABase


class TriviaQATask(QABase):
    """Answer the final trivia question using the few-shot passage context."""

    name = "triviaqa"
    defaultMaxNewTokens = 512
    defaultDataset = "triviaqa"
    prefixPrompt = (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n"
    )

    def __init__(
        self,
        dataset: Optional[str] = None,
        maxSamples: int = -1,
        startIdx: int = 0,
        dataDir: Optional[str] = None,
        tag: Optional[str] = None,
        maxNewTokens: int = 512,
    ):
        super().__init__(
            dataset=dataset,
            maxSamples=maxSamples,
            startIdx=startIdx,
            dataDir=dataDir,
            tag=tag,
            maxNewTokens=maxNewTokens,
        )

    def _Build(self, sample: Dict[str, Any]) -> Tuple[List[str], str]:
        context = str(sample.get("context") or "")
        question = str(sample.get("input") or "").strip()
        if not context or not question:
            return [], ""
        # ``input`` already contains the final Passage/Question/Answer block
        # from LongBench, so it is the fresh suffix after the cached examples.
        return PassageChunks(context, prefix=self.prefixPrompt), f"\n\n{question}"
