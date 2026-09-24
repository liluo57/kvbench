"""2WikiMultiHopQA task using the complete official labeled dev split.

Dataset: ``<DatasetPath>/2wikimultihopqa``
(``2wikimultihopqa_dev.json``), prepared by :mod:`scripts.PrepareDataset` from
the official authors' release. See :mod:`tasks.bases.KBBase` for the record
layout, chat-prompt wrapping and scoring.
"""

from .bases.KBBase import QABase


class TwoWikiMultiHopQATask(QABase):
    """Official 2WikiMultiHopQA dev split with CacheBlend-style prompts."""

    name = "2wikimultihopqa"
    defaultDataset = "2wikimultihopqa"
    prefixPrompt = (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n"
    )
    queryPrompt = (
        "\n\nAnswer the question based on the given passages. Answer the "
        "question within 5 words. Do NOT repeat the question or output any "
        "other words. Question: "
    )
