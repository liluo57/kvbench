"""CacheBlend single-hop QA over a knowledge base (``example/blend_wikimqa.py``).

Dataset: ``<DatasetPath>/wikimqa``. See :mod:`tasks._Cacheblend` for the record
layout, chat-prompt wrapping and scoring.
"""

from ._Cacheblend import _QABase


class WikimQATask(_QABase):
    """Single-hop QA (cacheblend ``blend_wikimqa.py``)."""

    name = "wikimqa"
    defaultDataset = "wikimqa"
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
