"""Single-hop QA over a knowledge base (``example/blend_wikimqa.py``).

Dataset: ``<DatasetPath>/wikimqa``. See :mod:`tasks.bases.KBBase` for the record
layout, chat-prompt wrapping and scoring.
"""

from .bases.KBBase import QABase


class WikimQATask(QABase):
    """Single-hop QA over a knowledge base (``blend_wikimqa.py``)."""

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
