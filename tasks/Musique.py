"""Multi-hop QA over a knowledge base (``example/blend_musique.py``).

Dataset: ``<DatasetPath>/musique``. See :mod:`tasks.bases.KBBase` for the record
layout, chat-prompt wrapping and scoring.
"""

from .bases.KBBase import QABase


class MusiqueTask(QABase):
    """Multi-hop QA over a knowledge base (``blend_musique.py``)."""

    name = "musique"
    defaultDataset = "musique"
    prefixPrompt = (
        "You will be asked a question after reading several passages. Please "
        "directly answer the question based on the given passages. Do NOT "
        "repeat the question. The answer should be within 5 words..\nPassages:\n"
    )
    queryPrompt = (
        "\n\nAnswer the question directly based on the given passages. Do NOT "
        "repeat the question. The answer should be within 5 words. \nQuestion:"
    )
