"""Task-side helpers for compiling independently tokenized prompt pieces."""

from typing import Iterable, List, Optional, Sequence, Tuple

from . import ModelAdapter
from .Prompt import ComposeInterleavedReuseText


def EncodeText(text: str, modelPath: str) -> List[int]:
    """Encode one text piece with special tokens disabled."""
    tokenizer = ModelAdapter._tokenizer(modelPath)
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def EncodeChunks(texts: Sequence[str], modelPath: str) -> List[List[int]]:
    """Encode each reusable text chunk independently."""
    return [EncodeText(text, modelPath) for text in texts]


def OutputFragmentIds(
    rawTokenIds: Sequence[int], terminalStopIds: Sequence[int] = ()
) -> List[int]:
    """Strip trailing stop IDs while preserving all interior generated IDs."""
    ids = [int(token) for token in rawTokenIds]
    stops = {int(token) for token in terminalStopIds}
    end = len(ids)
    while end and ids[end - 1] in stops:
        end -= 1
    return ids[:end]


def TerminalStopTokenIds(
    modelPath: str,
    samplingConfig: Optional[dict] = None,
    explicitStopIds: Iterable[int] = (),
) -> List[int]:
    """Resolve model EOS, sampling EOS overrides, and explicit stop IDs."""
    return TerminalStopIdsFromTokenizer(
        ModelAdapter._tokenizer(modelPath), samplingConfig, explicitStopIds
    )


def TerminalStopIdsFromTokenizer(
    tokenizer,
    samplingConfig: Optional[dict] = None,
    explicitStopIds: Iterable[int] = (),
) -> List[int]:
    """Resolve stop IDs from an already-loaded tokenizer."""
    stopIds = set()

    def add(values) -> None:
        if values is None:
            return
        if isinstance(values, int) and not isinstance(values, bool):
            stopIds.add(int(values))
        else:
            stopIds.update(int(value) for value in values)

    override = samplingConfig.get("eos_token_id") if samplingConfig is not None else None
    add(getattr(tokenizer, "eos_token_id", None) if override is None else override)
    add(explicitStopIds)
    return sorted(stopIds)


def EncodePrompt(
    prepare_texts: Sequence[str],
    full_text: str,
    modelPath: str,
) -> Tuple[List[List[int]], List[int]]:
    """Encode reusable chunks and fresh gaps, then concatenate those IDs.

    Text matching is only used to locate each independently encoded chunk in
    the rendered prompt. The token IDs passed to Methods are the chunk IDs and
    independently encoded gap IDs concatenated in rendered order.
    """
    prepareTexts = list(prepare_texts)
    prepareIds = EncodeChunks(prepareTexts, modelPath)
    runPieces: List[List[int]] = []
    for prepareIndex, text in ComposeInterleavedReuseText(prepareTexts, full_text):
        runPieces.append(
            EncodeText(text, modelPath)
            if prepareIndex is None
            else prepareIds[prepareIndex]
        )
    return prepareIds, [token for piece in runPieces for token in piece]
