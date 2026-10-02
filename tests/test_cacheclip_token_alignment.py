"""Focused tests for CacheClip's ByteLevel ID-to-text view."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods.CacheClip.Selector import (
    AuxiliaryRuntime,
    map_auxiliary_to_primary_indices,
    token_text_and_offsets,
)


_TOKENIZER_PATHS = {
    "qwen3": Path(
        "/data/lyh/deepspec/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/"
        "1cfa9a7208912126459214e8b04321603b3df60c"
    ),
    "smollm2": Path(
        "/data/lyh/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/"
        "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
    ),
}


def _load_local_tokenizer(name: str):
    path = _TOKENIZER_PATHS[name]
    if not (path / "tokenizer.json").is_file():
        pytest.skip(f"local {name} tokenizer is not cached")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, use_fast=True
    )


@pytest.mark.parametrize("model_name", ["qwen3", "smollm2"])
def test_bytelevel_id_text_spans_preserve_chunked_authoritative_ids(model_name, monkeypatch):
    tokenizer = _load_local_tokenizer(model_name)
    chunks = ["中", "文", "e", "\u0301", "👩", "\u200d", "💻", "A", "<|im_start|>", "B"]
    ids = [token for chunk in chunks for token in tokenizer.encode(chunk, add_special_tokens=False)]
    original_ids = ids.copy()
    expected = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # This sequence is a per-chunk ID concatenation. A whole-document tokenizer
    # call is allowed to produce different BPE boundaries.
    assert expected == "".join(chunks)
    if model_name == "qwen3":
        assert ids != tokenizer.encode("".join(chunks), add_special_tokens=False)

    def no_retokenize(*args, **kwargs):
        pytest.fail("token_text_and_offsets must not re-tokenize primary text")

    monkeypatch.setattr(tokenizer, "encode", no_retokenize)
    text, spans = token_text_and_offsets(tokenizer, ids)

    assert ids == original_ids
    assert text == expected
    assert len(spans) == len(ids)
    assert all(0 <= start <= end <= len(text) for start, end in spans)

    special_id = next(
        index for index, token_id in enumerate(ids)
        if token_id in tokenizer.all_special_ids
    )
    special = tokenizer.convert_ids_to_tokens([ids[special_id]])[0]
    start, end = spans[special_id]
    assert text[start:end] == special


@pytest.mark.parametrize("model_name", ["qwen3", "smollm2"])
def test_bytelevel_utf8_fragments_share_their_output_character_span(model_name):
    tokenizer = _load_local_tokenizer(model_name)
    ids = tokenizer.encode("\u200d", add_special_tokens=False)
    text, spans = token_text_and_offsets(tokenizer, ids)

    assert text == "\u200d"
    assert len(ids) == 2
    assert spans[0] == spans[1] == (0, 1)


def test_bytelevel_replacement_decode_keeps_original_ids_and_special_boundaries():
    tokenizer = _load_local_tokenizer("qwen3")
    start_ids = tokenizer.encode("A", add_special_tokens=False)
    special_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    middle_ids = tokenizer.encode("B", add_special_tokens=False)
    end_ids = tokenizer.encode("C", add_special_tokens=False)
    # A truncated prefix and an isolated continuation byte remain ordinary
    # model IDs/cache rows even though the selector view replaces them.
    ids = start_ids + special_ids + middle_ids + [378] + end_ids + [235]
    original_ids = ids.copy()
    expected = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    text, spans = token_text_and_offsets(tokenizer, ids)

    assert ids == original_ids
    assert text == expected == "A<|im_start|>B�C�"
    assert len(spans) == len(ids)
    assert text[spans[len(start_ids)][0]:spans[len(start_ids)][1]] == "<|im_start|>"
    replacement_index = len(start_ids) + len(special_ids) + len(middle_ids)
    replacement_span = spans[replacement_index]
    following_span = spans[replacement_index + 1]
    assert text[replacement_span[0]:replacement_span[1]] == "�"
    assert text[following_span[0]:following_span[1]] == "C"
    assert text[spans[-1][0]:spans[-1][1]] == "�"


def test_bytelevel_nfc_view_is_canonical_and_not_original_text():
    tokenizer = _load_local_tokenizer("qwen3")
    original = "e\u0301"
    ids = tokenizer.encode(original, add_special_tokens=False)
    text, spans = token_text_and_offsets(tokenizer, ids)

    assert text == "é"
    assert text != original
    assert len(spans) == len(ids) == 1
    assert spans == [(0, 1)]


def test_bytelevel_added_tokens_follow_the_decoder_byte_map():
    from transformers import AddedToken

    tokenizer = _load_local_tokenizer("qwen3")
    cases = [
        ("ĠNEW_ADDED", False, " NEW_ADDED"),
        ("中NEW_ADDED", False, "中NEW_ADDED"),
        ("éNEW_ADDED", False, "�NEW_ADDED"),
        ("ĠSPECIAL_ADDED", True, " SPECIAL_ADDED"),
        ("éSPECIAL_ADDED", True, "�SPECIAL_ADDED"),
    ]
    for literal, special, expected in cases:
        added = AddedToken(literal, special=special, normalized=False)
        if special:
            tokenizer.add_special_tokens({"additional_special_tokens": [added]})
        else:
            tokenizer.add_tokens([added])
        ids = tokenizer.encode(literal, add_special_tokens=False)
        assert len(ids) == 1
        assert tokenizer.convert_ids_to_tokens(ids) == [literal]

        text, spans = token_text_and_offsets(tokenizer, ids)

        assert text == expected
        assert text == tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        assert spans == [(0, len(text))]


def test_bytelevel_unknown_id_piece_is_rejected():
    tokenizer = _load_local_tokenizer("qwen3")
    tokenizer.convert_ids_to_tokens = lambda ids: [None for _ in ids]

    with pytest.raises(ValueError, match="invalid tokenizer ID"):
        token_text_and_offsets(tokenizer, [99999999])


def test_auxiliary_prefix_and_document_are_encoded_separately(monkeypatch):
    import importlib

    selector = importlib.import_module("methods.CacheClip.Selector")

    class FakeTokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors,
                     return_offsets_mapping):
            assert add_special_tokens is False
            assert return_tensors == "pt"
            if text == "warm":
                ids, offsets = [10, 11], [(0, 1), (1, 4)]
            elif text == "body":
                ids, offsets = [20, 21], [(0, 2), (2, 4)]
            else:
                raise AssertionError(f"unexpected combined tokenization: {text!r}")
            result = {"input_ids": torch.tensor([ids], dtype=torch.long)}
            if return_offsets_mapping:
                result["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
            return result

    class FakeModel:
        config = SimpleNamespace()

        def __init__(self):
            self.input_ids = []

        def __call__(self, *, input_ids, use_cache):
            assert use_cache is True
            self.input_ids.append(input_ids.tolist())
            return SimpleNamespace(past_key_values=object())

    model = FakeModel()
    runtime = object.__new__(AuxiliaryRuntime)
    runtime.model = model
    runtime.tokenizer = FakeTokenizer()
    monkeypatch.setattr(selector, "_as_dynamic_cache", lambda cache, config: cache)

    caches = runtime.prefill_documents("warm", ["body"])

    assert model.input_ids == [[[10, 11, 20, 21]]]
    assert caches[0].prefix_length == 2
    assert caches[0].continuation_offsets == [(0, 2), (2, 4)]


def test_auxiliary_to_primary_mapping_keeps_overlap_semantics():
    auxiliary = [(0, 3), (2, 5), (8, 9), (0, 0)]
    primary = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 10)]

    assert map_auxiliary_to_primary_indices(auxiliary, primary, [0, 2, 3]) == [0, 1, 4]


def test_auxiliary_mapping_sweep_matches_original_overlap_definition_randomly():
    import random

    rng = random.Random(27)
    for _ in range(200):
        auxiliary = []
        primary = []
        for target in (auxiliary, primary):
            cursor = 0
            for _ in range(rng.randrange(1, 20)):
                cursor += rng.randrange(0, 4)
                width = rng.randrange(0, 5)
                target.append((cursor, cursor + width))
                cursor += width
        selected = [index for index in range(len(auxiliary)) if rng.randrange(2)]
        spans = [auxiliary[index] for index in selected if auxiliary[index][0] != auxiliary[index][1]]
        expected = [
            primary_index
            for primary_index, (start, end) in enumerate(primary)
            if start != end and any(
                end > selected_start and start < selected_end
                for selected_start, selected_end in spans
            )
        ]
        assert map_auxiliary_to_primary_indices(auxiliary, primary, selected) == expected


def test_token_alignment_rejects_non_bytelevel_decoder():
    tokenizer = SimpleNamespace(
        is_fast=True,
        backend_tokenizer=SimpleNamespace(decoder=SimpleNamespace()),
    )
    with pytest.raises(ValueError, match="supports ByteLevel decoders"):
        token_text_and_offsets(tokenizer, [1])
