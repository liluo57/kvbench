"""Source text reconstruction and nondegenerate raw-input KV chunking."""
import pytest

from scripts.PrepareCacheRepairTaskValidation import chunks_for, raw_text


@pytest.mark.parametrize('task,context', [
    ('2wikimultihopqa', [['Page', ['First sentence.', 'Second sentence.']]]),
    ('hotpotqa', {'title': ['Page'], 'sentences': [['First sentence.', 'Second sentence.']]}),
])
def test_sentence_arrays_preserve_word_boundaries(task, context):
    row = {'context': context, 'question': 'Do not include this question.', 'answer': 'Do not include this answer.'}
    assert raw_text(task, row) == 'Page\nFirst sentence. Second sentence.'


@pytest.mark.parametrize('length', [2, 3, 29, 511, 512, 513, 1023, 4096, 16384])
def test_raw_input_chunking_preserves_tokens_and_causal_split(length):
    ids = list(range(length))
    chunks = chunks_for(ids)
    assert [token for chunk in chunks for token in chunk] == ids
    assert 2 <= len(chunks) <= 32
    assert min(map(len, chunks)) > 0 and max(map(len, chunks)) <= 512


def test_input_too_short_for_nontrivial_residual_is_rejected():
    with pytest.raises(ValueError):
        chunks_for([1])


def test_dialogue_input_does_not_include_reference_summary():
    assert raw_text('samsum', {'dialogue': 'A: Hello\nB: Hi', 'summary': 'Reference summary.'}) == 'A: Hello\nB: Hi'
