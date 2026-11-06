"""Document identity and task instructions remain part of the KV contract."""
import pytest

from methods.CacheRepair.qwen38.document_data import (
    compile_documents, fair_quotas, natural_parts, task_prefix,
)


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False,truncation=False,max_length=None):
        ids=list(map(ord,text))
        return ids[:max_length] if truncation else ids


def decode(ids):return ''.join(map(chr,ids))


def test_each_source_document_is_a_chunk_with_one_native_prefix():
    chunks,info=compile_documents(['Title A\nBody A.','Title B\nBody B.'],
        'hotpotqa',CharacterTokenizer())
    assert len(chunks)==2 and info['source_document_count']==2
    assert decode(chunks[0])==task_prefix('hotpotqa')+'Title A\n\nBody A.\n\n'
    assert decode(chunks[1])=='Title B\n\nBody B.\n\n'
    assert task_prefix('hotpotqa') not in decode(chunks[1])


def test_budget_retains_both_documents_and_first_instruction():
    prefix=task_prefix('musique')
    chunks,info=compile_documents(['A\n'+'x'*1000,'B\n'+'y'*1000],
        'musique',CharacterTokenizer(),max_tokens=len(prefix)+100)
    assert sum(map(len,chunks))==len(prefix)+100
    assert decode(chunks[0]).startswith(prefix+'A\n\n')
    assert decode(chunks[1]).startswith('B\n\n')
    assert all(decode(c).endswith('\n\n') for c in chunks)
    assert info['truncated_documents']==2


def test_dialogue_uses_turn_boundaries_and_preserves_all_text():
    text='Alice: Hello.\nBob: Hi.\nAlice: How are you?\nBob: Fine.'
    chunks,info=compile_documents([text],'samsum',CharacterTokenizer())
    assert info['single_document_boundary']=='line/turn' and len(chunks)==4
    assert ''.join(map(decode,chunks))==task_prefix('samsum')+text


def test_lone_report_uses_sentence_boundaries_not_token_slices():
    text='One sentence. A second sentence. The third one. The fourth sentence.'
    parts,kind=natural_parts(text)
    assert kind=='sentence' and len(parts)==4 and ''.join(parts)==text
    assert all(p.rstrip().endswith('.') for p in parts)


def test_ruler_retains_its_own_instruction_without_qa_prefix():
    text='Memorize these numbers. First fact. Second fact. Which number?'
    chunks,info=compile_documents(['[INST] '+text+' [/INST]'],
        'niah_shuffle',CharacterTokenizer())
    assert ''.join(map(decode,chunks))==text
    assert info['prefix_tokens']==0 and info['embedded_ruler_instruction']


def test_waterfill_keeps_short_documents_instead_of_losing_later_context():
    quotas=fair_quotas([10000,20,40],1000,first_reserve=50)
    assert quotas==[940,20,40]
    with pytest.raises(ValueError,match='budget'):
        fair_quotas([100,100],4,first_reserve=50)


def test_document_limit_selects_first_sources_without_merging_them():
    chunks,info=compile_documents(['A','B','C'],'multinews',CharacterTokenizer(),max_documents=2)
    assert len(chunks)==2 and info['dropped_documents']==1
    assert decode(chunks[1])=='B\n\n'


def test_unsegmentable_single_word_is_explicitly_rejected():
    with pytest.raises(ValueError,match='boundary'):
        compile_documents(['Hello'],'samsum',CharacterTokenizer())


def test_leading_and_trailing_blank_lines_do_not_create_empty_chunks():
    text='\n\nFirst section.\n\nSecond section.\n\n   '
    parts,kind=natural_parts(text)
    assert ''.join(parts)==text and len(parts)==2 and all(p.strip() for p in parts)
