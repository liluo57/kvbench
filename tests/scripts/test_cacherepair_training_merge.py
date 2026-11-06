"""Content deduplication, held-out protection and synthetic instance identity."""
import numpy as np

from scripts.CompileCacheRepairTrainingPool import has_overlap
from scripts.MergeCacheRepairTrainingPool import Protection, content_identity
from scripts.PrepareCacheRepairRulerTrain import signature, source_record_id
from methods.CacheRepair.qwen38.decontam import window_hashes
from methods.CacheRepair.qwen38.eval_inventory import normalize


def test_ordered_context_identity_ignores_whitespace_boundaries():
    first=[normalize('Title\nFirst paragraph.'),normalize('Second paragraph.')]
    same=[normalize('Title First paragraph.\n\nSecond paragraph.')]
    assert content_identity(first)==content_identity(same)
    assert content_identity(first)!=content_identity(list(reversed(first)))


def test_protection_distinguishes_source_title_from_heldout_content():
    p=Protection()
    held=' '.join(f'heldoutword{i}' for i in range(100))
    p.add('Same Wiki title\n'+held,'validation',spans=True)
    p.finish()
    assert p.reason(normalize('Same Wiki title\n'+held))=='heldout_exact_document'
    assert p.reason(normalize('Different title\n'+held))=='heldout_256_character_span'
    assert p.reason(normalize('Same Wiki title\nCompletely different body.')) is None


def test_cwe_signature_excludes_reordered_reuse_of_target_set():
    assert signature('cwe',{'outputs':['orange','apple']})==signature('cwe',{'outputs':['apple','orange']})


def test_niah_signature_contains_key_and_value():
    def row(key,value):
        return {'input':f'One of the special magic numbers for {key} is: {value}.','outputs':[value]}
    assert signature('niah',row('alpha','123'))!=signature('niah',row('beta','123'))
    assert signature('niah',row('alpha','123'))!=signature('niah',row('alpha','456'))


def test_repeated_native_generator_index_preserves_distinct_sampling_groups():
    first={'index':0,'input':'First generated input.'}
    second={'index':0,'input':'Different generated input.'}
    assert source_record_id('niah',4096,700001,first)!=source_record_id('niah',4096,700001,second)


def test_token_screen_requires_fifty_tokens_and_finds_last_possible_window():
    heldout=list(range(50))
    index=(window_hashes(heldout),np.array([0]),np.array([0]),[heldout])
    assert not has_overlap(heldout[:49],index)
    assert has_overlap([999,998]+heldout,index)
    assert not has_overlap([999,998]+heldout[:-1]+[1234],index)


def test_token_screen_rejects_hash_collision_without_actual_token_match():
    candidate=list(range(50))
    collision=(window_hashes(candidate),np.array([0]),np.array([0]),[list(range(100,150))])
    assert not has_overlap(candidate,collision)
