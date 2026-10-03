"""Exercise the real request protocol when the optional backend is installed."""
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def native():
    from methods.hypic.Build import _build_directory
    if not (_build_directory() / 'kvbench_hypic_ops.so').is_file():
        pytest.skip('Run setup.py --hypic to test the optional native backend')
    from methods.hypic.backend.kvbench_hypic.srt.managers.io_struct import GenerateReqInput
    from methods.hypic.backend.kvbench_hypic.srt.managers.tokenizer_manager import TokenizerManager
    from methods.hypic.backend.kvbench_hypic.srt.sampling.sampling_params import SamplingParams
    return GenerateReqInput, TokenizerManager, SamplingParams


def test_normalization_and_scheduler_message_preserve_ids_and_ranges(native):
    GenerateReqInput, TokenizerManager, SamplingParams = native
    ids = [151644, 3, 151645, 4]
    ranges = [(0, 1), (1, 3), (3, 4)]
    request = GenerateReqInput(input_ids=ids, pic_segments=ranges,
                               sampling_params={'max_new_tokens': 4})
    request.normalize_batch_and_arguments()
    manager = object.__new__(TokenizerManager)
    manager.preferred_sampling_params = None
    manager.sampling_params_class = SamplingParams
    manager.tokenizer = SimpleNamespace(encode=lambda *a, **k: pytest.fail('IDs were re-tokenized'))
    manager.model_config = SimpleNamespace(vocab_size=200000)
    manager.server_args = SimpleNamespace(disaggregation_transfer_backend='mooncake')
    manager.rid_to_state = {request.rid: SimpleNamespace(time_stats=SimpleNamespace(
        set_tokenize_finish_time=lambda: None))}
    message = manager._create_tokenized_object(request, None, request.input_ids)
    assert list(message.input_ids) == ids
    assert message.pic_segments == ranges


@pytest.mark.parametrize('kwargs', [
    {'input_ids': [[1, 2]], 'pic_segments': [(0, 2)]},
    {'input_ids': [1, 2], 'pic_segments': [(0, 2)], 'sampling_params': {'n': 2}},
])
def test_pic_rejects_batch_or_parallel_sample_normalization(native, kwargs):
    with pytest.raises(ValueError):
        native[0](**kwargs).normalize_batch_and_arguments()


def test_rope_theta_can_remain_in_the_transformers_v4_config(native):
    from methods.hypic.backend.kvbench_hypic.srt.utils.hf_transformers.common import get_rope_config
    config = SimpleNamespace(rope_parameters={'rope_type': 'default'}, rope_theta=1000000)
    assert get_rope_config(config) == (1000000, {'rope_type': 'default'})
