"""Tiny random-weight CPU models; no checkpoint or network access."""

import torch
import transformers

from helpers.backends.CacheBackend.HuggingFace import HFBackend
from helpers.backends.TransformersHelper import TransformersGenerator
from methods import FullRecomputeTransformer


class Tokenizer:
    eos_token_id = 1
    all_special_ids = [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)

    def __call__(self, *args, **kwargs):
        raise AssertionError("native token IDs must not be retokenized")


def runtime(family="qwen3", *, head_dim=8):
    prefix = {"llama": "Llama", "mistral": "Mistral", "qwen2": "Qwen2", "qwen3": "Qwen3"}[family]
    config = getattr(transformers, prefix + "Config")(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, max_position_embeddings=256, sliding_window=None,
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = getattr(transformers, prefix + "ForCausalLM")(config).eval()
    backend = HFBackend(model)
    sampler = TransformersGenerator.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.model = model
    sampler.tokenizer = Tokenizer()
    sampler.device = "cpu"
    sampler.prefillChunk = 64
    sampler.samplingConfig = {"mode": "greedy"}
    sampler.eosIds = {1}
    return backend, sampler


def bind(method, backend, sampler):
    method._backend, method._gen = backend, sampler
    return method


def assert_reordered_workflow_matches_full_recompute(cls, kwargs):
    backend, sampler = runtime()
    first, second = [2, 3, 4, 5], [6, 7, 8]
    method = bind(cls(**kwargs), backend, sampler)
    method.Prepare([[first, second]])
    saved = [[(layer.key.clone(), layer.value.clone()) for layer in item.cache.layers]
             for item in method._states[0]]
    baseline = bind(FullRecomputeTransformer(), backend, sampler)
    for prompt in [first + second + [9], [10] + second + [11] + first + second + [12],
                   second + first, [21, 22, 23]]:
        actual = method.Run([prompt], maxNewTokens=3)[0]
        expected = baseline.Run([prompt], maxNewTokens=3)[0]
        assert actual.output_ids == expected.output_ids
        assert actual.output == expected.output
        assert actual.metadata["n_input"] == len(prompt)
        assert actual.performance["num_output_tokens"] <= 3
        assert 0 < actual.performance["ttft"] <= actual.performance["total_time"]
    for chunk, layers in zip(method._states[0], saved):
        for layer, (key, value) in zip(chunk.cache.layers, layers):
            torch.testing.assert_close(layer.key, key, rtol=0, atol=0)
            torch.testing.assert_close(layer.value, value, rtol=0, atol=0)
