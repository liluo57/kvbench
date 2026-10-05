"""EPIC keeps generated output as a standalone native KV chunk."""

from test_recompute_backend import bind, runtime
from methods import EPIC


def test_epic_retains_only_output_ids_with_local_positions_and_no_output_prefill(monkeypatch):
    backend, sampler = runtime()
    method = bind(EPIC(recomputeTokens=8), backend, sampler)
    method.Prepare([[[2, 3]]])
    samples = iter([14, 15])
    sampler._SampleNextToken = lambda *args: next(samples)
    prompt = [2, 3, 4, 5]
    result = method.Run([prompt], retainOutput=[True], maxNewTokens=2)[0]
    retained = method._states[0][-1]

    assert retained.ids == result.output_ids
    assert retained.ids
    assert retained.cache.seq_len == retained.cache.doc_len == len(retained.ids)
    assert retained.cache.layers[0].logical_positions.tolist() == [list(range(len(retained.ids)))]
    assert all(layer.key.shape[2] == len(result.output_ids) for layer in retained.cache.layers)

    # Matching the output IDs later must reuse the standalone fragment.
    later = [9, *retained.ids, 10]
    sampler._SampleNextToken = lambda *args: 16
    reused = method.Run([later], maxNewTokens=1)[0]
    assert reused.metadata["cached_tokens"] == len(retained.ids)


def test_epic_keeps_native_interior_special_ids_and_drops_terminal_eos():
    backend, sampler = runtime()
    samples = iter([2, 1])
    sampler._SampleNextToken = lambda *args: next(samples)
    method = bind(EPIC(), backend, sampler)
    result = method.Run([[7, 8, 9]], retainOutput=[True], maxNewTokens=4)[0]

    assert result.output_ids == [2]
    assert method._states[0][-1].ids == [2]
    assert method._states[0][-1].cache.seq_len == 1
