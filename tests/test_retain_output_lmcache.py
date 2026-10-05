from importlib import import_module

import pytest

from methods.CacheblendLmcache import CacheblendLmcache


def test_true_retention_fails_before_any_batch_request_is_submitted():
    method = object.__new__(CacheblendLmcache)
    method._states = [
        {"prepare": [[1]], "context_tokens": [1]},
        {"prepare": [[2]], "context_tokens": [2]},
    ]
    states_before = [dict(state) for state in method._states]

    with pytest.raises(NotImplementedError, match="native output KV retention"):
        method.Run([[1], [2]], retainOutput=[False, True])

    assert method._states == states_before
    assert not hasattr(method, "llm")


@pytest.mark.parametrize("retain", [None, [], [False], [False, True]])
def test_missing_false_or_out_of_range_flags_keep_existing_path(monkeypatch, retain):
    module = import_module("methods.CacheblendLmcache")
    method = object.__new__(CacheblendLmcache)
    method._states = [{"prepare": [], "context_tokens": []}]
    method.recompRatio = 0.15
    submitted = []
    monkeypatch.setattr(
        module, "ComposeReuse", lambda _prepare, _run: (None, [])
    )

    def generate(streams, _max_tokens, *, onlineStart):
        submitted.extend(streams)
        return [("answer", [], 0.1, 1, 0.2, 0)]

    monkeypatch.setattr(method, "_GenerateBatch", generate)
    results = method.Run([[7, 8]], retainOutput=retain, maxNewTokens=1)

    assert submitted == [[7, 8]]
    assert len(results) == 1
    assert results[0].output == "answer"
