from methods.CacheblendLmcache import CacheblendLmcache


def test_lmcache_reuse_maps_wrapped_cached_prefix_to_canonical_input(monkeypatch):
    method = object.__new__(CacheblendLmcache)
    method.sep = [98]
    monkeypatch.setattr(method, "_SaltTokens", lambda _prepare: [90, 91])

    stream, canonical = method._SaltedContextTokensWithCanonicalMask(
        [[10, 11], [20]], [[1], [2]]
    )

    assert stream == [90, 91, 10, 11, 98, 90, 91, 20]
    assert canonical == [False, False, True, True, False, False, False, True]
    assert method._CanonicalCachedPrefixTokens(2, canonical) == 0
    assert method._CanonicalCachedPrefixTokens(3, canonical) == 1
    assert method._CanonicalCachedPrefixTokens(6, canonical) == 2
    assert method._CanonicalCachedPrefixTokens(8, canonical) == 3


def test_lmcache_reuse_clamps_scheduler_count_to_stream_length():
    assert CacheblendLmcache._CanonicalCachedPrefixTokens(
        100, [True, False, True]
    ) == 2


def test_lmcache_run_excludes_namespace_wrappers_from_input_and_hit_ratio(monkeypatch):
    from importlib import import_module

    module = import_module("methods.CacheblendLmcache")

    method = object.__new__(CacheblendLmcache)
    method.sep = [98]
    method.recompRatio = 0.15
    method._states = [{"prepare": [[1, 2], [3, 4]], "context_tokens": [1]}]
    monkeypatch.setattr(method, "_SaltTokens", lambda _prepare: [90])
    monkeypatch.setattr(
        module, "ComposeReuse", lambda _prepare, _run: ([[1, 2], [3, 4]], [])
    )
    submitted = []

    def generate(streams, _max_tokens, *, onlineStart):
        submitted.extend(streams)
        return [("answer", [], 0.1, 1, 0.2, 4)]

    monkeypatch.setattr(method, "_GenerateBatch", generate)

    result = method.Run([[1, 2, 3, 4]], maxNewTokens=1)[0]

    assert submitted == [[90, 1, 2, 98, 90, 3, 4]]
    assert result.metadata["n_input"] == 4
    assert result.metadata["cache_hit_tokens"] == 2
    assert result.metadata["reuse_ratio"] == 0.5
