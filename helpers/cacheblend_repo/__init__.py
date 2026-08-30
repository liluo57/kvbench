"""CacheBlend (upstream repo) integration.

Modules:

- :mod:`helpers.cacheblend_repo.CacheblendRepoHelper` — JSON-lines worker
  that drives the upstream CacheBlend vLLM fork in its own virtualenv.
- :mod:`helpers.cacheblend_repo.Qwen3ForCacheBlendRepo` — out-of-tree
  Qwen3 model adapter registered via side-effect for that worker (see
  the file's docstring for the explicit registration contract).

These two modules are unrelated to the vLLM + LMCache CacheBlend path
under :mod:`helpers.backends.VllmCacheblendPatches` — they target the
separate CacheBlend repo (vLLM 0.4.1) and exist to bridge its missing
Qwen3 support.
"""