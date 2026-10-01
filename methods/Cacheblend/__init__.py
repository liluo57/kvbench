"""CacheBlend with a bundled backend sharing the main project environment.

Backend source: https://github.com/YaoJiayi/CacheBlend,
commit 55ad02675939f783a38d579393527d218a7fd581 (vllm_blend).
The bundled subset serves Llama/Mistral and dense Qwen3. KVBench's sparse
causal masks, retained KV capture and Qwen3 support are integrated directly.
Upstream Apache-2.0 and embedded native-code licenses remain in backend/.
"""

from .Method import CacheblendRepo, NaiveCacheblendRepo

__all__ = ["CacheblendRepo", "NaiveCacheblendRepo"]
