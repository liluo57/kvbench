"""Compression extension contracts; optional third-party compressors are excluded."""

from .base import CacheCompressor, CompressionSpec, NO_COMPRESSION_FINGERPRINT

__all__ = ["CacheCompressor", "CompressionSpec", "NO_COMPRESSION_FINGERPRINT"]
