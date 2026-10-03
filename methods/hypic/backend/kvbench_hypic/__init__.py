"""Bundled HYPIC runtime; CUDA and model imports are worker-only."""

def __getattr__(name):
    if name == "Engine":
        from .srt.entrypoints.engine import Engine
        return Engine
    raise AttributeError(name)
