"""Dependency-isolated runtime definitions for HYPIC CPU regressions.

These helpers exercise real orchestration code, not CUDA kernels/model loading.
"""
import ast
import copy
import dataclasses
import functools
from pathlib import Path

import torch


@functools.lru_cache(None)
def runtime_definition(relative_path, class_name, members=(), include_fields=False):
    """Execute real runtime definitions while isolating their CUDA-only imports.

    This exercises dataclass replacement, eager dispatch and state-copy code;
    it is not a replacement for testing the actual CUDA kernels/model loader.
    """
    path = Path(__file__).resolve().parents[2] / 'methods/hypic/backend/kvbench_hypic/srt' / relative_path
    tree = ast.parse(path.read_text())
    definition = copy.deepcopy(next(node for node in tree.body
                                    if isinstance(node, ast.ClassDef) and node.name == class_name))
    definition.bases = []
    definition.body = [node for node in definition.body
                       if (include_fields and isinstance(node, ast.AnnAssign))
                       or (isinstance(node, ast.FunctionDef) and node.name in members)]
    if not include_fields:
        definition.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                             names=[ast.alias(name='annotations')], level=0), definition], type_ignores=[])
    namespace = dict(__name__=__name__, dataclass=dataclasses.dataclass,
                     field=dataclasses.field, torch=torch)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[class_name]
