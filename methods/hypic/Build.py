"""Build/load the retained upstream ops against the active PyTorch ABI."""
from pathlib import Path
import json
import os
import re
import subprocess


def _build_directory():
    import torch
    return Path(__file__).resolve().parents[2] / '.cache' / 'hypic' / (
        'torch-' + torch.__version__.replace('+', '-'))


def load_kernels():
    import torch
    library = _build_directory() / 'kvbench_hypic_ops.so'
    if not library.is_file():
        raise RuntimeError('HYPIC CUDA ops are not built for this PyTorch; run ./setup.py --hypic')
    # Upstream also has shape-specialized JIT kernels. Reuse setup's compiler
    # choices in workers, while respecting explicit environment overrides.
    settings = library.with_suffix('.json')
    if settings.is_file():
        for key, value in json.loads(settings.read_text()).items():
            os.environ.setdefault(key, value)
    torch.ops.load_library(str(library))


def build_kernels():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load
    import flashinfer
    if CUDA_HOME is None:
        raise RuntimeError('Set CUDA_HOME to a CUDA 12.9+ development toolkit')
    version = subprocess.check_output([str(Path(CUDA_HOME) / 'bin/nvcc'), '-V'], text=True)
    match = re.search(r'release (\d+)\.(\d+)', version)
    if match is None or tuple(map(int, match.groups())) < (12, 9):
        raise RuntimeError('HYPIC with project PyTorch requires CUDA 12.9+; set CUDA_HOME to that toolkit')
    source = Path(__file__).resolve().parent / 'backend' / 'kvbench_hypic_kernel'
    build = _build_directory()
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '2')
    files = [source / 'csrc/extension.cpp', *sorted((source / 'csrc').rglob('*.cu')),
             source / 'csrc/memory/weak_ref_tensor.cpp']
    # CUDA library headers may come from the same wheels as PyTorch.
    nvidia = Path(torch.__file__).resolve().parent.parent / 'nvidia'
    includes = [str(nvidia / name / 'include') for name in
                ('cublas', 'cusparse', 'cusolver', 'curand', 'cufft', 'nvjitlink')
                if (nvidia / name / 'include').is_dir()]
    driver_paths = [Path(CUDA_HOME) / 'lib64/stubs', Path(CUDA_HOME) / 'lib/stubs',
                    *Path(CUDA_HOME).glob('targets/*/lib/stubs')]
    module = load(name='kvbench_hypic_ops', sources=[str(p) for p in files],
                extra_include_paths=[str(source / 'include'),
                    str(Path(flashinfer.__file__).parent / 'data/include'), *includes],
                extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '--expt-relaxed-constexpr',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-DFLASHINFER_ENABLE_F16', '-DFLASHINFER_ENABLE_BF16'],
                extra_ldflags=[*(f'-L{path}' for path in driver_paths if path.is_dir()), '-lcuda'],
                build_directory=str(build), verbose=True)
    settings = {key: os.environ[key] for key in ('CUDA_HOME', 'CC', 'CXX', 'CUDAHOSTCXX')
                if key in os.environ}
    # A bootstrapped compiler's JIT output may require a newer libstdc++ than
    # the host provides. Scheduler spawn must load that library at process start.
    compiler = os.environ.get('CXX', 'c++')
    stdlib = subprocess.check_output([compiler, '-print-file-name=libstdc++.so.6'], text=True).strip()
    if Path(stdlib).is_absolute() and Path(stdlib).is_file():
        settings['LD_PRELOAD'] = str(Path(stdlib).resolve())
    (build / 'kvbench_hypic_ops.json').write_text(json.dumps(settings))
    return module
