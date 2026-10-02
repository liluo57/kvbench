#!/usr/bin/env python3
"""Install KVBench and optional methods into the current project Python environment."""

import argparse
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
BUILD_CACHE = ROOT / ".cache" / "cacheblend"

_CHECK_CACHEBLEND = """
import importlib.metadata as metadata
from pathlib import Path
from packaging.requirements import Requirement
import sys

assert (3, 12) <= sys.version_info[:2] < (3, 14), "Use project Python 3.12 or 3.13"
backend = Path(sys.argv[1]).resolve()
def check_requirements(path):
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            check_requirements(path.parent / line[3:].strip())
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        actual = metadata.version(req.name)
        assert actual in req.specifier, f"{req.name}: expected {req.specifier}, found {actual}"
for path in sys.argv[2:]:
    check_requirements(Path(path))

import torch
import os
nccl = Path(torch.__file__).resolve().parent.parent / "nvidia/nccl/lib/libnccl.so.2"
if nccl.is_file():
    os.environ.setdefault("VLLM_NCCL_SO_PATH", str(nccl))
import kvbench_cacheblend
import kvbench_cacheblend._C
import kvbench_cacheblend._moe_C
from xformers import ops
for module in (kvbench_cacheblend, kvbench_cacheblend._C, kvbench_cacheblend._moe_C):
    assert Path(module.__file__).resolve().is_relative_to(backend), (
        f"Unexpected backend: {module.__file__}"
    )
print(kvbench_cacheblend.__file__)
"""

_CHECK_KVPACKET = """
from pathlib import Path
import importlib.metadata as metadata
import sys

root = Path(sys.argv[1]).resolve()
assert (3, 12) <= sys.version_info[:2] < (3, 14), (
    "Use project Python 3.12 or 3.13"
)
sys.path.insert(0, str(root))

expected = {
    "torch": "2.13.0",
    "transformers": "4.57.6",
    "PyYAML": "6.0.3",
}
for name, version in expected.items():
    actual = metadata.version(name)
    if name == "torch":
        from packaging.version import Version
        assert Version(actual).release == (2, 13, 0), (
            f"torch: expected 2.13.0 (any CUDA build), found {actual}"
        )
    else:
        assert actual == version, f"{name}: expected {version}, found {actual}"

import importlib
method = importlib.import_module("methods.KVPacket")
backend = importlib.import_module("methods.KVPacket.backend.kvpacket")
assert Path(method.__file__).resolve().is_relative_to(root / "methods" / "KVPacket"), (
    f"Unexpected KVPacket method: {method.__file__}"
)
assert Path(backend.__file__).resolve().is_relative_to(root / "methods" / "KVPacket"), (
    f"Unexpected KVPacket backend: {backend.__file__}"
)
print(backend.__file__)
"""


def _compiler_major(command):
    try:
        result = subprocess.check_output(
            [command, "-dumpfullversion", "-dumpversion"], text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(result.split(".", 1)[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0


def _build_compiler(env, offline):
    """Use supplied compilers, or bootstrap GCC 12 on older Linux hosts."""
    if env.get("CXX") or _compiler_major(shutil.which("g++") or "g++") >= 11:
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Set CC, CXX and CUDAHOSTCXX to a CUDA-compatible C++20 compiler.")
    prefix = BUILD_CACHE / "gcc12"
    cxx = prefix / "bin" / "x86_64-conda-linux-gnu-g++"
    if not cxx.is_file():
        if offline:
            raise RuntimeError("GCC 12 is not cached; set CC/CXX/CUDAHOSTCXX or run setup online once.")
        print("Installing GCC 12 build tools (no additional Python environment)...", flush=True)
        BUILD_CACHE.mkdir(parents=True, exist_ok=True)
        micromamba = BUILD_CACHE / "micromamba"
        if not micromamba.is_file():
            url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
            with urllib.request.urlopen(url, timeout=120) as response:
                archive = response.read()
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:bz2") as package:
                with package.extractfile("bin/micromamba") as executable:
                    micromamba.write_bytes(executable.read())
            micromamba.chmod(0o755)
        compiler_env = dict(env, MAMBA_ROOT_PREFIX=str(BUILD_CACHE / "mamba"))
        subprocess.run(
            [str(micromamba), "create", "-y", "--no-rc", "-p", str(prefix),
             "-c", "conda-forge", "gcc_linux-64=12", "gxx_linux-64=12"],
            env=compiler_env, check=True,
        )
    env["CC"] = str(prefix / "bin" / "x86_64-conda-linux-gnu-gcc")
    env["CXX"] = str(cxx)
    env.setdefault("CUDAHOSTCXX", str(cxx))


def setup_cacheblend(args):
    directory = ROOT / "methods" / "Cacheblend"
    backend = directory / "backend"
    requirements = directory / "requirements.txt"
    python = shutil.which(args.python) or str(Path(args.python).expanduser())
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_PRELOAD", None)
    env.setdefault("UV_CACHE_DIR", str(ROOT / ".cache" / "uv"))
    env.setdefault("MAX_JOBS", "4")
    env.setdefault("NVCC_THREADS", "1")
    env["VLLM_TARGET_DEVICE"] = "cuda"

    def run(command):
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    def check():
        return subprocess.run(
            [python, "-I", "-B", "-c", _CHECK_CACHEBLEND,
             str(backend), str(ROOT / "requirement.txt"), str(requirements)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )

    run([python, "-I", "-c",
         'import sys; assert (3, 12) <= sys.version_info[:2] < (3, 14), '
         '"Use project Python 3.12 or 3.13"'])
    if not args.rebuild and check().returncode == 0:
        print(f"CacheBlend is ready in the project environment: {python}", flush=True)
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv before setting up CacheBlend.")
    options = []
    if args.offline:
        options.append("--offline")
    if args.no_index:
        options.append("--no-index")
    for location in args.find_links:
        if "://" not in location:
            location = str(Path(location).expanduser().resolve())
        options.extend(["--find-links", location])
    print(f"Installing project and CacheBlend dependencies into {python}...", flush=True)
    run([uv, "pip", "install", "--python", python,
         "-r", str(ROOT / "requirement.txt"), "-r", str(requirements), *options])
    _build_compiler(env, args.offline)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    print("Building the bundled CacheBlend CUDA extensions...", flush=True)
    run([uv, "pip", "install", "--python", python, "--no-build-isolation",
         "--no-deps", "--reinstall-package", "kvbench-cacheblend-backend",
         "--editable", str(backend), *options])
    result = check()
    if result.returncode:
        raise RuntimeError(f"CacheBlend installation check failed:\n{result.stderr.strip()}")
    print(f"CacheBlend is ready in the project environment: {python}", flush=True)


def setup_kvpacket(args):
    """Check the vendored backend in the selected main Python environment.

    KVPacket adds no Python dependencies or compiled extensions. Its runtime
    shares the pinned project PyTorch, Transformers, and PyYAML packages, so
    this command is deliberately read-only and can be repeated offline.
    """
    python = shutil.which(args.python) or str(Path(args.python).expanduser())
    result = subprocess.run(
        [python, "-I", "-B", "-c", _CHECK_KVPACKET, str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "KVPacket setup check failed; use the project Python 3.12 or 3.13 environment "
            f"with torch 2.13.0, transformers 4.57.6, and PyYAML 6.0.3.\n{detail}"
        )
    print(f"KVPacket is ready in the project environment: {python}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cacheblend", action="store_true", help="install the bundled CacheBlend backend")
    parser.add_argument("--kvpacket", action="store_true", help="check the bundled KVPacket backend in the main environment")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if the installation is ready")
    parser.add_argument("--python", default=sys.executable,
                        help="existing project Python interpreter (defaults to this script's interpreter)")
    parser.add_argument("--offline", action="store_true", help="use locally available packages only")
    parser.add_argument("--no-index", action="store_true", help="do not query Python package indexes")
    parser.add_argument("--find-links", action="append", default=[], metavar="LOCATION",
                        help="additional directory or URL containing dependency wheels")
    args = parser.parse_args()
    if not args.cacheblend and not args.kvpacket:
        parser.error("select a method, e.g. --cacheblend or --kvpacket")
    try:
        if args.cacheblend:
            setup_cacheblend(args)
        if args.kvpacket:
            setup_kvpacket(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
