import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_setup_installs_into_selected_project_python_without_creating_a_venv(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("kvbench_setup", root / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    monkeypatch.setattr(setup, "_build_compiler", lambda *args: None)
    setup.setup_cacheblend(SimpleNamespace(
        python="/project/bin/python", rebuild=True,
        offline=False, no_index=False, find_links=[],
    ))

    installs = [command for command in commands if command[:3] == ["uv", "pip", "install"]]
    assert len(installs) == 2
    assert all(command[command.index("--python") + 1] == "/project/bin/python" for command in installs)
    assert str(root / "requirement.txt") in installs[0]
    assert str(root / "methods/Cacheblend/requirements.txt") in installs[0]
    assert "kvbench-cacheblend-backend" in installs[1]
    assert not any("venv" in command for command in commands)
    assert not any("vllm" in command for command in installs)


def test_main_transformers_optional_head_dim_falls_back_to_hidden_size():
    config_module = pytest.importorskip("kvbench_cacheblend.config")
    config = object.__new__(config_module.ModelConfig)
    config.hf_text_config = SimpleNamespace(
        head_dim=None, hidden_size=4096, num_attention_heads=32,
    )
    assert config.get_head_size() == 128
    config.hf_text_config.head_dim = 160
    assert config.get_head_size() == 160


def test_private_backend_coexists_with_main_vllm():
    backend = pytest.importorskip("kvbench_cacheblend")
    modern = pytest.importorskip("vllm")
    root = Path(__file__).resolve().parents[1] / "methods/Cacheblend/backend"
    assert Path(backend.__file__).resolve().is_relative_to(root)
    assert not Path(modern.__file__).resolve().is_relative_to(root)
    from methods.Cacheblend.Worker import _load_backend
    assert _load_backend() == (backend.LLM, backend.SamplingParams)
