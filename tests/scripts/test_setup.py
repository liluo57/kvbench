"""Tests for setup."""

import importlib
import importlib.util
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_setup_uses_selected_project_python_and_no_checkout(monkeypatch):
    spec = importlib.util.spec_from_file_location("kvbench_setup_a3", ROOT / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    setup.setup_a3(SimpleNamespace(python="/project/bin/python", rebuild=True,
        offline=True, no_index=True, find_links=["/wheels"]))
    install = commands[0]
    assert install[:3] == ["uv", "pip", "install"]
    assert install[install.index("--python") + 1] == "/project/bin/python"
    assert str(ROOT / "requirement.txt") in install
    assert str(ROOT / "methods/A3Repo/requirements.txt") in install
    assert "--offline" in install and "--no-index" in install
    assert not any("venv" in command or "git" in command for command in commands)


def test_setup_installs_into_selected_project_python_without_creating_a_venv(monkeypatch):
    root = Path(__file__).resolve().parents[2]
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


def test_setup_uses_the_selected_project_python_and_pinned_requirements(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('kvbench_setup', root / 'setup.py')
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr(setup.subprocess, 'run', run)
    monkeypatch.setattr(setup.shutil, 'which', lambda name: name)
    monkeypatch.setattr(setup, '_build_compiler', lambda *args: None)
    setup.setup_hypic(SimpleNamespace(python='/project/bin/python', rebuild=True,
        offline=True, no_index=False, find_links=[]))
    install = commands[0]
    assert install[:3] == ['uv', 'pip', 'install']
    assert install[install.index('--python') + 1] == '/project/bin/python'
    assert str(root / 'requirement.txt') in install
    assert str(root / 'methods/hypic/requirements.txt') in install
    assert '--offline' in install
    assert all('venv' not in command for command in commands)


def _load_setup():
    root = Path(__file__).resolve().parents[2]
    spec = spec_from_file_location("kvbench_setup", root / "setup.py")
    setup = module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup, root


def test_kvpacket_setup_checks_the_selected_main_python_without_installing(monkeypatch):
    setup, root = _load_setup()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    monkeypatch.setattr(setup.subprocess, "run", run)

    setup.setup_kvpacket(SimpleNamespace(python="/project/bin/python"))

    assert len(calls) == 1
    command, options = calls[0]
    assert command[:3] == ["/project/bin/python", "-I", "-B"]
    assert command[-1] == str(root)
    assert "2.13.0" in command[4]
    assert "4.57.6" in command[4]
    assert options["cwd"] == root
    assert options["capture_output"] is True
    assert not any("pip" in part or "uv" in part or "venv" in part for part in command)


def test_kvpacket_setup_reports_environment_mismatch_without_mutating_it(monkeypatch):
    setup, _ = _load_setup()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="AssertionError: transformers: expected 4.57.6, found 5.0",
        )

    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    monkeypatch.setattr(setup.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="expected 4.57.6"):
        setup.setup_kvpacket(SimpleNamespace(python="python3"))

    assert len(calls) == 1
    assert "install" not in calls[0]


def test_setup_can_check_cacheblend_then_kvpacket_in_one_invocation(monkeypatch):
    setup, _ = _load_setup()
    calls = []
    monkeypatch.setattr(setup.sys, "argv", ["setup.py", "--cacheblend", "--kvpacket"])
    monkeypatch.setattr(setup, "setup_cacheblend", lambda args: calls.append("cacheblend"))
    monkeypatch.setattr(setup, "setup_kvpacket", lambda args: calls.append("kvpacket"))

    assert setup.main() == 0
    assert calls == ["cacheblend", "kvpacket"]
