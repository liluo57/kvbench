from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_setup():
    root = Path(__file__).resolve().parents[1]
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
