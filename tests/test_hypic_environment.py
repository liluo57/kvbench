import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def test_constructors_do_not_import_or_initialize_gpu_backend():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-B', '-c', '''
import sys
from methods import HypicMethod, LinearKVMethod
HypicMethod()
LinearKVMethod()
assert not any(name.startswith('methods.hypic.backend.') for name in sys.modules)
import torch
assert not torch.cuda.is_initialized()
'''], cwd=root, check=True, capture_output=True, text=True)


def test_setup_uses_the_selected_project_python_and_pinned_requirements(monkeypatch):
    root = Path(__file__).resolve().parents[1]
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
