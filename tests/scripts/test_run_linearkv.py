"""Offline checks for the public LinearKV runner's execution boundary."""
import importlib
import json
import sys

import pytest
import torch

from scripts import RunLinearKV as runner


def test_dry_run_never_loads_tasks_or_initializes_the_backend(monkeypatch, capsys):
    before = set(sys.modules)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not start inference")

    from methods import LinearKVMethod

    monkeypatch.setattr(runner, "build_task", forbidden)
    monkeypatch.setattr(LinearKVMethod, "Initialize", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    assert runner.main(["--dry-run", "--ratios", "0.1", "0.2", "--gpu-nums", "2"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["methods"]) == 8
    assert {row["selector"] for row in plan["methods"]} == {"epic", "cacheblend", "prophetkv"}
    assert all(row["gpu_nums"] == 2 for row in plan["methods"])
    assert not any(name.startswith("methods.hypic.backend.") for name in set(sys.modules) - before
                   if name.endswith("entrypoints.engine"))


@pytest.mark.parametrize("status,failures,expected", [
    ("completed", [], 0),
    ("completed", [{"error": "one configuration failed"}], 1),
    ("cancelled", [], 1),
])
def test_run_preserves_task_and_reports_failures(monkeypatch, status, failures, expected):
    engine_module = importlib.import_module("core.engine")
    task = object()
    observed = {}

    class FakeEngine:
        def Evaluate(self, **kwargs):
            observed.update(kwargs)
            return dict(status=status, failures=failures, cores=[], output_dir="outputs/example")

    monkeypatch.setattr(engine_module, "Engine", FakeEngine)
    monkeypatch.setattr(runner, "build_task", lambda args: task)
    assert runner.main(["--selectors", "prophetkv", "--ratios", "0.2"]) == expected
    assert observed["tasks"] == [task]
    assert [method.Label for method in observed["methods"]] == [
        "linearkv(prophetkv,r=0.2)", "linearkv(naive)", "linearkv(full_prefill)",
    ]
