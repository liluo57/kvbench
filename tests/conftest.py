"""Shared offline fixtures and explicit opt-ins for compiled/GPU tests."""

import pytest
import torch

from core import Config
from helpers.backends import ModelAdapter


def pytest_addoption(parser):
    parser.addoption("--run-native", action="store_true", help="Run optional compiled backend tests")
    parser.addoption("--run-gpu", action="store_true", help="Run CUDA kernel tests")


def pytest_collection_modifyitems(config, items):
    for marker in ("native", "gpu"):
        if not config.getoption(f"--run-{marker}"):
            skip = pytest.mark.skip(reason=f"Enable with --run-{marker}")
            for item in items:
                if item.get_closest_marker(marker):
                    item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    # Tiny numerical models are slower with a large machine's thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolate_random_state():
    import random

    state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(state)


@pytest.fixture
def clock(monkeypatch):
    import time

    from tests.support.clock import FakeClock

    clock = FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)
    monkeypatch.setattr(time, "time", clock)
    return clock


@pytest.fixture
def context_limit(monkeypatch):
    original = Config.Get

    def set_limit(limit):
        monkeypatch.setattr(
            Config, "Get",
            lambda key, *args, **kwargs: limit if key == "MaxModelLen" else original(key, *args, **kwargs),
        )

    set_limit(5)
    return set_limit


@pytest.fixture
def modelPath(monkeypatch):
    """Use a deterministic chat-template double without downloading a model."""
    class FakeTokenizer:
        all_special_ids = [999]

        def encode(self, text, add_special_tokens=False):
            return [ord(character) for character in text]

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            tools=None,
            **kwargs,
        ):
            userContent = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is True:
                suffix += "<think>\n"
            elif kwargs.get("enable_thinking") is False:
                suffix += "<think>\n\n</think>\n\n"
            return f"<|im_start|>user\n{userContent}{suffix}"

    ModelAdapter.set_arch_for_testing(None)
    monkeypatch.setattr(ModelAdapter, "_tokenizer", lambda _path: FakeTokenizer())
    yield "synthetic-chat-model"
    ModelAdapter.set_arch_for_testing(None)


@pytest.fixture
def fakeSkillsbench(tmp_path):
    for taskId in ("alpha", "beta", "citation-check"):
        taskDir = tmp_path / "tasks" / taskId
        taskDir.mkdir(parents=True)
        (taskDir / "task.md").write_text(f"# {taskId}\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fakeTaskTokenizer(monkeypatch):
    class FakeTokenizer:
        all_special_ids = []

        def encode(self, text, add_special_tokens=False):
            return [ord(character) for character in text]

    monkeypatch.setattr(ModelAdapter, "_tokenizer", lambda _path: FakeTokenizer())
