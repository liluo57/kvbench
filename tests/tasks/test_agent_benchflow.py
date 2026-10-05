"""Tests for agent benchflow."""

import importlib

import pytest

from core.Result import Result
from tasks.AgentBenchFlowTask import AgentBenchFlowTask

pytestmark = pytest.mark.usefixtures("fakeTaskTokenizer")

agentBenchFlowTaskModule = importlib.import_module("tasks.AgentBenchFlowTask")


def test_task_loads_benchflow_configuration_and_keeps_task_tag(
    monkeypatch, fakeSkillsbench
):
    config = {
        "SourceMode": "local",
        "SkillsBenchRepo": str(fakeSkillsbench),
        "Agent": "opencode",
        "Sandbox": "remote-docker",
        "RemoteDocker": {
            "Endpoint": "http://127.0.0.1:9000",
            "UploadTimeoutSec": 123,
        },
        "SkillMode": "WithoutSkill",
        "ProviderHost": "host.docker.internal",
        "RetryAttempts": 2,
    }
    monkeypatch.setattr(
        agentBenchFlowTaskModule,
        "Get",
        lambda key, default=None: config if key == "AgentBenchFlow" else default,
    )
    task = AgentBenchFlowTask("citation-check", firstRunOnly=True)
    case = next(iter(task.Cases()))
    assert case.metadata["task_id"] == "citation-check"
    assert task.Label == "agent_benchflow(citation-check)"
    assert case.input.agent == "opencode"
    assert case.input.skill_mode == "no-skill"
    assert case.input.source_mode == "local"
    assert case.input.provider_host == "host.docker.internal"
    assert case.input.retry_attempts == 2
    assert case.input.remote_upload_timeout == pytest.approx(123)
    assert case.input.first_run_only is True


def test_task_selects_remote_runtime_without_local_docker_validation(
    monkeypatch, fakeSkillsbench
):
    config = {
        "SourceMode": "local",
        "SkillsBenchRepo": str(fakeSkillsbench),
        "Sandbox": "remote-docker",
        "RemoteDocker": {"Endpoint": "http://127.0.0.1:9000"},
        "EndpointPortRange": [8000, 8031],
    }
    monkeypatch.setattr(
        agentBenchFlowTaskModule,
        "Get",
        lambda key, default=None: config if key == "AgentBenchFlow" else default,
    )
    task = AgentBenchFlowTask("citation-check")
    case = next(iter(task.Cases()))
    assert case.input.sandbox == "remote-docker"
    assert case.input.remote_endpoint == "http://127.0.0.1:9000"
    assert case.input.remote_advertise_host is None
    assert case.input.endpoint_port_range == (8000, 8031)
    assert case.input.remote_poll_interval == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"rewards": {"reward": 1.0}}, 1.0),
        ({"rewards": {"criteria_a": 1.0, "criteria_b": 0.0}}, 0.5),
        ({"rewards": [0.0, 1.0]}, 0.5),
        ({"final_metrics": {"pass_rate": 0.25}}, 0.25),
        ({"error": "verifier failed", "rewards": None}, 0.0),
    ],
)
def test_task_extracts_official_rewards(payload, expected):
    assert AgentBenchFlowTask._ExtractReward(payload) == pytest.approx(expected)


def test_task_evaluate_keeps_infrastructure_diagnostics_available():
    task = AgentBenchFlowTask("citation-check")
    result = Result(
        output={"rewards": {"reward": 0.0}, "error": "Docker environment failed"},
        metadata={"benchflow_error": "Docker environment failed"},
    )
    assert task.Evaluate(result, {}) == {"reward": 0.0, "accuracy": 0.0}
    assert result.metadata["benchflow_error"] == "Docker environment failed"


def test_task_evaluate_surfaces_first_run_task_scores_when_present():
    task = AgentBenchFlowTask("citation-check")
    result = Result(
        output={"rewards": {"reward": 1.0}},
        metadata={
            "first_run_ttft": 0.38,
            "first_run_reuse_ratio": 0.83,
            "first_run_prompt_length": 12000,
        },
    )
    assert task.Evaluate(result, {}) == {
        "reward": 1.0,
        "accuracy": 1.0,
        "first_run_ttft": pytest.approx(0.38),
        "first_run_reuse_ratio": pytest.approx(0.83),
        "first_run_prompt_length": pytest.approx(12000.0),
    }


def test_task_evaluate_omits_first_run_scores_when_unavailable():
    task = AgentBenchFlowTask("citation-check")
    result = Result(
        output={"rewards": {"reward": 0.0}, "error": "no provider reachable"},
        metadata={"benchflow_error": "no provider reachable"},
    )
    scores = task.Evaluate(result, {})
    assert "first_run_ttft" not in scores
    assert "first_run_reuse_ratio" not in scores
    assert "first_run_prompt_length" not in scores

