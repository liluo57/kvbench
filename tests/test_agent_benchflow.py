"""Locks the new AgentBenchFlow pieces — Task discovery, Evaluate, Workload
final_result contract — without spawning a ``bench eval run`` subprocess.

End-to-end behaviour (HTTP server + Docker sandbox) is intentionally out of
scope here: the helper has its own watchdog lifecycle and is best exercised
in a real sandbox. These tests pin down the kvbench-facing surface.
"""

from pathlib import Path

import pytest

from core.Result import Result
from core.Task import Case
from helpers.BenchflowHelper import (
    ApptainerSandbox,
    _ParseToolCalls,
    _RenderChatPrompt,
)
from helpers.SkillInjector import BuildSkillsBlock, ParseSkillFrontmatter
from tasks.AgentBenchFlowTask import AgentBenchFlowTask
from workload.AgentBenchFlowWorkload import AgentBenchFlowInput, AgentBenchFlowWorkload


@pytest.fixture
def fakeSkillsbench(tmp_path):
    """A minimal SkillsBench repo layout: ``tasks/<id>/task.md`` per task."""
    for tid in ("alpha", "beta", "gamma"):
        (tmp_path / "tasks" / tid).mkdir(parents=True)
        (tmp_path / "tasks" / tid / "task.md").write_text(f"# {tid}\n")
    return tmp_path


# --------------------------------------------------------------- Task.Cases


def test_cases_default_scan(fakeSkillsbench):
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench)
    ids = [c.metadata["task_id"] for c in task.Cases()]
    assert ids == ["alpha", "beta", "gamma"]


def test_cases_explicit_task_ids(fakeSkillsbench):
    task = AgentBenchFlowTask(
        skillsbench_dir=fakeSkillsbench, task_ids=["gamma", "alpha"]
    )
    ids = [c.metadata["task_id"] for c in task.Cases()]
    assert ids == ["gamma", "alpha"]


def test_cases_exclude_task_ids(fakeSkillsbench):
    task = AgentBenchFlowTask(
        skillsbench_dir=fakeSkillsbench, exclude_task_ids=["beta"]
    )
    ids = [c.metadata["task_id"] for c in task.Cases()]
    assert ids == ["alpha", "gamma"]


def test_cases_max_samples_caps(fakeSkillsbench):
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench, max_samples=2)
    ids = [c.metadata["task_id"] for c in task.Cases()]
    assert ids == ["alpha", "beta"]


def test_cases_empty_task_ids_raises(fakeSkillsbench):
    with pytest.raises(ValueError, match="typo"):
        AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench, task_ids=[])


def test_cases_missing_tasks_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no tasks/ directory"):
        AgentBenchFlowTask(skillsbench_dir=tmp_path)


def test_cases_filtered_to_empty_raises(fakeSkillsbench):
    with pytest.raises(ValueError, match="no SkillsBench tasks to run"):
        AgentBenchFlowTask(
            skillsbench_dir=fakeSkillsbench,
            task_ids=["alpha"],
            exclude_task_ids=["alpha"],
        )


def test_case_metadata_has_required_keys(fakeSkillsbench):
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench)
    first = next(iter(task.Cases()))
    assert first.metadata["task_id"] == "alpha"
    assert first.metadata["case_id"] == 0
    assert first.metadata["skillsbench_dir"] == str(fakeSkillsbench)
    assert isinstance(first.input, AgentBenchFlowInput)


def test_sandbox_type_defaults_to_docker(fakeSkillsbench):
    """Production path: SkillsBench task Dockerfile container."""
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench)
    assert task.sandboxType == "docker"
    case = next(iter(task.Cases()))
    assert case.input.sandbox_type == "docker"


def test_sandbox_type_can_be_overridden_to_local(fakeSkillsbench):
    """Dev / smoke path: skip the Docker sandbox, run on the host."""
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench, sandbox_type="local")
    case = next(iter(task.Cases()))
    assert case.input.sandbox_type == "local"


def test_sandbox_type_can_be_overridden_to_apptainer(fakeSkillsbench):
    """Rootless path: no Docker daemon needed."""
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench, sandbox_type="apptainer")
    case = next(iter(task.Cases()))
    assert case.input.sandbox_type == "apptainer"


def test_image_override_default_is_none(fakeSkillsbench):
    """Without override, sandbox parses the Dockerfile's FROM line."""
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench, sandbox_type="apptainer")
    case = next(iter(task.Cases()))
    assert case.input.image_override is None


def test_image_override_per_task_dict(fakeSkillsbench):
    """Per-task ImageOverrides dict wins over the single ImageOverride."""
    task = AgentBenchFlowTask(
        skillsbench_dir=fakeSkillsbench,
        sandbox_type="apptainer",
        image_override="docker://default:1",
        image_overrides={"alpha": "docker://python:3.11-slim"},
    )
    cases = list(task.Cases())
    byId = {c.metadata["task_id"]: c.input.image_override for c in cases}
    assert byId["alpha"] == "docker://python:3.11-slim"
    assert byId["beta"] == "docker://default:1"
    assert byId["gamma"] == "docker://default:1"


def test_agent_command_default_is_mini_swe_agent(fakeSkillsbench):
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench)
    assert task.agentCommand == "mini-swe-agent"


def test_qwen_prompt_exposes_tools_and_preserves_tool_history():
    prompt = _RenderChatPrompt([
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Inspect the files."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"ls"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "{\"returncode\": 0}",
        },
    ])
    assert "<tools>" in prompt
    assert '"name": "bash"' in prompt
    assert "<tool_call>" in prompt
    assert "<tool_response>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen_output_becomes_openai_tool_call():
    content, calls = _ParseToolCalls(
        '<think>brief</think>\n<tool_call>\n'
        '{"name":"bash","arguments":{"command":"ls -la"}}\n'
        "</tool_call>"
    )
    assert content == "<think>brief</think>"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "bash"
    assert '"command": "ls -la"' in calls[0]["function"]["arguments"]


def test_legacy_fenced_command_is_adapted_to_tool_call():
    _content, calls = _ParseToolCalls(
        "I will inspect the task.\n```mswea_bash_command\n"
        "cat /root/task.md\n```"
    )
    assert calls[0]["function"]["name"] == "bash"
    assert "cat /root/task.md" in calls[0]["function"]["arguments"]


# ---------------------------------------------------- SkillInjector (frontmatter)


def test_parse_frontmatter_with_simple_keys():
    text = "---\nname: foo\ndescription: bar\n---\n# body\n"
    fields, body = ParseSkillFrontmatter(text)
    assert fields == {"name": "foo", "description": "bar"}
    assert body == "# body"


def test_parse_frontmatter_strips_surrounding_quotes():
    text = '---\nname: foo\ndescription: "a b c"\n---\nbody\n'
    fields, body = ParseSkillFrontmatter(text)
    assert fields["description"] == "a b c"
    assert body == "body"


def test_parse_frontmatter_skips_nested_metadata_block():
    text = (
        "---\n"
        "name: foo\n"
        "description: d\n"
        "license: MIT\n"
        "metadata:\n"
        "    skill-author: K-Dense\n"
        "    extra: nested\n"
        "---\n"
        "# body\n"
    )
    fields, body = ParseSkillFrontmatter(text)
    assert fields == {"name": "foo", "description": "d", "license": "MIT"}
    assert "skill-author" not in fields
    assert body == "# body"


def test_parse_frontmatter_missing_returns_empty_fields():
    text = "# no frontmatter\njust body\n"
    fields, body = ParseSkillFrontmatter(text)
    assert fields == {}
    assert body == "# no frontmatter\njust body\n"


def test_parse_frontmatter_unterminated_returns_body():
    text = "---\nname: foo\nstill frontmatter\n"
    fields, body = ParseSkillFrontmatter(text)
    assert fields == {}
    assert body.startswith("---")


# --------------------------------------------- SkillInjector.BuildSkillsBlock


def _MakeSkill(skillsDir: Path, name: str, body: str = "do the thing") -> Path:
    skillDir = skillsDir / name
    skillDir.mkdir(parents=True, exist_ok=True)
    (skillDir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: describes {name}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skillDir


def test_build_skills_block_empty_when_no_skills_dir(tmp_path):
    """Tasks with no environment/skills/ yield an empty block."""
    skillsbench = tmp_path / "sb"
    (skillsbench / "tasks" / "alpha").mkdir(parents=True)
    assert BuildSkillsBlock(skillsbench, "alpha") == ""


def test_build_skills_block_includes_all_skill_names(tmp_path):
    skillsbench = tmp_path / "sb"
    skillsDir = skillsbench / "tasks" / "alpha" / "environment" / "skills"
    _MakeSkill(skillsDir, "alpha-skill", "alpha body")
    _MakeSkill(skillsDir, "beta-skill", "beta body")
    _MakeSkill(skillsDir, "gamma-skill", "gamma body")

    block = BuildSkillsBlock(skillsbench, "alpha")
    assert "## alpha-skill" in block
    assert "## beta-skill" in block
    assert "## gamma-skill" in block
    assert "alpha body" in block
    assert "beta body" in block
    assert "gamma body" in block


def test_build_skills_block_strips_frontmatter_from_output(tmp_path):
    skillsbench = tmp_path / "sb"
    skillsDir = skillsbench / "tasks" / "alpha" / "environment" / "skills"
    _MakeSkill(skillsDir, "foo", "real body content")

    block = BuildSkillsBlock(skillsbench, "alpha")
    # The opening ``---`` fence from the frontmatter must not leak.
    assert "name: foo" not in block
    assert "describes foo" in block
    assert "real body content" in block


def test_build_skills_block_renders_description_before_body(tmp_path):
    skillsbench = tmp_path / "sb"
    skillsDir = skillsbench / "tasks" / "alpha" / "environment" / "skills"
    _MakeSkill(skillsDir, "foo", "body goes here")

    block = BuildSkillsBlock(skillsbench, "alpha")
    fooIdx = block.index("## foo")
    descIdx = block.index("describes foo")
    bodyIdx = block.index("body goes here")
    assert fooIdx < descIdx < bodyIdx


def test_build_skills_block_handles_quoted_description(tmp_path):
    skillsbench = tmp_path / "sb"
    skillsDir = skillsbench / "tasks" / "alpha" / "environment" / "skills"
    skillDir = skillsDir / "quoted"
    skillDir.mkdir(parents=True, exist_ok=True)
    (skillDir / "SKILL.md").write_text(
        '---\nname: quoted\ndescription: "A long description with spaces."\n---\n\nbody\n',
        encoding="utf-8",
    )
    block = BuildSkillsBlock(skillsbench, "alpha")
    assert "A long description with spaces." in block
    # The surrounding quotes must be stripped.
    assert '"A long description with spaces."' not in block


# ----------------------------------------------- _RenderChatPrompt with prefix


def test_render_prompt_with_system_prefix_appends_to_first_system():
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Inspect the files."},
    ]
    prefix = "# Skills\n\n## foo\ndoes foo\n"
    prompt = _RenderChatPrompt(messages, system_prefix=prefix)
    # The prefix appears inside the first system region, before the user's
    # original system content.
    assert prompt.index(prefix.rstrip()) < prompt.index("You are an agent.")
    # The skills block is part of the system block, not a separate region.
    sysIdx = prompt.index("system\n")
    userIdx = prompt.index("user\n")
    assert prompt[sysIdx:userIdx].count(prefix.rstrip()) == 1


def test_render_prompt_without_system_prefix_unchanged():
    """Existing behaviour when no prefix is passed must be preserved."""
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "hi"},
    ]
    prompt = _RenderChatPrompt(messages)
    assert "You are an agent." in prompt
    assert "# Skills" not in prompt


def test_render_prompt_skills_block_does_not_leak_to_user_turn():
    """A second ``system`` message falls through to the user branch."""
    prefix = "# Skills\n\n## foo\ndoes foo\n"
    messages = [
        {"role": "system", "content": "first system"},
        {"role": "system", "content": "second system"},
    ]
    prompt = _RenderChatPrompt(messages, system_prefix=prefix)
    # The prefix appears exactly once.
    assert prompt.count(prefix.rstrip()) == 1
    # The second system message becomes a user-role turn (not a second system).
    assert "second system" in prompt
    assert prompt.index("second system") > prompt.index("first system")


def test_apptainer_stages_copy_to_root_directory(tmp_path):
    taskDir = tmp_path / "tasks" / "sample"
    environmentDir = taskDir / "environment"
    environmentDir.mkdir(parents=True)
    (environmentDir / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
        "COPY scan_data.stl /root/\n"
        "COPY material_density_table.md /root\n"
    )
    (environmentDir / "scan_data.stl").write_text("mesh")
    (environmentDir / "material_density_table.md").write_text("density")

    outputDir = tmp_path / "output"
    outputDir.mkdir()
    sandbox = ApptainerSandbox.__new__(ApptainerSandbox)
    sandbox._agentWorkspace = outputDir / "agent_workspace"
    sandbox._agentWorkspace.mkdir()

    sandbox._StageEnvironmentInputs(taskDir)

    assert (sandbox._agentWorkspace / "scan_data.stl").read_text() == "mesh"
    assert (
        sandbox._agentWorkspace / "material_density_table.md"
    ).read_text() == "density"


def test_apptainer_stages_directory_copy(tmp_path):
    """Directory sources from Dockerfile COPY must be recursively staged.

    Regression: ``ada-bathroom-plan-repair`` ships ``COPY input /root/input``,
    the staging helper only copied files (not directories), so the agent saw
    an empty ``/root/input`` and invented a placeholder DXF, then gave up.
    """
    taskDir = tmp_path / "tasks" / "sample"
    environmentDir = taskDir / "environment"
    environmentDir.mkdir(parents=True)
    (environmentDir / "Dockerfile").write_text(
        "FROM python:3.13-slim\n"
        "COPY input /root/input\n"
        "COPY output_schema /root/output_schema\n"
    )
    # Populate the source directories with files that must reach the agent.
    inputDir = environmentDir / "input"
    inputDir.mkdir()
    (inputDir / "ada_bath_input.dxf").write_text("real dxf bytes")
    (inputDir / "layer_schema.json").write_text("{}")
    schemaDir = environmentDir / "output_schema"
    schemaDir.mkdir()
    (schemaDir / "schema.json").write_text("{}")

    outputDir = tmp_path / "output"
    outputDir.mkdir()
    sandbox = ApptainerSandbox.__new__(ApptainerSandbox)
    sandbox._agentWorkspace = outputDir / "agent_workspace"
    sandbox._agentWorkspace.mkdir()

    sandbox._StageEnvironmentInputs(taskDir)

    # Full directory tree must be staged under agent_workspace/input/.
    assert (sandbox._agentWorkspace / "input" / "ada_bath_input.dxf").read_text() == "real dxf bytes"
    assert (sandbox._agentWorkspace / "input" / "layer_schema.json").read_text() == "{}"
    assert (sandbox._agentWorkspace / "output_schema" / "schema.json").read_text() == "{}"


def test_apptainer_fallback_copies_directory_top_level(tmp_path):
    """Tasks without an explicit COPY line still get their top-level dirs."""
    taskDir = tmp_path / "tasks" / "sample"
    environmentDir = taskDir / "environment"
    environmentDir.mkdir(parents=True)
    (environmentDir / "Dockerfile").write_text("FROM python:3.13-slim\n")
    extras = environmentDir / "extras"
    extras.mkdir()
    (extras / "notes.md").write_text("x")

    outputDir = tmp_path / "output"
    outputDir.mkdir()
    sandbox = ApptainerSandbox.__new__(ApptainerSandbox)
    sandbox._agentWorkspace = outputDir / "agent_workspace"
    sandbox._agentWorkspace.mkdir()

    sandbox._StageEnvironmentInputs(taskDir)

    assert (sandbox._agentWorkspace / "extras" / "notes.md").read_text() == "x"


# ------------------------------------------------------------- Task.Evaluate


@pytest.mark.parametrize(
    "payload,expected",
    [
        (None, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        ({"reward": 0.5}, 0.5),
        ({"rewards": [0.0, 1.0, 1.0]}, pytest.approx(2 / 3)),
        ({"scores": {"a": 1.0, "b": 0.0}}, 0.5),
        ({"reward": "0.8"}, 0.8),
        ({"reward": True}, 1.0),
        ({"unrelated": 42}, 0.0),
    ],
)
def test_extract_reward_shapes(payload, expected):
    assert AgentBenchFlowTask._ExtractReward(payload) == expected


def test_evaluate_returns_reward_and_accuracy(fakeSkillsbench):
    task = AgentBenchFlowTask(skillsbench_dir=fakeSkillsbench)
    result = Result(output={"reward": 0.75})
    scores = task.Evaluate(result, {"task_id": "alpha"})
    assert scores == {"reward": 0.75, "accuracy": 0.75}


# ------------------------------------------------------- Workload contract


def _Workload() -> AgentBenchFlowWorkload:
    data = AgentBenchFlowInput(skillsbench_dir="/nonexistent", task_id="alpha")
    return AgentBenchFlowWorkload(case_id=0, data=data)


def test_workload_starts_without_helper():
    """Construction is lazy — no subprocess is spawned until ``next`` runs."""
    wl = _Workload()
    assert wl._helper is None
    assert wl.finished is False
    assert wl.final_result is None


def test_final_result_prefers_synthesised_over_last_inference():
    wl = _Workload()
    wl._lastResult = Result(output="ignored")
    wl._finalResult = Result(output={"reward": 0.5}, metadata={"endpoint_url": "x"})
    assert wl.final_result.output == {"reward": 0.5}


def test_final_result_falls_back_to_last_inference_when_no_synthesis():
    wl = _Workload()
    wl._lastResult = Result(output="last inference")
    assert wl.final_result.output == "last inference"


def test_final_result_is_none_before_any_observation():
    wl = _Workload()
    assert wl.final_result is None
