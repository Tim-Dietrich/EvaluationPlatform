import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import ConfigurationError
from evaluation_platform.self_collaboration_agent import (
    HYPERPARAMETERS_PATH,
    OPENAI_PACKAGE,
    SELF_COLLABORATION_COMMIT,
    SELF_COLLABORATION_REPOSITORY,
    TASK_INSTRUCTION_PATH,
    TASK_WORKSPACE,
    SelfCollaborationAgent,
)
from evaluation_platform import run_self_collaboration
from evaluation_platform.failure_categories import FailureTally
from evaluation_platform.model_usage import UsageTotals, record_response_usage
from evaluation_platform.run_self_collaboration import (
    _initialize_repository,
    _model_settings,
    _read_instruction,
    _usage_recording_call,
)


class RecordingEnvironment:
    default_user = "root"

    def __init__(self):
        self.commands = []
        self.uploads = []
        self.upload_contents = []

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target):
        self.uploads.append((Path(source), target))
        self.upload_contents.append(Path(source).read_text(encoding="utf-8"))

    async def download_file(self, source, target):
        Path(target).write_text(
            '{"input_tokens": 0, "cached_input_tokens": 0, '
            '"output_tokens": 0, "cost_usd": null}',
            encoding="utf-8",
        )


def test_install_pins_upstream_and_uploads_runner(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert SELF_COLLABORATION_COMMIT in commands
    assert "git clone" in commands
    assert OPENAI_PACKAGE in commands
    assert environment.uploads[0][1] == "/installed-agent/run_self_collaboration.py"


def test_run_uploads_instruction_instead_of_putting_it_on_docker_command_line(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(
        logs_dir=tmp_path,
        extra_env={"OPENROUTER_API_KEY": "secret", "MODEL": "test/model"},
    )
    instruction = "Build the library\n" * 10_000

    asyncio.run(
        agent.run(
            instruction,
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    invocation = environment.commands[-1]
    uploaded = dict(zip([target for _, target in environment.uploads],
                        environment.upload_contents))
    assert "Build the library" not in invocation["command"]
    assert "secret" not in invocation["command"]
    assert invocation["env"] == {
        "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
        "SELF_COLLABORATION_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
    }
    assert uploaded[TASK_INSTRUCTION_PATH] == instruction
    assert agent._extra_env["OPENROUTER_API_KEY"] == "secret"
    assert invocation["cwd"] == TASK_WORKSPACE


def test_run_uploads_the_configured_hyperparameters_for_the_runner(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(
        logs_dir=tmp_path,
        max_rounds=4,
        coder_steps=25,
        test_command="pytest -q",
    )

    asyncio.run(
        agent.run(
            "Build the library",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    uploaded = dict(zip([target for _, target in environment.uploads],
                        environment.upload_contents))
    hyperparameters = json.loads(uploaded[HYPERPARAMETERS_PATH])
    assert hyperparameters["max_rounds"] == 4
    assert hyperparameters["coder_steps"] == 25
    assert hyperparameters["test_command"] == "pytest -q"
    # Defaults are filled in, so the uploaded file is the complete setup.
    assert hyperparameters["analyst_steps"] == 10
    assert hyperparameters["temperature"] == 0.0


def test_agent_rejects_a_hyperparameter_value_the_runner_cannot_use(tmp_path):
    with pytest.raises(ConfigurationError):
        SelfCollaborationAgent(logs_dir=tmp_path, max_rounds=0)


def test_install_uses_the_configured_tool_revision(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(
        logs_dir=tmp_path,
        repository="https://example.invalid/fork.git",
        commit="0123456789abcdef0123456789abcdef01234567",
    )

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert "https://example.invalid/fork.git" in commands
    assert "0123456789abcdef0123456789abcdef01234567" in commands
    assert SELF_COLLABORATION_COMMIT not in commands


class RecordingSession:
    """Stands in for Self-Collaboration's Analyst -> Coder <-> Tester session."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_kwargs = None
        RecordingSession.instances.append(self)

    def run(self, task_description, test_cmd=None):
        self.run_kwargs = {"task": task_description, "test_cmd": test_cmd}
        return {}, "analysis", "patch"


def install_fake_self_collaboration(monkeypatch, tmp_path):
    """Install stand-ins for the modules the runner imports in the container."""
    RecordingSession.instances.clear()

    class ModelConfig:
        def __init__(self, max_tokens=4096, temperature=0.0, top_p=0.95,
                     reasoning_effort=None, extra_body=None):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.reasoning_effort = reasoning_effort
            self.extra_body = extra_body
            self.model = "test/free-model"
            self.base_url = "https://openrouter.ai/api/v1"

    agent_module = SimpleNamespace(
        SelfCollabSession=RecordingSession,
        call_llm_with_tools=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "core.agent", agent_module)
    monkeypatch.setitem(sys.modules, "core.config", SimpleNamespace(ModelConfig=ModelConfig))
    monkeypatch.setitem(
        sys.modules,
        "core.repo_tools",
        SimpleNamespace(get_repo_structure=lambda path: "(empty)"),
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(run_self_collaboration, "WORKSPACE", workspace)
    monkeypatch.setattr(run_self_collaboration, "SELF_COLLABORATION_ROOT", tmp_path)
    monkeypatch.setattr(run_self_collaboration, "HISTORY_PATH", logs / "history.json")
    monkeypatch.setattr(run_self_collaboration, "USAGE_PATH", logs / "usage.json")
    monkeypatch.setattr(
        run_self_collaboration, "RESOLVED_SETUP_PATH", logs / "resolved-setup.json"
    )
    monkeypatch.setattr(run_self_collaboration, "_initialize_repository", lambda: None)

    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Build the library", encoding="utf-8")
    monkeypatch.setenv("HARBOR_TASK_INSTRUCTION_PATH", str(instruction_path))
    return logs


def run_runner(monkeypatch, tmp_path, hyperparameters):
    logs = install_fake_self_collaboration(monkeypatch, tmp_path)
    hyperparameters_path = tmp_path / "hyperparameters.json"
    hyperparameters_path.write_text(json.dumps(hyperparameters), encoding="utf-8")
    monkeypatch.setenv(
        "SELF_COLLABORATION_HYPERPARAMETERS_PATH", str(hyperparameters_path)
    )

    run_self_collaboration.main()

    return RecordingSession.instances[-1], logs


def test_runner_hands_the_tester_its_test_command(tmp_path, monkeypatch):
    session, _ = run_runner(
        monkeypatch,
        tmp_path,
        {
            "max_rounds": 3,
            "analyst_steps": 8,
            "coder_steps": 20,
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
            "test_command": "python -m pytest -q",
        },
    )

    # Both are required: the Tester phase runs between Coder rounds, so it is
    # unreachable with a single round even when a test command is configured.
    assert session.run_kwargs["test_cmd"] == "python -m pytest -q"
    assert session.kwargs["max_round"] == 3
    assert session.kwargs["analyst_steps"] == 8
    assert session.kwargs["coder_steps"] == 20


def test_runner_leaves_the_tester_out_when_no_test_command_is_configured(
        tmp_path, monkeypatch
):
    session, _ = run_runner(
        monkeypatch,
        tmp_path,
        {
            "max_rounds": 1,
            "analyst_steps": 10,
            "coder_steps": 15,
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
        },
    )

    assert session.run_kwargs["test_cmd"] is None


def test_runner_records_whether_reasoning_was_requested(tmp_path, monkeypatch):
    _, logs = run_runner(
        monkeypatch,
        tmp_path,
        {
            "max_rounds": 1,
            "analyst_steps": 10,
            "coder_steps": 15,
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
            "reasoning_effort": "high",
        },
    )

    recorded = json.loads(
        (logs / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["reasoning_requested"] is True
    assert recorded["hyperparameters"]["reasoning_effort"] == "high"


def test_runner_records_the_sampling_every_role_actually_used(tmp_path, monkeypatch):
    """Read off the config object the tool's own requests were built from.

    One `ModelConfig` serves the Analyst, the Coder and the Tester alike, so
    these are the values every role sampled at. `top_p` is recorded as inert
    because this revision declares it and sends it only from its non-agentic
    entry point — the record says what was resolved and what that was worth.
    """
    _, logs = run_runner(
        monkeypatch,
        tmp_path,
        {
            "max_rounds": 1,
            "analyst_steps": 10,
            "coder_steps": 15,
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
        },
    )

    recorded = json.loads(
        (logs / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["generation"]["temperature"] == 0.2
    assert recorded["generation"]["top_p"] == 0.9
    assert recorded["generation"]["max_tokens"] == 4096
    assert recorded["generation"]["source"] == "configs/generation.yaml"
    assert recorded["generation"]["inert_on_this_revision"] == ["top_p"]


def test_runner_records_the_failure_categories_separately_from_a_reward(
        tmp_path, monkeypatch
):
    """Every category, at zero, on a run that hit none of them.

    A run that reached no ceiling and a run whose logging predates these
    categories must not look alike in the record, so all three are written.
    """
    _, logs = run_runner(
        monkeypatch,
        tmp_path,
        {
            "max_rounds": 1,
            "analyst_steps": 10,
            "coder_steps": 15,
            "max_tokens": 8192,
            "temperature": 0.0,
            "top_p": 0.95,
        },
    )

    recorded = json.loads(
        (logs / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["failures"]["counts"] == {
        "output_limit_exhausted": 0,
        "context_exhausted": 0,
        "request_timeout": 0,
    }
    assert "functional test" in recorded["failures"]["excludes"]


def test_a_truncated_role_reply_is_counted_wherever_the_tool_calls_the_model():
    """The tool has one call site for every role, so wrapping it covers them all.

    A Coder whose `edit_file` call arrives half-written is not a Coder that made
    a wrong edit, and only this tells them apart.
    """
    failures = FailureTally()
    call = _usage_recording_call(
        lambda *args, **kwargs: SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(finish_reason="length")],
        ),
        UsageTotals(),
        failures=failures,
        sleep=lambda seconds: None,
    )

    call([], SimpleNamespace())

    assert failures.to_dict()["counts"]["output_limit_exhausted"] == 1


def test_runner_records_the_setup_that_actually_ran(tmp_path, monkeypatch):
    hyperparameters = {
        "max_rounds": 2,
        "analyst_steps": 10,
        "coder_steps": 15,
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "test_command": "python -m pytest -q",
    }

    session, logs = run_runner(monkeypatch, tmp_path, hyperparameters)

    recorded = json.loads(
        (logs / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["hyperparameters"] == hyperparameters
    assert recorded["tester_enabled"] is True
    assert recorded["reasoning_requested"] is False
    assert recorded["model"] == "test/free-model"
    assert session.kwargs["config"].max_tokens == 4096
    assert session.kwargs["config"].temperature == 0.2
    assert session.kwargs["config"].top_p == 0.9


def test_runner_reads_utf8_instruction_file(tmp_path, monkeypatch):
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Verify ∪ and 中文 input", encoding="utf-8")
    monkeypatch.setenv("HARBOR_TASK_INSTRUCTION_PATH", str(instruction_path))

    assert _read_instruction() == "Verify ∪ and 中文 input"


def test_record_response_usage_aggregates_openrouter_metrics():
    totals = UsageTotals()
    first_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=18),
            cost=0.0015,
        )
    )
    second_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=10,
            prompt_tokens_details={"cached_tokens": 5},
            completion_tokens_details={"reasoning_tokens": 7},
            cost=0.0005,
        )
    )

    record_response_usage(first_response, totals)
    record_response_usage(second_response, totals)

    assert totals.to_dict() == {
        "input_tokens": 200,
        "cached_input_tokens": 25,
        "output_tokens": 40,
        # Billed inside output_tokens; recorded separately so a run shows how
        # much of its budget went to reasoning rather than to answering.
        "reasoning_tokens": 25,
        "cost_usd": 0.002,
    }


def test_usage_without_reasoning_details_records_no_reasoning_tokens():
    totals = UsageTotals()

    record_response_usage(
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        ),
        totals,
    )

    assert totals.to_dict()["reasoning_tokens"] == 0


def test_sampling_settings_always_reach_the_model():
    settings = _model_settings(
        {"max_tokens": 4096, "temperature": 0.2, "top_p": 0.9}
    )

    assert settings == {"max_tokens": 4096, "temperature": 0.2, "top_p": 0.9}


def test_reasoning_settings_are_sent_only_when_configured():
    unset = _model_settings({"max_tokens": 1, "temperature": 0.0, "top_p": 1.0})
    effort = _model_settings(
        {
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_effort": "high",
        }
    )
    provider_specific = _model_settings(
        {
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "request_extra": {"reasoning": {"effort": "high"}},
        }
    )

    # Unset leaves the provider default in force, and lets an unmodified
    # revision of the tool still be pinned for a baseline run.
    assert "reasoning_effort" not in unset
    assert "extra_body" not in unset
    assert effort["reasoning_effort"] == "high"
    assert provider_specific["extra_body"] == {"reasoning": {"effort": "high"}}


def test_usage_wrapper_retries_exhausted_upstream_rate_limits():
    attempts = 0
    response = SimpleNamespace(usage=None)

    def rate_limited_then_success(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Failed to call LLM API with tools")
        return response

    sleeps = []
    wrapped = _usage_recording_call(
        rate_limited_then_success,
        UsageTotals(),
        model="test/free-model",
        sleep=sleeps.append,
    )

    assert wrapped() is response
    assert attempts == 3
    assert sleeps == [15, 30]


def test_usage_wrapper_reports_rate_limit_after_bounded_retries():
    def always_rate_limited(*args, **kwargs):
        raise RuntimeError("Failed to call LLM API with tools")

    wrapped = _usage_recording_call(
        always_rate_limited,
        UsageTotals(),
        model="test/free-model",
        sleep=lambda _: None,
    )

    try:
        wrapped()
    except RuntimeError as error:
        assert "rate-limited" in str(error)
        assert "test/free-model" in str(error)
        assert "API_KEY" not in str(error)
    else:
        raise AssertionError("Expected exhausted rate limits to fail")


def test_post_run_populates_harbor_context_from_usage_log(tmp_path):
    (tmp_path / "model-usage.json").write_text(
        '{"input_tokens": 90, "cached_input_tokens": 15, '
        '"output_tokens": 25, "cost_usd": 0.0}',
        encoding="utf-8",
    )
    context = AgentContext()
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 90
    assert context.n_cache_tokens == 15
    assert context.n_output_tokens == 25
    assert context.cost_usd == 0.0


def test_an_unreadable_tool_repository_fails_instead_of_waiting_for_a_password(tmp_path):
    """A private or misspelled repository is a fast failure, not a stalled trial."""
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    clone = next(
        command for command in environment.commands if "git clone" in command["command"]
    )
    assert clone["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_the_tool_is_pinned_to_the_authors_own_repository():
    """The evaluated revision carries nothing of ours.

    An earlier revision of this platform pinned a fork with three patches on
    top of upstream. Those adapted the tool to tasks built from scratch, but
    they also made the measured system a modified one; the accommodation this
    platform still makes lives in the experiment's test command instead.
    """
    assert "YihongDong" in SELF_COLLABORATION_REPOSITORY
    assert SELF_COLLABORATION_COMMIT == "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"


def test_build_output_stays_out_of_the_repository_view_the_tool_reads(
        tmp_path, monkeypatch
):
    """What `git add --intent-to-add` in the test command is allowed to register.

    The tool reads `git diff` both to decide whether the Coder produced
    anything and to show a later round what an earlier one did. The test
    command registers the generated files so that command can see them; were
    it to sweep up what `pip install` and `pytest` leave behind as well, a
    round in which the Coder wrote nothing would still report work, and the
    record of the previous attempt would be padded with build output.
    """
    monkeypatch.setattr(run_self_collaboration, "WORKSPACE", tmp_path)
    _initialize_repository()

    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    package = tmp_path / "src" / "library"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("value = 1", encoding="utf-8")
    (package / "__pycache__" / "m.cpython-313.pyc").write_bytes(bytes(4))
    egg_info = tmp_path / "src" / "library.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1", encoding="utf-8")

    registered = _intent_to_add(tmp_path, "--name-only").split()

    assert sorted(registered) == ["pyproject.toml", "src/library/__init__.py"]


def test_build_output_alone_still_reads_as_no_code_changes(tmp_path, monkeypatch):
    """The check the test command must not defeat.

    The session substitutes a failure where it finds no change, which is what
    stops a Coder that edited nothing from being credited with a passing test
    run. Installing the package and running the tests leaves files behind on
    its own, and those must not read as work.
    """
    monkeypatch.setattr(run_self_collaboration, "WORKSPACE", tmp_path)
    _initialize_repository()

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "m.cpython-313.pyc").write_bytes(bytes(4))

    assert _intent_to_add(tmp_path).strip() == ""


def _intent_to_add(workspace, *diff_arguments):
    """Register untracked files as the experiment's test command does, then diff."""
    subprocess.run(
        ["git", "add", "--intent-to-add", "."], cwd=workspace, check=True
    )
    return subprocess.run(
        ["git", "diff", *diff_arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
