import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from src.evaluation_platform.self_collaboration_agent import (
    OPENAI_PACKAGE,
    SELF_COLLABORATION_COMMIT,
    TASK_INSTRUCTION_PATH,
    TASK_WORKSPACE,
    SelfCollaborationAgent,
)
from src.evaluation_platform.run_self_collaboration import (
    UsageTotals,
    _read_instruction,
    _usage_recording_call,
    _record_response_usage,
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
    assert "Build the library" not in invocation["command"]
    assert "secret" not in invocation["command"]
    assert invocation["env"] == {
        "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH
    }
    assert environment.uploads[-1][1] == TASK_INSTRUCTION_PATH
    assert environment.upload_contents[-1] == instruction
    assert agent._extra_env["OPENROUTER_API_KEY"] == "secret"
    assert invocation["cwd"] == TASK_WORKSPACE


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
            cost=0.0015,
        )
    )
    second_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=10,
            prompt_tokens_details={"cached_tokens": 5},
            cost=0.0005,
        )
    )

    _record_response_usage(first_response, totals)
    _record_response_usage(second_response, totals)

    assert totals.to_dict() == {
        "input_tokens": 200,
        "cached_input_tokens": 25,
        "output_tokens": 40,
        "cost_usd": 0.002,
    }


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
