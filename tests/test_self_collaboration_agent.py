import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from src.evaluation_platform.self_collaboration_agent import (
    SELF_COLLABORATION_COMMIT,
    TASK_INSTRUCTION_PATH,
    SelfCollaborationAgent,
)
from src.evaluation_platform.run_self_collaboration import _read_instruction


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


def test_install_pins_upstream_and_uploads_runner(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert SELF_COLLABORATION_COMMIT in commands
    assert "git clone" in commands
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
    assert invocation["cwd"] == "/app"


def test_runner_reads_utf8_instruction_file(tmp_path, monkeypatch):
    instruction_path = tmp_path / "instruction.md"
    instruction_path.write_text("Verify ∪ and 中文 input", encoding="utf-8")
    monkeypatch.setenv("HARBOR_TASK_INSTRUCTION_PATH", str(instruction_path))

    assert _read_instruction() == "Verify ∪ and 中文 input"
