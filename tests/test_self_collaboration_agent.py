import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from src.evaluation_platform.self_collaboration_agent import (
    SELF_COLLABORATION_COMMIT,
    SelfCollaborationAgent,
)


class RecordingEnvironment:
    default_user = "root"

    def __init__(self):
        self.commands = []
        self.uploads = []

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target):
        self.uploads.append((Path(source), target))


def test_install_pins_upstream_and_uploads_runner(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert SELF_COLLABORATION_COMMIT in commands
    assert "git clone" in commands
    assert environment.uploads[0][1] == "/installed-agent/run_self_collaboration.py"


def test_run_passes_instruction_and_credentials_only_via_environment(tmp_path):
    environment = RecordingEnvironment()
    agent = SelfCollaborationAgent(
        logs_dir=tmp_path,
        extra_env={"OPENROUTER_API_KEY": "secret", "MODEL": "test/model"},
    )

    asyncio.run(
        agent.run(
            "Build the library",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    invocation = environment.commands[-1]
    assert "Build the library" not in invocation["command"]
    assert "secret" not in invocation["command"]
    assert invocation["env"]["HARBOR_TASK_INSTRUCTION"] == "Build the library"
    assert agent._extra_env["OPENROUTER_API_KEY"] == "secret"
    assert invocation["cwd"] == "/app"
