from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SELF_COLLABORATION_COMMIT = "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"
SELF_COLLABORATION_REPOSITORY = (
    "https://github.com/Tim-Dietrich/Self-collaboration-Code-Generation.git"
)
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"


class SelfCollaborationAgent(BaseInstalledAgent):
    """Run Self-Collaboration directly in Harbor's task workspace."""

    @staticmethod
    def name() -> str:
        return "self-collaboration"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("git",))
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "rm -rf /installed-agent/self-collaboration; "
                f"git clone --quiet {SELF_COLLABORATION_REPOSITORY} "
                "/installed-agent/self-collaboration; "
                "git -C /installed-agent/self-collaboration checkout --quiet "
                f"{SELF_COLLABORATION_COMMIT}; "
                "python -m pip install --user --quiet 'openai>=1.0'"
            ),
        )
        await self._upload_agent_owned_file(
            environment,
            Path(__file__).with_name("run_self_collaboration.py"),
            "/installed-agent/run_self_collaboration.py",
        )

    async def run(
            self,
            instruction: str,
            environment: BaseEnvironment,
            context: AgentContext,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            instruction_source = Path(temp_dir) / "task-instruction.md"
            instruction_source.write_text(instruction, encoding="utf-8")
            await self._upload_agent_owned_file(
                environment,
                instruction_source,
                TASK_INSTRUCTION_PATH,
            )
            await self.exec_as_agent(
                environment,
                command=(
                    "python /installed-agent/run_self_collaboration.py "
                    "2>&1 | tee /logs/agent/self-collaboration.log"
                ),
                env={"HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH},
                cwd="/app",
            )
