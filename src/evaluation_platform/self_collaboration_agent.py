import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import SELF_COLLABORATION
from evaluation_platform.model_usage import populate_usage_context

# The authors' own repository at its current head, evaluated unmodified.
# An earlier revision of this platform pinned a fork carrying three patches
# that adapted the tool to tasks built from scratch. The fork remains for
# discussion, but no longer takes part in benchmarking, so that a measured
# difference is attributable to the published method rather than to our
# changes to it. What those patches worked around, and what running without
# them costs, is described in docs/latex/tool-modifications.tex.
SELF_COLLABORATION_COMMIT = "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"
SELF_COLLABORATION_REPOSITORY = (
    "https://github.com/YihongDong/Self-collaboration-Code-Generation.git"
)
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
TASK_WORKSPACE = "/workspace"
# Git asks a terminal for credentials it was not given, and a task
# container has no terminal to ask.
GIT_NON_INTERACTIVE = {"GIT_TERMINAL_PROMPT": "0"}
USAGE_FILENAME = "model-usage.json"
OPENAI_PACKAGE = "openai==2.54.0"


class SelfCollaborationAgent(BaseInstalledAgent):
    """Run Self-Collaboration directly in Harbor's task workspace.

    The tool revision and the hyperparameters of its Analyst, Coder, and Tester
    roles come from the experiment configuration by way of Harbor's agent
    `kwargs`, which Harbor records in the job's `config.json`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in SELF_COLLABORATION.types
        }
        self.repository = kwargs.pop("repository", SELF_COLLABORATION_REPOSITORY)
        self.commit = kwargs.pop("commit", SELF_COLLABORATION_COMMIT)
        SELF_COLLABORATION.reject_foreign(kwargs)
        super().__init__(*args, **kwargs)
        self.hyperparameters = SELF_COLLABORATION.resolve(hyperparameters)

    @staticmethod
    def name() -> str:
        return "self-collaboration"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("git",))
        await self.exec_as_root(
            environment,
            command=f"python -m pip install --no-cache-dir '{OPENAI_PACKAGE}'",
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "rm -rf /installed-agent/self-collaboration; "
                f"git clone --quiet {self.repository} "
                "/installed-agent/self-collaboration; "
                "git -C /installed-agent/self-collaboration checkout --quiet "
                f"{self.commit}"
            ),
            # The container has no credentials and nobody to ask for them. A
            # repository that is private, renamed, or misspelled otherwise
            # reaches Git as an authentication challenge, which it answers by
            # prompting: a trial that stalls until its own timeout rather than
            # reporting in seconds that it could not read the repository.
            env=GIT_NON_INTERACTIVE,
        )
        for module in (
                "run_self_collaboration.py",
                "model_usage.py",
                "model_routing.py",
        ):
            await self._upload_agent_owned_file(
                environment,
                Path(__file__).with_name(module),
                f"/installed-agent/{module}",
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
            hyperparameters_source = Path(temp_dir) / "hyperparameters.json"
            hyperparameters_source.write_text(
                json.dumps(self.hyperparameters, indent=2),
                encoding="utf-8",
            )
            await self._upload_agent_owned_file(
                environment,
                hyperparameters_source,
                HYPERPARAMETERS_PATH,
            )
            await self.exec_as_agent(
                environment,
                command=(
                    "python -u /installed-agent/run_self_collaboration.py "
                    "2>&1 | tee /logs/agent/self-collaboration.log"
                ),
                env={
                    "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
                    "SELF_COLLABORATION_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_usage_context(context, self.logs_dir / USAGE_FILENAME)
