import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    HYPERPARAMETER_TYPES,
    resolve_hyperparameters,
)

SELF_COLLABORATION_COMMIT = "d5f8a2339bbabd5f892dcffd04469cdc512cb03a"
SELF_COLLABORATION_REPOSITORY = (
    "https://github.com/Tim-Dietrich/Self-collaboration-Code-Generation.git"
)
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
TASK_WORKSPACE = "/workspace"
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
            if name in HYPERPARAMETER_TYPES
        }
        self.repository = kwargs.pop("repository", SELF_COLLABORATION_REPOSITORY)
        self.commit = kwargs.pop("commit", SELF_COLLABORATION_COMMIT)
        super().__init__(*args, **kwargs)
        self.hyperparameters = resolve_hyperparameters(hyperparameters)

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
                    "python /installed-agent/run_self_collaboration.py "
                    "2>&1 | tee /logs/agent/self-collaboration.log"
                ),
                env={
                    "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
                    "SELF_COLLABORATION_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        self._populate_usage_context(context, self.logs_dir / USAGE_FILENAME)

    @staticmethod
    def _populate_usage_context(context: AgentContext, usage_path: Path) -> None:
        if not usage_path.exists():
            return
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(usage.get("input_tokens"), int):
            context.n_input_tokens = usage["input_tokens"]
        if isinstance(usage.get("cached_input_tokens"), int):
            context.n_cache_tokens = usage["cached_input_tokens"]
        if isinstance(usage.get("output_tokens"), int):
            context.n_output_tokens = usage["output_tokens"]
        if isinstance(usage.get("cost_usd"), (int, float)):
            context.cost_usd = float(usage["cost_usd"])
