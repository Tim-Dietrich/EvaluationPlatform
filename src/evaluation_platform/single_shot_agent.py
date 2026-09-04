"""The Single-Shot baseline: one request, one reply, one workspace.

Every code generation solution in this platform claims, in one form or
another, to improve on prompting a model directly. This arm is that direct
prompt, run through the same benchmark, the same grading path and the same
usage accounting as the solutions it exists to be compared against, so that a
difference between them is a difference in method rather than in apparatus.

It has no upstream repository. Where the other adapters pin a revision of
somebody else's tool, this one's subject *is* the platform: the prompt and the
runner that sends it, both recorded by digest with every run. See
`run_single_shot.py` for why the prompt is fixed there rather than exposed as
a hyperparameter.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    SINGLE_SHOT,
    resolve_generation_kwargs,
)
from evaluation_platform.model_usage import populate_usage_context

TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
TASK_WORKSPACE = "/workspace"
USAGE_FILENAME = "model-usage.json"
# The same client, at the same version, as the three solutions this baseline is
# compared against. What a request reports about what it spent is read from the
# response by `model_usage.py`, so the arms only stay comparable while they are
# all asking the same library the same way.
OPENAI_PACKAGE = "openai==2.54.0"
# What identifies a Single-Shot run in place of an upstream revision.
IDENTITY = (
    "the prompt it sends and the runner that sends it, both recorded as "
    "digests in the run's resolved-setup.json"
)


class SingleShotAgent(BaseInstalledAgent):
    """Send one request per task and write the files its reply names.

    The hyperparameters come from the experiment configuration by way of
    Harbor's agent `kwargs`, which Harbor records in the job's `config.json`,
    and the agent hands them to the in-container runner.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in SINGLE_SHOT.types
        }
        SINGLE_SHOT.reject_upstream_revision(kwargs, IDENTITY)
        # The sampling parameters are set for every arm at once and arrive
        # beside the solution's own hyperparameters. Taking them out here keeps
        # them from reaching Harbor's constructor, which has no use for them,
        # and merging them back in below is what puts them in front of the
        # in-container runner under the names it already reads.
        self.generation = resolve_generation_kwargs(kwargs)
        SINGLE_SHOT.reject_foreign(kwargs)
        super().__init__(*args, **kwargs)
        self.hyperparameters = SINGLE_SHOT.resolve(hyperparameters) | self.generation

    @staticmethod
    def name() -> str:
        return "single-shot"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=f"python -m pip install --no-cache-dir '{OPENAI_PACKAGE}'",
        )
        for module in (
                "run_single_shot.py",
                "model_usage.py",
                "model_routing.py",
                "failure_categories.py",
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
                    "python -u /installed-agent/run_single_shot.py "
                    "2>&1 | tee /logs/agent/single-shot.log"
                ),
                env={
                    "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
                    "SINGLE_SHOT_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_usage_context(context, self.logs_dir / USAGE_FILENAME)
