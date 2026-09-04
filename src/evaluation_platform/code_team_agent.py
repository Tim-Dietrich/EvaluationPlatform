import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    CODE_TEAM,
    resolve_generation_kwargs,
)
from evaluation_platform.model_usage import populate_usage_context

CODE_TEAM_COMMIT = "c095631730bf669c9c9ef1a0e87739e18caf6a7c"
# Upstream, not a fork. Nothing in CodeTeam is modified for this project,
# so there is nothing for a fork to hold; pointing a configuration at one
# is a matter of setting `agent.repository` when there is.
CODE_TEAM_REPOSITORY = "https://github.com/WhitenWhiten/CodeTeam.git"
CODE_TEAM_ROOT = "/installed-agent/code-team"
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
TASK_WORKSPACE = "/workspace"
# Git asks a terminal for credentials it was not given, and a task
# container has no terminal to ask.
GIT_NON_INTERACTIVE = {"GIT_TERMINAL_PROMPT": "0"}
USAGE_FILENAME = "model-usage.json"

# What the tool imports on the path a run actually takes. Its own
# `requirements.txt` additionally names pandas, openpyxl and scipy, which
# belong to the paper's offline analysis script rather than to a run, and
# numpy, which only the retrieval client imports. Installing the run's
# dependencies rather than the repository's keeps a per-trial download from
# carrying a scientific stack no container here executes.
RUNTIME_PACKAGES = (
    "openai==2.54.0",
    "pydantic==2.13.4",
    "jsonschema==4.26.0",
    # The QA role runs pytest in the workspace between repair rounds. Without
    # it the tool silently substitutes a hand-written test runner of its own.
    "pytest==9.1.1",
)
# Retrieval grounding for the Architect stage, installed only for a run that
# asks for it. The vector path is the paper's own and pulls an embedding model
# at first use; the lexical path needs nothing beyond numpy.
RAG_PACKAGES = ("numpy==2.5.2",)
VECTOR_RAG_PACKAGES = ("sentence-transformers==6.0.0", "faiss-cpu==1.15.0")


class CodeTeamAgent(BaseInstalledAgent):
    """Run CodeTeam directly in Harbor's task workspace.

    The tool revision and the hyperparameters of its Architect, CTO, Developer,
    and QA roles come from the experiment configuration by way of Harbor's
    agent `kwargs`, which Harbor records in the job's `config.json`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in CODE_TEAM.types
        }
        self.repository = kwargs.pop("repository", CODE_TEAM_REPOSITORY)
        self.commit = kwargs.pop("commit", CODE_TEAM_COMMIT)
        # The sampling parameters are set for every arm at once and arrive
        # beside the solution's own hyperparameters. Taking them out here keeps
        # them from reaching Harbor's constructor, which has no use for them,
        # and merging them back in below is what puts them in front of the
        # in-container runner under the names it already reads.
        self.generation = resolve_generation_kwargs(kwargs)
        CODE_TEAM.reject_foreign(kwargs)
        super().__init__(*args, **kwargs)
        self.hyperparameters = CODE_TEAM.resolve(hyperparameters) | self.generation

    @staticmethod
    def name() -> str:
        return "code-team"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("git",))
        await self.exec_as_root(
            environment,
            command=(
                "python -m pip install --no-cache-dir "
                + " ".join(f"'{package}'" for package in self._packages())
            ),
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"rm -rf {CODE_TEAM_ROOT}; "
                f"git clone --quiet {self.repository} {CODE_TEAM_ROOT}; "
                f"git -C {CODE_TEAM_ROOT} checkout --quiet {self.commit}"
            ),
            # The container has no credentials and nobody to ask for them. A
            # repository that is private, renamed, or misspelled otherwise
            # reaches Git as an authentication challenge, which it answers by
            # prompting: a trial that stalls until its own timeout rather than
            # reporting in seconds that it could not read the repository.
            env=GIT_NON_INTERACTIVE,
        )
        for module in (
                "run_code_team.py",
                "model_usage.py",
                "model_routing.py",
                "failure_categories.py",
        ):
            await self._upload_agent_owned_file(
                environment,
                Path(__file__).with_name(module),
                f"/installed-agent/{module}",
            )

    def _packages(self) -> tuple[str, ...]:
        """The packages this run needs, which depends on how it is configured.

        Retrieval is off by default and its stack is large, so a run that does
        not use it does not pay to install it.
        """
        if not self.hyperparameters["rag_enabled"]:
            return RUNTIME_PACKAGES
        packages = RUNTIME_PACKAGES + RAG_PACKAGES
        if self.hyperparameters["rag_backend"] == "faiss_hnsw":
            packages += VECTOR_RAG_PACKAGES
        return packages

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
                    "python -u /installed-agent/run_code_team.py "
                    "2>&1 | tee /logs/agent/code-team.log"
                ),
                env={
                    "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
                    "CODE_TEAM_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_usage_context(context, self.logs_dir / USAGE_FILENAME)
