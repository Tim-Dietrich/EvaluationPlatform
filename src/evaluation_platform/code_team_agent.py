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
# Upstream, not a fork. What this project changes in CodeTeam travels as the
# patch below, applied to this commit after checkout, so that the deviation
# stays legible in the harness and in every run's own record rather than
# disappearing into a branch nobody diffs. Pointing a configuration at a fork
# is still a matter of setting `agent.repository`.
CODE_TEAM_REPOSITORY = "https://github.com/WhitenWhiten/CodeTeam.git"
CODE_TEAM_ROOT = "/installed-agent/code-team"
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
# Two defects on the asynchronous path, which is the one a run takes. Neither
# is a matter of how CodeTeam decides anything, and each is contradicted by
# upstream's own code, so the patch restores stated behaviour rather than
# altering it.
#
# The QA role runs pytest and hands what it reads to the router that decides
# which developer repairs which file. The scraper in between keeps only lines
# beginning `E   ` or `Traceback`, or carrying `FAILED` or `ERROR at`, and
# pytest names the failing source file on none of those: it prints that on its
# own location lines, `path/to/file.py:LINENO:`. Every failure therefore
# reached the router with no source path, was dropped as unroutable, and the
# repair loop stopped before its first round — the arm's defining mechanism,
# never running. Upstream's `tests/test_qa_routing.py` asserts the successful
# QA run this prevents. The router itself is untouched and still chooses the
# file and the developer on its own.
#
# The second is a shallow `__dict__` in the asynchronous workflow's `sds_map`.
# A `ClassBrief`'s `methods` survives it as a list of `FuncBrief` objects, and
# the developer prompt subscripts them, so any file whose specification
# declares a class method ends the run before a line is written. The
# synchronous workflow already converts recursively, under a comment saying to
# keep doing so; this ports that conversion across.
CODE_TEAM_PATCH = "code_team_upstream_fixes.patch"
CODE_TEAM_PATCH_PATH = f"/installed-agent/{CODE_TEAM_PATCH}"
TASK_WORKSPACE = "/workspace"
# The tester sidecar grades with the workspace importable: it copies the agent's
# tree over the reference checkout, runs `pip install -e .`, and imports the
# package from there. The agent container offers QA no equivalent, so CodeTeam's
# `pytest -q .codeteam_qa/tests` puts only that test directory on `sys.path` and
# cannot import the package it was asked to test — a collection error before any
# assertion runs, which the tool reads as a failure it has no fix for. Giving the
# run the view the grader already has costs one variable, and is the same one the
# benchmark's own test image sets.
AGENT_PYTHONPATH = TASK_WORKSPACE
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
    # Pinned to the version the tester sidecar grades with, which is also the
    # version the task specification tells the agent is installed: a repair
    # round and the verifier should not disagree about what collects and how a
    # failure is reported.
    "pytest==8.4.1",
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
        await self._upload_agent_owned_file(
            environment,
            Path(__file__).with_name(CODE_TEAM_PATCH),
            CODE_TEAM_PATCH_PATH,
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"rm -rf {CODE_TEAM_ROOT}; "
                f"git clone --quiet {self.repository} {CODE_TEAM_ROOT}; "
                f"git -C {CODE_TEAM_ROOT} checkout --quiet {self.commit}; "
                # Deliberately strict, and part of the same failing command: a
                # revision the patch no longer fits should end the trial here,
                # rather than let it run to a reward with the repair loop
                # silently disabled and nothing in the result to say so.
                f"git -C {CODE_TEAM_ROOT} apply {CODE_TEAM_PATCH_PATH}"
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
                    "PYTHONPATH": AGENT_PYTHONPATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_usage_context(context, self.logs_dir / USAGE_FILENAME)
