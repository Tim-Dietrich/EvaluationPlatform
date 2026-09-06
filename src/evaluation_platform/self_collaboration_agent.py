import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    SELF_COLLABORATION,
    resolve_generation_kwargs,
)
from evaluation_platform.model_usage import populate_usage_context
from evaluation_platform.self_collaboration_trajectory import write_trajectory

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

# The tool's two entry points, and the runner that drives each. `repository`
# is `core.agent.SelfCollabSession`, which the authors' SWE scripts use;
# `humaneval` is `run_humaneval.py`, which is what `bash run.sh` runs and so is
# the path behind the paper's HumanEval figures. `experiment_config` describes
# how the two differ and which hyperparameters reach which.
RUNNERS = {
    "repository": "run_self_collaboration.py",
    "humaneval": "run_humaneval_self_collaboration.py",
}
# Git asks a terminal for credentials it was not given, and a task
# container has no terminal to ask.
GIT_NON_INTERACTIVE = {"GIT_TERMINAL_PROMPT": "0"}
USAGE_FILENAME = "model-usage.json"

# The tool's dependencies, pinned. `requirements.txt` at the evaluated commit
# asks for `openai>=1.0`, `datasets`, `tqdm`, `docker` and `swebench`, and its
# own Dockerfile installs the file wholesale. Only the first three are
# installed here: `docker` and `swebench` belong to `run_swe*.py`, an entry
# point no configuration of this platform runs, and `swebench` is large.
#
# The versions are pinned where upstream leaves them open, for the reason the
# tool's own commit is pinned: a replication that silently resolves a different
# `datasets` next month is a replication with an unrecorded variable.
OPENAI_PACKAGE = "openai==2.54.0"
# `run_humaneval.py` imports `datasets` and `tqdm` at module scope for the CLI
# loop in its `main()`. The runner calls `run_task` and never `main()`, so
# neither package is reached at run time — but the import is, and the tool's
# environment is reproduced rather than worked around.
HUMANEVAL_PACKAGES = ("datasets==5.0.1", "tqdm==4.70.0")


class SelfCollaborationAgent(BaseInstalledAgent):
    """Run Self-Collaboration directly in Harbor's task workspace.

    The tool revision and the hyperparameters of its Analyst, Coder, and Tester
    roles come from the experiment configuration by way of Harbor's agent
    `kwargs`, which Harbor records in the job's `config.json`.
    """

    # The run is recorded as a trajectory after the fact, by converting the
    # session history the tool already writes. `self_collaboration_trajectory`
    # describes what that conversion can and cannot recover.
    SUPPORTS_ATIF: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in SELF_COLLABORATION.types
        }
        self.repository = kwargs.pop("repository", SELF_COLLABORATION_REPOSITORY)
        self.commit = kwargs.pop("commit", SELF_COLLABORATION_COMMIT)
        # The sampling parameters are set for every arm at once and arrive
        # beside the solution's own hyperparameters. Taking them out here keeps
        # them from reaching Harbor's constructor, which has no use for them,
        # and merging them back in below is what puts them in front of the
        # in-container runner under the names it already reads.
        self.generation = resolve_generation_kwargs(kwargs)
        SELF_COLLABORATION.reject_foreign(kwargs)
        super().__init__(*args, **kwargs)
        self.hyperparameters = SELF_COLLABORATION.resolve(hyperparameters) | self.generation
        self.task_shape = self.hyperparameters["task_shape"]
        # The HumanEval entry point returns its generated code and writes no
        # session history, so there is nothing for the converter to read. Said
        # here rather than left to fail quietly: a trajectory that is absent
        # because the code path produces none is different from one that is
        # absent because the conversion broke.
        if self.task_shape != "repository":
            self.SUPPORTS_ATIF = False
        # Kept from `run` so that the trajectory written afterwards can open on
        # the task the solution was given. The session history the tool writes
        # does not record it, and the conversion happens after the container is
        # gone, so this is the only place it survives.
        self.instruction: str | None = None

    @staticmethod
    def name() -> str:
        return "self-collaboration"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("git",))
        packages = [OPENAI_PACKAGE]
        if self.task_shape == "humaneval":
            packages.extend(HUMANEVAL_PACKAGES)
        await self.exec_as_root(
            environment,
            command=(
                "python -m pip install --no-cache-dir "
                + " ".join(f"'{package}'" for package in packages)
            ),
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
        # Both runners are uploaded whichever shape runs: the HumanEval one
        # imports the shared usage wrapper from the other, and a module that
        # is present but unused costs nothing.
        for module in (
                "run_self_collaboration.py",
                "run_humaneval_self_collaboration.py",
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
        self.instruction = instruction
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
            runner = RUNNERS[self.task_shape]
            await self.exec_as_agent(
                environment,
                command=(
                    f"python -u /installed-agent/{runner} "
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
        # Harbor has synced the trial's logs back by now, so the files the
        # runner wrote inside the container are readable here.
        write_trajectory(
            self.logs_dir,
            instruction=self.instruction,
            session_id=self.session_id,
            logger=self.logger,
        )
