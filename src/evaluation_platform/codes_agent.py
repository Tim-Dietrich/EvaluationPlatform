import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    CODE_S,
    resolve_generation_kwargs,
)
from evaluation_platform.model_usage import populate_usage_context

CODES_COMMIT = "0b624ab4ef22b0d9d223f274a986eb27fe090c88"
# Upstream, not a fork. No file of CodeS is modified for this project; what is
# supplied around it is described in docs/latex/tool-modifications.tex.
# Pointing a configuration at a fork is a matter of setting `agent.repository`
# when there is one.
CODES_REPOSITORY = "https://github.com/NL2Code/CodeS.git"
CODES_ROOT = "/installed-agent/codes"
# The only part of the checkout a run reads: the tool's OpenAI-compatible
# driver, the phase helpers beside it, and the sketch utilities at the root.
CODES_SPARSE_PATH = "validation/evaluation_scripts/baselines"
TASK_INSTRUCTION_PATH = "/installed-agent/task-instruction.md"
HYPERPARAMETERS_PATH = "/installed-agent/hyperparameters.json"
TASK_WORKSPACE = "/workspace"
# Git asks a terminal for credentials it was not given, and a task
# container has no terminal to ask.
GIT_NON_INTERACTIVE = {"GIT_TERMINAL_PROMPT": "0"}
USAGE_FILENAME = "model-usage.json"

# What the tool imports on the path a run actually takes. Its own
# `requirements.txt` names `tree-sitter`, `isort` and `autopep8` besides, which
# belong to the corpus preparation scripts that build its training data rather
# than to a run, and its inference driver additionally imports `torch` and
# `transformers`, which serve the fine-tuned model this platform does not use.
# `astor` and `black` are the sketch machinery itself: every file sketch is
# parsed, rewritten as a syntax tree, and formatted before it becomes a prompt
# or a file.
RUNTIME_PACKAGES = (
    "openai==2.54.0",
    "astor==0.8.1",
    "black==26.5.1",
    "tqdm==4.70.0",
)


class CodeSAgent(BaseInstalledAgent):
    """Run CodeS directly in Harbor's task workspace.

    The tool revision and the hyperparameters of its RepoSketcher,
    FileSketcher, and SketchFiller phases come from the experiment
    configuration by way of Harbor's agent `kwargs`, which Harbor records in
    the job's `config.json`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name)
            for name in list(kwargs)
            if name in CODE_S.types
        }
        self.repository = kwargs.pop("repository", CODES_REPOSITORY)
        self.commit = kwargs.pop("commit", CODES_COMMIT)
        # The sampling parameters are set for every arm at once and arrive
        # beside the solution's own hyperparameters. Taking them out here keeps
        # them from reaching Harbor's constructor, which has no use for them,
        # and merging them back in below is what puts them in front of the
        # in-container runner under the names it already reads.
        self.generation = resolve_generation_kwargs(kwargs)
        CODE_S.reject_foreign(kwargs)
        super().__init__(*args, **kwargs)
        self.hyperparameters = CODE_S.resolve(hyperparameters) | self.generation

    @staticmethod
    def name() -> str:
        return "codes"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("git",))
        await self.exec_as_root(
            environment,
            command=(
                "python -m pip install --no-cache-dir "
                + " ".join(f"'{package}'" for package in RUNTIME_PACKAGES)
            ),
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"rm -rf {CODES_ROOT}; "
                # The CodeS repository carries the hundred public repositories
                # its training data was extracted from and the hundred more its
                # benchmark grades, which is well over a hundred megabytes of
                # material no run reads. A partial, sparse clone takes the
                # commits and trees but fetches blobs only for the directory
                # named below, which is a couple of megabytes and seconds
                # rather than minutes — once per trial, of which a benchmark
                # run has hundreds. Pinning is unaffected: the working tree is
                # narrower, the revision is exactly the one configured.
                f"git clone --quiet --filter=blob:none --sparse "
                f"{self.repository} {CODES_ROOT}; "
                f"git -C {CODES_ROOT} sparse-checkout set {CODES_SPARSE_PATH}; "
                f"git -C {CODES_ROOT} checkout --quiet {self.commit}"
            ),
            # The container has no credentials and nobody to ask for them. A
            # repository that is private, renamed, or misspelled otherwise
            # reaches Git as an authentication challenge, which it answers by
            # prompting: a trial that stalls until its own timeout rather than
            # reporting in seconds that it could not read the repository.
            env=GIT_NON_INTERACTIVE,
        )
        for module in (
                "run_codes.py",
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
                    "python -u /installed-agent/run_codes.py "
                    "2>&1 | tee /logs/agent/codes.log"
                ),
                env={
                    "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
                    "CODES_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
                },
                cwd=TASK_WORKSPACE,
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        populate_usage_context(context, self.logs_dir / USAGE_FILENAME)
