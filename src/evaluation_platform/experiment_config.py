"""Experiment setup configuration.

`configs/*.yaml` is the single home for the setup of a run: which task is
attempted, which model answers, which revision of the code generation tool is
used, and which hyperparameters its roles receive. Loading a file yields both
the Harbor job configuration that `main.py` launches and a resolved snapshot
that is archived next to the job, so the setup behind a recorded result stays
inspectable after the fact.
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from evaluation_platform.benchmark import BenchmarkSettings, ImageMirrorRule
from evaluation_platform.model_routing import ROUTING_ENV, ROUTING_KEYS


DEFAULT_CONFIG_PATH = Path("configs/math-verify-self-collaboration.yaml")

# The project root, as seen from this module inside `src/evaluation_platform/`.
_ROOT = Path(__file__).resolve().parents[2]

# The sampling parameters, and the one file that sets them.
#
# These three are deliberately not hyperparameters of any solution. They
# describe the model call rather than the method wrapped around it, in the way
# `model.routing` describes the endpoint rather than the method, and an arm that
# samples differently from another is not a comparison of scaffolds. So no arm
# below declares them, no arm defaults them, and a configuration that states one
# under `agent.hyperparameters` is rejected with a pointer to the file that
# does. `configs/generation.yaml` explains the values themselves.
GENERATION_PARAMETERS: Mapping[str, type] = {
    "temperature": float,
    "top_p": float,
    "max_tokens": int,
}
GENERATION_CONFIG_PATH = _ROOT / "configs" / "generation.yaml"

# Each code generation solution has its own roles and its own knobs, so the
# hyperparameters a configuration may state depend on which one it runs. A
# solution is identified by the module of `agent.import_path`, and a
# configuration naming an unknown one is rejected before Docker starts rather
# than having its hyperparameters silently ignored by an agent that does not
# know them.
#
# Harbor forwards the resolved values to the agent as `agents[].kwargs`, which
# it records in the job's `config.json`, and the agent hands them to the
# in-container runner.


# The configuration keys that pin a revision of an upstream tool. Every
# solution reads them; the two that have no upstream repository reject them.
UPSTREAM_REVISION_KEYS = ("repository", "commit")


@dataclass(frozen=True)
class AgentHyperparameters:
    """The hyperparameters one code generation solution accepts."""

    solution: str
    # The name Harbor records for this arm in a job's agent configuration.
    # Harbor's own results view builds a job's agent column from it and shows
    # nothing when it is unset, which is why it is stated rather than left to
    # the agent class alone: the class's `name()` reaches the *trial* record,
    # while this reaches the *job* record, and a comparison is read at both
    # levels.
    #
    # It must not be one of Harbor's built-in agent names. Where a name is,
    # Harbor's factory builds that built-in agent and never looks at
    # `import_path` — so a colliding label would silently run a different agent
    # than the one the configuration names. `tests/test_experiment_config.py`
    # holds that line.
    label: str
    types: Mapping[str, type]
    defaults: Mapping[str, Any]
    # The values a hyperparameter is restricted to, where the solution
    # implements a fixed set rather than a range. A name absent from here
    # takes any value of its type. This belongs to the solution rather than to
    # the coercion beside it because one name means different things to
    # different tools: `reasoning_effort` is a free string forwarded to an
    # OpenAI-compatible provider by three of the solutions here, and one of
    # eight literals accepted by a fourth.
    choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def resolve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Fill in defaults, validate, and reject names the runner would ignore.

        The result is the complete set of hyperparameters the run uses, so
        what is archived and what Harbor records is the effective setup rather
        than only the part that was written down.
        """
        sampling = sorted(set(values) & set(GENERATION_PARAMETERS))
        if sampling:
            raise ConfigurationError(
                f"{', '.join(sampling)} are set for every arm at once in "
                f"{GENERATION_CONFIG_PATH.name}, not per solution. Stating "
                f"{sampling[0]!r} here would sample {self.solution} differently "
                "from the arms it is compared with, which is a difference the "
                f"reward cannot show. Remove the key; edit "
                f"configs/{GENERATION_CONFIG_PATH.name} to change the value for "
                "the whole comparison."
            )
        unknown = set(values) - set(self.types)
        if unknown:
            raise ConfigurationError(
                f"Unknown {self.solution} hyperparameter(s): "
                f"{', '.join(sorted(unknown))}. Known hyperparameters: "
                f"{', '.join(sorted(self.types))}."
            )
        return dict(self.defaults) | {
            name: _coerce_hyperparameter(
                name, value, self.types[name], self.choices.get(name)
            )
            for name, value in values.items()
        }

    def reject_foreign(self, names: Iterable[str]) -> None:
        """Reject names that belong to a different solution's schema.

        Harbor builds an agent from `kwargs` that carry more than
        hyperparameters, so a name this solution does not know is not by
        itself an error. A name another integrated solution *does* know is:
        it was written for a different agent, and left in place it would be
        accepted here and then quietly do nothing.
        """
        foreign = sorted(
            name
            for name in names
            if name not in self.types
            and any(name in other.types for other in AGENT_HYPERPARAMETERS.values())
        )
        if foreign:
            raise ConfigurationError(
                f"{', '.join(foreign)} configure a different code generation "
                f"solution, not {self.solution}. Known {self.solution} "
                f"hyperparameters: {', '.join(sorted(self.types))}."
            )

    def reject_upstream_revision(self, names: Iterable[str], identity: str) -> None:
        """Reject a tool revision this solution has nothing to pin.

        Most entries here are somebody else's repository at a commit, and
        pinning it is what keeps the tool's version part of the recorded
        setup. Two are not: one is this platform's own prompt, the other is
        whatever Terminus ships in the installed Harbor. Harbor ignores agent
        `kwargs` an agent has no use for, so a `commit` written out of habit
        would be archived in the experiment's setup as though it had decided
        something.
        """
        named = sorted(name for name in UPSTREAM_REVISION_KEYS if name in names)
        if named:
            raise ConfigurationError(
                f"{self.solution} has no upstream tool to pin, so "
                f"'agent.{named[0]}' would be recorded and then ignored. What "
                f"identifies this run is {identity}."
            )


# Self-Collaboration: an Analyst, a Coder, and a Tester that runs between Coder
# rounds. `test_command`, `reasoning_effort` and `request_extra` have no
# defaults. Without a test command the Tester phase does not run at all;
# without a reasoning setting the provider's own default applies. Both are
# choices a configuration has to make deliberately, and both are recorded
# either way.
SELF_COLLABORATION = AgentHyperparameters(
    solution="Self-Collaboration",
    label="self-collaboration",
    types={
        "max_rounds": int,
        "analyst_steps": int,
        "coder_steps": int,
        "test_command": str,
        "reasoning_effort": str,
        "request_extra": dict,
    },
    defaults={
        "max_rounds": 3,
        "analyst_steps": 10,
        "coder_steps": 15,
    },
)

# CodeTeam: competing Architects propose software design sketches, a CTO
# selects one, Developers implement the files it names under a dependency-aware
# scheduler, and a QA agent tests and repairs the result. The defaults are the
# tool's own, so a configuration that states nothing runs the published setup.
#
# The three ablations of the paper are each one key here: `rag_enabled`,
# `dynamic_developer_allocation`, and `git_coordination`. The two budgets and
# the two seeds have no defaults: a budget left unset means the run is bounded
# only by Harbor's own timeout, and a seed left unset means the architect
# profiles and the fixed developer assignment are not pinned.
# The retrieval backends CodeTeam's RAG client implements. `faiss_hnsw` is the
# paper's own path and needs the optional retrieval stack; `lexical` needs
# nothing beyond the tool's own dependencies.
RAG_BACKENDS = ("faiss_hnsw", "lexical")

CODE_TEAM = AgentHyperparameters(
    solution="CodeTeam",
    label="code-team",
    types={
        # Planning.
        "architects": int,
        "architect_seed": int,
        "sds_retry": int,
        "preprocess_requirements": bool,
        # Implementation.
        "dynamic_developer_allocation": bool,
        "fixed_developer_agents": int,
        "developer_assignment_seed": int,
        "git_coordination": bool,
        # QA test-and-repair rounds after the first implementation round.
        "max_qa_rounds": int,
        # Retrieval grounding for the Architect stage.
        "rag_enabled": bool,
        "rag_backend": str,
        "rag_top_k": int,
        # The budgets that bound a run the tool would otherwise let run until
        # Harbor's timeout. Sampling is not here: see GENERATION_PARAMETERS.
        "reasoning_effort": str,
        "request_extra": dict,
        "max_wall_clock_seconds": int,
        "max_token_budget": int,
    },
    defaults={
        "architects": 4,
        "sds_retry": 1,
        "preprocess_requirements": True,
        "dynamic_developer_allocation": True,
        "fixed_developer_agents": 4,
        "git_coordination": True,
        "max_qa_rounds": 2,
        "rag_enabled": False,
        "rag_backend": "faiss_hnsw",
        "rag_top_k": 5,
    },
    choices={"rag_backend": RAG_BACKENDS},
)

# CodeS: a multi-layer sketch rather than a team. RepoSketcher proposes the
# file tree for the specification, FileSketcher writes each Python file as
# signatures with empty bodies, and SketchFiller implements one function per
# request from that file's sketch and the sketches it imports. There are no
# roles to size and no rounds to bound, so what a configuration states about
# CodeS is how a request is made rather than who makes it.
#
# The defaults are the tool's own, including the two it hardcodes in its
# driver: five attempts per request, the first at the run's temperature and the
# rest at `retry_temperature`. `concurrent_requests` defaults to 1, which is the
# published pipeline exactly — every request in sequence.
#
# Four hyperparameters have no default: the budgets bound a pipeline that bounds
# nothing itself, since the number of requests is the number of files and
# functions the model chose to propose. `retry_temperature` is the one sampling
# figure that stays here rather than moving to `configs/generation.yaml`,
# because it is not a setting the other arms have: it is the tool's own
# behaviour on a refused request, and the temperature every arm shares is the
# one the first attempt uses.
CODE_S = AgentHyperparameters(
    solution="CodeS",
    label="codes",
    types={
        # How a request is made, and what happens when one fails.
        "request_attempts": int,
        "retry_temperature": float,
        # How many of the requests within one phase are in flight at once. Both
        # fan-out phases are batches of independent requests, so this changes
        # how long a task takes and not what is asked.
        "concurrent_requests": int,
        # Sampling is not here: see GENERATION_PARAMETERS.
        "reasoning_effort": str,
        "request_extra": dict,
        # The budgets that bound a pipeline whose length the model chooses.
        "max_wall_clock_seconds": int,
        "max_token_budget": int,
    },
    defaults={
        "request_attempts": 5,
        "retry_temperature": 0.1,
        "concurrent_requests": 1,
    },
)

# Single-Shot: the baseline, and the only entry here that is not a published
# tool. One request carries the task's specification together with the
# instruction that says how to lay a repository out in a reply, and a
# deterministic writer puts the files that reply names into the workspace.
# There is no plan, no role, no test run, and no second look at the result.
#
# Every solution beside it claims to improve on prompting a model directly,
# and a comparison with no direct-prompting arm cannot say whether the
# scaffolding or the model is doing the work. `docs/baselines.md` is the
# longer form of that argument.
#
# The prompt is deliberately not a hyperparameter. It is fixed in the runner
# and recorded by digest with every run, because a control whose prompt is a
# knob has stopped being a control and become a fourth solution.
#
# This arm is the reason the shared output ceiling is set as high as it is. The
# others spend their budget one file or one function at a time; this one has to
# fit a whole repository into a single reply. A reply that reaches the ceiling
# is recorded as truncated, and counted as `output_limit_exhausted`, rather than
# graded silently — because a baseline that ran out of output tokens and one
# that did not know the answer are different findings.
SINGLE_SHOT = AgentHyperparameters(
    solution="Single-Shot",
    label="single-shot",
    types={
        "reasoning_effort": str,
        "request_extra": dict,
        # One request means one chance, so a provider hiccup would otherwise
        # cost the whole task. These attempts re-send the same request; they
        # are not a second look at the answer, and the run is still one reply.
        "request_attempts": int,
        # How long one attempt may wait for a reply. A request for tens of
        # thousands of tokens takes minutes to answer, and a stalled one that
        # is never abandoned takes the trial's whole timeout with it and
        # records nothing about why. Stated here so that what bounds the run is
        # part of its setup, and so that the attempts above stay the only
        # retries: the client makes none of its own.
        "request_timeout_seconds": int,
    },
    defaults={
        "request_attempts": 3,
        "request_timeout_seconds": 600,
    },
)

# Terminus 2: Harbor's own reference agent, and the second baseline. A model
# in a loop with a shell — one agent, no roles, no plan, no review — which is
# what makes it the more searching control of the two. The question a
# multi-agent solution has to answer is not whether it beats a single request,
# but whether its structure beats the same model iterating on its own.
#
# It is also the one entry that is not installed into the task container. The
# agent runs on the host, drives a tmux session inside the container, and
# reaches the provider through LiteLLM rather than through the OpenAI client
# the other solutions share, so what it spends is counted by Harbor rather
# than by `model_usage.py`. There is no upstream revision to pin either: what
# runs is the Terminus of whichever Harbor release this project is installed
# with, which the adapter records per run.
#
# `max_turns` has no default, for the reason CodeTeam's and CodeS's budgets have
# none: Terminus bounds itself at a million turns, which is no bound at all, and
# a run's cost should be a choice a configuration makes out loud.
#
# It is also the arm where the shared sampling parameters take two routes rather
# than one. Terminus accepts a `temperature` of its own and has no parameter for
# the other two, so the adapter passes `top_p` and `max_tokens` through
# `llm_kwargs`, which Harbor's LiteLLM wrapper spreads into the body of every
# request. Same three values, same requests; each run's `resolved-setup.json`
# says which route each one took.
TERMINUS_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "default",
)
TERMINUS_PARSERS = ("json", "xml")

TERMINUS = AgentHyperparameters(
    solution="Terminus 2",
    # Not plain `terminus-2`: that is one of Harbor's own agent names, and a
    # job whose agent is named it gets Harbor's built-in Terminus rather than
    # the adapter that points it at this experiment's endpoint. The suffix is
    # what keeps the configuration's `import_path` in charge. What ran is still
    # Harbor's Terminus 2, and each run's resolved-setup.json says which.
    label="terminus-2-baseline",
    types={
        "max_turns": int,
        "reasoning_effort": str,
        # Merged into the request body verbatim, exactly as it is for the three
        # solutions that reach the provider through the OpenAI client. Terminus
        # goes through LiteLLM, which carries this through to the same place,
        # so a provider-level setting such as OpenRouter's `reasoning` object is
        # stated once and means the same thing in every arm of the comparison.
        "request_extra": dict,
        # How the agent is asked to phrase a tool call. Terminus ships both and
        # defaults to JSON.
        "parser_name": str,
        # Whether the agent compresses its own history when the context fills.
        # On by default, and worth leaving on: a task that writes a repository
        # is long, and the alternative to summarizing is failing at the ceiling.
        "enable_summarize": bool,
    },
    defaults={
        "parser_name": "json",
        "enable_summarize": True,
    },
    choices={
        "reasoning_effort": TERMINUS_REASONING_EFFORTS,
        "parser_name": TERMINUS_PARSERS,
    },
)

AGENT_HYPERPARAMETERS: dict[str, AgentHyperparameters] = {
    "evaluation_platform.self_collaboration_agent": SELF_COLLABORATION,
    "evaluation_platform.code_team_agent": CODE_TEAM,
    "evaluation_platform.codes_agent": CODE_S,
    "evaluation_platform.single_shot_agent": SINGLE_SHOT,
    "evaluation_platform.terminus_agent": TERMINUS,
}

# Integer hyperparameters that are not counts. Every other one bounds a number
# of rounds, steps, agents, or tokens, where zero means the phase does not
# happen and a configuration should say so by leaving the key out. A seed
# names a draw rather than bounding one, and zero is as good a seed as any.
_SEEDS = ("architect_seed", "developer_assignment_seed")

# Harbor's own default when a configuration does not set one. Named here so
# the launcher reports the concurrency that will actually be used rather than
# staying silent about a figure nobody wrote down.
DEFAULT_N_CONCURRENT_TRIALS = 4

_TOP_LEVEL_KEYS = {
    "name",
    "description",
    "run",
    "benchmark",
    "task",
    "model",
    "agent",
    # Only an archived setup states this; see `load_experiment_config`.
    "generation",
}
# The `run` block is Harbor's own job configuration, narrowed to the keys this
# project has a use for. Beyond where results land, it is what makes a
# benchmark-scale run finish: `n_concurrent_trials` is how many tasks execute
# at once, `retry` is what a transient provider or network failure costs, and
# `environment` is how a trial's declared resources are adjusted to the
# machine actually running it.
_RUN_KEYS = {
    "jobs_dir",
    "n_attempts",
    "n_concurrent_trials",
    "artifacts",
    "retry",
    "environment",
    "timeout_multiplier",
    "agent_timeout_multiplier",
    "verifier_timeout_multiplier",
    "quiet",
}
_RETRY_KEYS = {
    "max_retries",
    "include_exceptions",
    "exclude_exceptions",
    "wait_multiplier",
    "min_wait_sec",
    "max_wait_sec",
}
_ENVIRONMENT_KEYS = {
    "force_build",
    "delete",
    "cpu_enforcement_policy",
    "memory_enforcement_policy",
    "override_cpus",
    "override_memory_mb",
    "override_storage_mb",
}
_TASK_KEYS = {"path"}
_BENCHMARK_KEYS = {
    "dataset",
    "ref",
    "task_names",
    "exclude_task_names",
    "n_tasks",
    "image_mirror",
}
_MIRROR_KEYS = {"expects", "pull_from"}
_MODEL_KEYS = {"provider", "name", "base_url", "api_key_env", "routing"}
# Routing keys whose value is a list of endpoint or provider names.
_ROUTING_NAME_LISTS = ("order", "only", "ignore", "quantizations")
_AGENT_KEYS = {
    "import_path",
    "n_concurrent",
    "repository",
    "commit",
    "hyperparameters",
}

_TEMPLATE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigurationError(ValueError):
    """An experiment configuration is missing, invalid, or contradictory."""


@dataclass(frozen=True)
class GenerationConfig:
    """The sampling parameters this run used, and where they came from.

    One of these per run, shared by every arm. `source` is recorded rather than
    assumed: a run launched from `configs/` reads the central file, and a run
    resumed from an archived snapshot reads the values pinned in that snapshot,
    which is what keeps a resumed job sampling the way its first half did.
    """

    temperature: float
    top_p: float
    max_tokens: int
    source: str

    def as_kwargs(self) -> dict[str, Any]:
        """The three values, under the names every request builder uses."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def to_snapshot(self) -> dict[str, Any]:
        """The block an archived setup carries, so a resume pins these values."""
        return self.as_kwargs()


def load_generation_config(
        path: Path | None = None,
        stated: Mapping[str, Any] | None = None,
) -> GenerationConfig:
    """The run's sampling parameters, from the one file that sets them.

    `stated` is the block an archived snapshot carries. It is read in place of
    the file so that resuming a job continues it at the sampling it started
    with, rather than at whatever the central file says today.
    """
    if stated is not None:
        return _build_generation(stated, source="the archived setup")

    path = path or GENERATION_CONFIG_PATH
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"No sampling configuration at {path}. It is the one place "
            f"{', '.join(GENERATION_PARAMETERS)} are set, and every arm reads "
            "it."
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{path} is not valid YAML: {error}") from error
    return _build_generation(
        _require_mapping(document, "the sampling configuration", str(path)),
        source=str(path),
    )


def _build_generation(
        document: Mapping[str, Any],
        source: str,
) -> GenerationConfig:
    _reject_unknown(
        document, set(GENERATION_PARAMETERS), "the sampling configuration", source
    )
    missing = sorted(set(GENERATION_PARAMETERS) - set(document))
    if missing:
        raise ConfigurationError(
            f"The sampling configuration in {source} is missing "
            f"{', '.join(missing)}. All of {', '.join(GENERATION_PARAMETERS)} "
            "are stated there, so that what a run sampled at is a value that "
            "was written down rather than a library's default."
        )
    values = {
        name: _coerce_hyperparameter(name, document[name], expected)
        for name, expected in GENERATION_PARAMETERS.items()
    }
    return GenerationConfig(**values, source=source)


def resolve_generation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Take the shared sampling values out of an agent's kwargs.

    `to_harbor_config` puts them there beside the solution's own
    hyperparameters, so an agent built by a job receives them and this removes
    them before Harbor's own constructor sees names it has no use for.

    An agent built by hand — in a test, or from a Python prompt — receives none
    of them, and falls back to the same central file the launcher reads. That
    keeps a directly constructed agent sampling the way a launched one does,
    with the values still coming from exactly one place. Some but not all three
    is the one case that is refused: they travel together, and two out of three
    means a run sampling at a value nobody chose.
    """
    stated = {
        name: kwargs.pop(name)
        for name in list(GENERATION_PARAMETERS)
        if name in kwargs
    }
    if not stated:
        return load_generation_config().as_kwargs()
    missing = sorted(set(GENERATION_PARAMETERS) - set(stated))
    if missing:
        raise ConfigurationError(
            f"The sampling parameters reach an agent together, and "
            f"{', '.join(missing)} did not arrive. Pass all of "
            f"{', '.join(GENERATION_PARAMETERS)} or none of them, in which case "
            f"configs/{GENERATION_CONFIG_PATH.name} is read."
        )
    return _build_generation(stated, source="the agent's kwargs").as_kwargs()


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully resolved experiment setup, ready to launch and to archive."""

    name: str
    description: str
    run: dict[str, Any]
    benchmark: BenchmarkSettings | None
    task_path: str | None
    model_provider: str
    model_name: str
    base_url: str
    api_key_env: str
    # Which of the provider's servers may answer. Beside the model rather than
    # in any solution's hyperparameters, because it is not a property of a
    # method: an aggregator serves one model from many providers at different
    # quantizations, and an arm running fp4 where another runs fp8 is not the
    # same experiment. `None` leaves the choice to the aggregator, which makes
    # it by price and does not record what it chose.
    model_routing: dict[str, Any] | None
    # The sampling parameters, shared by every arm and set in one file. Beside
    # the model rather than in any solution's hyperparameters, for the reason
    # routing is: how a model draws its tokens is not a property of the method
    # wrapped around it.
    generation: GenerationConfig
    agent_import_path: str
    agent_n_concurrent: int | None
    agent_repository: str | None
    agent_commit: str | None
    hyperparameters: dict[str, Any]

    @property
    def model_label(self) -> str:
        """The provider-qualified model identity Harbor reports per run."""
        return f"{self.model_provider}/{self.model_name}"

    def with_pinned_benchmark(self, resolved_ref: str | None) -> "ExperimentConfig":
        """Pin the benchmark to the digest its reference resolved to.

        A configuration may follow a floating reference. Pinning the resolved
        digest before the job starts keeps every trial of the run — and its
        archived setup — on one version of the benchmark.
        """
        if self.benchmark is None or resolved_ref is None:
            return self
        return replace(self, benchmark=replace(self.benchmark, ref=resolved_ref))

    def to_harbor_config(self, job_name: str) -> dict[str, Any]:
        agent: dict[str, Any] = {
            "import_path": self.agent_import_path,
            # Harbor's results view reads a job's agent column from this and
            # leaves it blank when it is unset, even though every trial beneath
            # the job records the agent it ran.
            "name": agent_hyperparameters(self.agent_import_path).label,
            "model_name": self.model_label,
            # The solution's own hyperparameters, plus the three sampling
            # values every arm shares. Both reach the agent the same way and
            # both are recorded in the job's `config.json`, so what a run
            # sampled at is on file beside what it was configured to do.
            "kwargs": {**self.hyperparameters, **self.generation.as_kwargs()},
            "env": {
                self.api_key_env: "${" + self.api_key_env + "}",
                "API_KEY_ENV": self.api_key_env,
                "BASE_URL": self.base_url,
                "MODEL": self.model_name,
                # One directive, delivered the same way to every arm, whether
                # the solution runs in the task container or on the host.
                ROUTING_ENV: json.dumps(self.model_routing or {}),
            },
        }
        if self.agent_n_concurrent is not None:
            agent["n_concurrent"] = self.agent_n_concurrent
        if self.agent_repository is not None:
            agent["kwargs"]["repository"] = self.agent_repository
        if self.agent_commit is not None:
            agent["kwargs"]["commit"] = self.agent_commit
        harbor_config: dict[str, Any] = {
            "job_name": job_name,
            **self.run,
            "agents": [agent],
        }
        if self.benchmark is not None:
            harbor_config["datasets"] = [self.benchmark.to_harbor_dataset()]
        else:
            harbor_config["tasks"] = [{"path": self.task_path}]
        return harbor_config

    def to_snapshot(self) -> dict[str, Any]:
        """The resolved setup in the shape of its configuration file.

        Templates are expanded and omitted sections filled in, so the snapshot
        records the setup that ran rather than the one that was written. It
        names the credential's environment variable but never its value.
        """
        agent: dict[str, Any] = {"import_path": self.agent_import_path}
        if self.agent_n_concurrent is not None:
            agent["n_concurrent"] = self.agent_n_concurrent
        if self.agent_repository is not None:
            agent["repository"] = self.agent_repository
        if self.agent_commit is not None:
            agent["commit"] = self.agent_commit
        agent["hyperparameters"] = dict(self.hyperparameters)
        source = (
            {"benchmark": self.benchmark.to_snapshot()}
            if self.benchmark is not None
            else {"task": {"path": self.task_path}}
        )
        return {
            "name": self.name,
            "description": self.description,
            "run": dict(self.run),
            **source,
            # Pinned into the snapshot, in the way the benchmark's digest is:
            # a job resumed next week continues at the sampling it started
            # with, whatever the central file says by then.
            "generation": self.generation.to_snapshot(),
            "model": {
                "provider": self.model_provider,
                "name": self.model_name,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                **(
                    {"routing": dict(self.model_routing)}
                    if self.model_routing
                    else {}
                ),
            },
            "agent": agent,
        }


def load_experiment_config(
        path: Path,
        environment: Mapping[str, str],
        archived: bool = False,
) -> ExperimentConfig:
    """Read, expand, and validate the experiment configuration at `path`.

    `archived` reads a snapshot this platform wrote rather than a configuration
    somebody wrote. The difference is the `generation` block: a snapshot carries
    the sampling the job ran at and is resumed on it, while a hand-written
    configuration has no say in sampling at all and is told so.
    """
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"No experiment configuration at {path}.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{path} is not valid YAML: {error}") from error

    document = _expand_templates(document, environment, where=str(path))
    return _build_config(document, environment, source=str(path), archived=archived)


def agent_hyperparameters(import_path: str) -> AgentHyperparameters:
    """The hyperparameter schema of the solution `import_path` names.

    The module is the identity: `agent.import_path` also carries the class,
    which the tests vary, and every class in one module drives one solution.
    """
    module = import_path.split(":", 1)[0]
    try:
        return AGENT_HYPERPARAMETERS[module]
    except KeyError:
        raise ConfigurationError(
            f"No code generation solution is integrated as {module!r}. "
            f"Known agent modules: {', '.join(sorted(AGENT_HYPERPARAMETERS))}."
        ) from None


def _coerce_hyperparameter(
        name: str,
        value: Any,
        expected: type,
        choices: tuple[str, ...] | None = None,
) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigurationError(
                f"Hyperparameter {name!r} must be true or false, got {value!r}."
            )
        return value
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ConfigurationError(
            f"Hyperparameter {name!r} must be a {expected.__name__}, got {value!r}."
        )
    if expected is int and name not in _SEEDS and value < 1:
        raise ConfigurationError(f"Hyperparameter {name!r} must be at least 1.")
    if name == "test_command" and not value.strip():
        raise ConfigurationError(
            "Hyperparameter 'test_command' must be a command; remove the key to "
            "run the Coder without the Tester."
        )
    if choices is not None and value not in choices:
        raise ConfigurationError(
            f"Hyperparameter {name!r} must be one of "
            f"{', '.join(choices)}, got {value!r}."
        )
    if expected is dict:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Hyperparameter {name!r} must be JSON-serializable, since it "
                f"is sent to the model provider verbatim: {error}"
            ) from error
    return value


def _build_config(
        document: Any,
        environment: Mapping[str, str],
        source: str,
        archived: bool = False,
) -> ExperimentConfig:
    document = _require_mapping(document, "the configuration", source)
    _reject_unknown(document, _TOP_LEVEL_KEYS, "the configuration", source)

    stated_generation = document.get("generation")
    if stated_generation is not None and not archived:
        raise ConfigurationError(
            f"{source} states a 'generation' block. "
            f"{', '.join(GENERATION_PARAMETERS)} are set once for every arm in "
            f"configs/{GENERATION_CONFIG_PATH.name}, so that a difference "
            "between two runs is a difference between the methods. Remove the "
            "block; the value archived beside a job is a record of what ran, "
            "not a second place to set it."
        )
    generation = load_generation_config(
        stated=_require_mapping(stated_generation, "'generation'", source)
        if stated_generation is not None
        else None
    )

    run = _require_mapping(document.get("run", {}), "'run'", source)
    _reject_unknown(run, _RUN_KEYS, "'run'", source)
    _reject_unknown(
        _require_mapping(run.get("retry", {}), "'run.retry'", source),
        _RETRY_KEYS,
        "'run.retry'",
        source,
    )
    _reject_unknown(
        _require_mapping(run.get("environment", {}), "'run.environment'", source),
        _ENVIRONMENT_KEYS,
        "'run.environment'",
        source,
    )
    benchmark, task_path = _build_source(document, source)
    model = _require_mapping(document.get("model"), "'model'", source)
    _reject_unknown(model, _MODEL_KEYS, "'model'", source)
    agent = _require_mapping(document.get("agent"), "'agent'", source)
    _reject_unknown(agent, _AGENT_KEYS, "'agent'", source)

    api_key_env = _require_str(model, "api_key_env", "'model'", source)
    _require_credential(api_key_env, environment, source)
    provider = _require_str(model, "provider", "'model'", source)
    base_url = _require_str(model, "base_url", "'model'", source)
    _validate_endpoint(provider, base_url, source)

    import_path = _require_str(agent, "import_path", "'agent'", source)
    hyperparameters = agent_hyperparameters(import_path).resolve(
        _require_mapping(
            agent.get("hyperparameters", {}), "'agent.hyperparameters'", source
        )
    )
    _validate_concurrency(run, agent, source)

    return ExperimentConfig(
        name=_require_str(document, "name", "the configuration", source),
        description=str(document.get("description", "")).strip(),
        run=dict(run),
        benchmark=benchmark,
        task_path=task_path,
        model_provider=provider,
        model_name=_require_str(model, "name", "'model'", source),
        base_url=base_url,
        api_key_env=api_key_env,
        model_routing=_build_routing(model, source),
        generation=generation,
        agent_import_path=import_path,
        agent_n_concurrent=agent.get("n_concurrent"),
        agent_repository=agent.get("repository"),
        agent_commit=agent.get("commit"),
        hyperparameters=hyperparameters,
    )


def _validate_concurrency(
        run: Mapping[str, Any],
        agent: Mapping[str, Any],
        source: str,
) -> None:
    """Check the two concurrency limits against each other before launching.

    Harbor runs at most `run.n_concurrent_trials` trials at once, and
    `agent.n_concurrent` is a sub-limit within that on how many of them may be
    calling the model at the same time — the knob for a provider's rate limit,
    as distinct from the machine's capacity. Harbor rejects a sub-limit above
    the total, and raising one without the other is the natural mistake when
    scaling a configuration up, so it is worth catching here with the reason
    attached rather than as a validation error once the run has started.
    """
    n_concurrent_trials = run.get("n_concurrent_trials")
    if n_concurrent_trials is not None and (
            isinstance(n_concurrent_trials, bool)
            or not isinstance(n_concurrent_trials, int)
            or n_concurrent_trials < 1
    ):
        raise ConfigurationError(
            f"'run.n_concurrent_trials' in {source} must be a positive integer."
        )

    n_concurrent = agent.get("n_concurrent")
    if n_concurrent is None:
        return
    if isinstance(n_concurrent, bool) or not isinstance(n_concurrent, int) or n_concurrent < 1:
        raise ConfigurationError(
            f"'agent.n_concurrent' in {source} must be a positive integer."
        )

    effective_trials = (
        n_concurrent_trials
        if n_concurrent_trials is not None
        else DEFAULT_N_CONCURRENT_TRIALS
    )
    if n_concurrent > effective_trials:
        raise ConfigurationError(
            f"'agent.n_concurrent' ({n_concurrent}) in {source} exceeds "
            f"'run.n_concurrent_trials' ({effective_trials}); it is a limit on "
            "how many of the concurrent trials may call the model at once, so "
            "it can never be the larger of the two. Raise "
            "'run.n_concurrent_trials' to run more tasks in parallel, and "
            "lower 'agent.n_concurrent' only to stay under a rate limit."
        )


def _build_source(
        document: Mapping[str, Any],
        source: str,
) -> tuple[BenchmarkSettings | None, str | None]:
    """Read what a run is executed against: a benchmark, or one local task."""
    if ("benchmark" in document) == ("task" in document):
        raise ConfigurationError(
            f"{source} must state exactly one of 'benchmark' (a dataset from "
            "Harbor's registry) and 'task' (a single task directory)."
        )

    if "task" in document:
        task = _require_mapping(document.get("task"), "'task'", source)
        _reject_unknown(task, _TASK_KEYS, "'task'", source)
        return None, _require_str(task, "path", "'task'", source)

    benchmark = _require_mapping(document.get("benchmark"), "'benchmark'", source)
    _reject_unknown(benchmark, _BENCHMARK_KEYS, "'benchmark'", source)
    dataset = _require_str(benchmark, "dataset", "'benchmark'", source)
    if "/" not in dataset:
        raise ConfigurationError(
            f"'benchmark.dataset' in {source} must name a registry dataset as "
            f"'org/name', got {dataset!r}."
        )
    n_tasks = benchmark.get("n_tasks")
    if n_tasks is not None and (not isinstance(n_tasks, int) or n_tasks < 1):
        raise ConfigurationError(
            f"'benchmark.n_tasks' in {source} must be a positive integer."
        )

    return (
        BenchmarkSettings(
            dataset=dataset,
            ref=benchmark.get("ref"),
            task_names=_require_str_list(benchmark, "task_names", source),
            exclude_task_names=_require_str_list(
                benchmark, "exclude_task_names", source
            ),
            n_tasks=n_tasks,
            image_mirror=_build_mirror_rules(benchmark, source),
        ),
        None,
    )


def _build_mirror_rules(
        benchmark: Mapping[str, Any],
        source: str,
) -> list[ImageMirrorRule]:
    rules = benchmark.get("image_mirror", [])
    if not isinstance(rules, list):
        raise ConfigurationError(
            f"'benchmark.image_mirror' in {source} must be a list of rules."
        )
    resolved = []
    for rule in rules:
        rule = _require_mapping(rule, "'benchmark.image_mirror'", source)
        _reject_unknown(rule, _MIRROR_KEYS, "'benchmark.image_mirror'", source)
        resolved.append(
            ImageMirrorRule(
                expects=_require_str(
                    rule, "expects", "'benchmark.image_mirror'", source
                ),
                pull_from=_require_str(
                    rule, "pull_from", "'benchmark.image_mirror'", source
                ),
            )
        )
    return resolved


def _require_str_list(
        section: Mapping[str, Any],
        key: str,
        source: str,
) -> list[str]:
    values = section.get(key, [])
    if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
    ):
        raise ConfigurationError(
            f"'benchmark.{key}' in {source} must be a list of strings."
        )
    return list(values)


def _build_routing(
        model: Mapping[str, Any],
        source: str,
) -> dict[str, Any] | None:
    """Read and check the provider routing a configuration pins.

    A misspelled key is the failure worth catching here: the API accepts an
    unknown routing field and ignores it, so the run is unpinned and every
    record still says it was pinned.
    """
    routing = model.get("routing")
    if routing is None:
        return None
    routing = _require_mapping(routing, "'model.routing'", source)
    if not routing:
        raise ConfigurationError(
            f"'model.routing' in {source} is empty. Remove the key to leave "
            "the choice of server to the provider, or name the endpoints the "
            "run may use."
        )

    unknown = set(routing) - ROUTING_KEYS
    if unknown:
        raise ConfigurationError(
            f"Unknown key(s) in 'model.routing' of {source}: "
            f"{', '.join(sorted(unknown))}. The provider accepts an unknown "
            "routing field and ignores it, which would leave the run routed by "
            f"price while its record said otherwise. Known keys: "
            f"{', '.join(sorted(ROUTING_KEYS))}."
        )

    for key in _ROUTING_NAME_LISTS:
        if key in routing and (
                not isinstance(routing[key], list)
                or not routing[key]
                or not all(isinstance(name, str) and name for name in routing[key])
        ):
            raise ConfigurationError(
                f"'model.routing.{key}' in {source} must be a non-empty list "
                "of endpoint names."
            )

    if "allow_fallbacks" in routing and not isinstance(
            routing["allow_fallbacks"], bool
    ):
        raise ConfigurationError(
            f"'model.routing.allow_fallbacks' in {source} must be true or false."
        )

    return dict(routing)


def _validate_endpoint(provider: str, base_url: str, source: str) -> None:
    if provider.lower() == "deepseek" and base_url.rstrip("/").endswith("/api/v1"):
        raise ConfigurationError(
            f"{source} points DeepSeek at '/api/v1', which it does not serve. "
            "Use https://api.deepseek.com (or https://api.deepseek.com/v1)."
        )


def _require_credential(
        api_key_env: str,
        environment: Mapping[str, str],
        source: str,
) -> None:
    if not environment.get(api_key_env, "").strip():
        raise ConfigurationError(
            f"{source} reads the API credential from {api_key_env}, which is "
            "unset. Set it in .env before launching."
        )


def _expand_templates(value: Any, environment: Mapping[str, str], where: str) -> Any:
    """Expand `${VAR}` and `${VAR:-default}` throughout the document."""
    if isinstance(value, dict):
        return {
            key: _expand_templates(item, environment, where)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_templates(item, environment, where) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = environment.get(name) or default
        if resolved is None:
            raise ConfigurationError(
                f"{where} references ${{{name}}}, which is not set and has no "
                "default. Set it in .env or write the value into the "
                "configuration."
            )
        return resolved

    return _TEMPLATE_PATTERN.sub(replace, value)


def _require_mapping(value: Any, where: str, source: str) -> dict[str, Any]:
    if value is None:
        raise ConfigurationError(f"{source} is missing {where}.")
    if not isinstance(value, dict):
        raise ConfigurationError(f"{where} in {source} must be a mapping.")
    return value


def _reject_unknown(
        section: Mapping[str, Any],
        allowed: set[str],
        where: str,
        source: str,
) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown key(s) in {where} of {source}: "
            f"{', '.join(sorted(unknown))}. Known keys: "
            f"{', '.join(sorted(allowed))}."
        )


def _require_str(
        section: Mapping[str, Any],
        key: str,
        where: str,
        source: str,
) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{where} in {source} needs a non-empty {key!r}.")
    return value
