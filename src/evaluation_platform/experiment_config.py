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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from evaluation_platform.benchmark import BenchmarkSettings, ImageMirrorRule


DEFAULT_CONFIG_PATH = Path("configs/math-verify-self-collaboration.yaml")

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


@dataclass(frozen=True)
class AgentHyperparameters:
    """The hyperparameters one code generation solution accepts."""

    solution: str
    types: Mapping[str, type]
    defaults: Mapping[str, Any]

    def resolve(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Fill in defaults, validate, and reject names the runner would ignore.

        The result is the complete set of hyperparameters the run uses, so
        what is archived and what Harbor records is the effective setup rather
        than only the part that was written down.
        """
        unknown = set(values) - set(self.types)
        if unknown:
            raise ConfigurationError(
                f"Unknown {self.solution} hyperparameter(s): "
                f"{', '.join(sorted(unknown))}. Known hyperparameters: "
                f"{', '.join(sorted(self.types))}."
            )
        return dict(self.defaults) | {
            name: _coerce_hyperparameter(name, value, self.types[name])
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


# Self-Collaboration: an Analyst, a Coder, and a Tester that runs between Coder
# rounds. `test_command`, `reasoning_effort` and `request_extra` have no
# defaults. Without a test command the Tester phase does not run at all;
# without a reasoning setting the provider's own default applies. Both are
# choices a configuration has to make deliberately, and both are recorded
# either way.
SELF_COLLABORATION = AgentHyperparameters(
    solution="Self-Collaboration",
    types={
        "max_rounds": int,
        "analyst_steps": int,
        "coder_steps": int,
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
        "test_command": str,
        "reasoning_effort": str,
        "request_extra": dict,
    },
    defaults={
        "max_rounds": 3,
        "analyst_steps": 10,
        "coder_steps": 15,
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.95,
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
CODE_TEAM = AgentHyperparameters(
    solution="CodeTeam",
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
        # Sampling, and the budgets that bound a run the tool would otherwise
        # let run until Harbor's timeout.
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
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
        "max_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.95,
    },
)

AGENT_HYPERPARAMETERS: dict[str, AgentHyperparameters] = {
    "evaluation_platform.self_collaboration_agent": SELF_COLLABORATION,
    "evaluation_platform.code_team_agent": CODE_TEAM,
}

# The retrieval backends CodeTeam's RAG client implements. `faiss_hnsw` is the
# paper's own path and needs the optional retrieval stack; `lexical` needs
# nothing beyond the tool's own dependencies.
RAG_BACKENDS = ("faiss_hnsw", "lexical")

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
_MODEL_KEYS = {"provider", "name", "base_url", "api_key_env"}
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
            "model_name": self.model_label,
            "kwargs": dict(self.hyperparameters),
            "env": {
                self.api_key_env: "${" + self.api_key_env + "}",
                "API_KEY_ENV": self.api_key_env,
                "BASE_URL": self.base_url,
                "MODEL": self.model_name,
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
            "model": {
                "provider": self.model_provider,
                "name": self.model_name,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
            },
            "agent": agent,
        }


def load_experiment_config(
        path: Path,
        environment: Mapping[str, str],
) -> ExperimentConfig:
    """Read, expand, and validate the experiment configuration at `path`."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"No experiment configuration at {path}.") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{path} is not valid YAML: {error}") from error

    document = _expand_templates(document, environment, where=str(path))
    return _build_config(document, environment, source=str(path))


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


def _coerce_hyperparameter(name: str, value: Any, expected: type) -> Any:
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
    if name == "rag_backend" and value not in RAG_BACKENDS:
        raise ConfigurationError(
            f"Hyperparameter 'rag_backend' must be one of "
            f"{', '.join(RAG_BACKENDS)}, got {value!r}."
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
) -> ExperimentConfig:
    document = _require_mapping(document, "the configuration", source)
    _reject_unknown(document, _TOP_LEVEL_KEYS, "the configuration", source)

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
