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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from evaluation_platform.benchmark import BenchmarkSettings, ImageMirrorRule


DEFAULT_CONFIG_PATH = Path("configs/math-verify-self-collaboration.yaml")

# Hyperparameters of the Self-Collaboration roles. Harbor forwards these to the
# agent as `agents[].kwargs`, which it records in the job's `config.json`, and
# the agent hands them to the in-container runner.
HYPERPARAMETER_TYPES: dict[str, type] = {
    "max_rounds": int,
    "analyst_steps": int,
    "coder_steps": int,
    "max_tokens": int,
    "temperature": float,
    "top_p": float,
    "test_command": str,
    "reasoning_effort": str,
    "request_extra": dict,
}

# Values used for hyperparameters a configuration leaves out. `test_command`,
# `reasoning_effort` and `request_extra` have no defaults. Without a test
# command the Tester phase does not run at all; without a reasoning setting the
# provider's own default applies. Both are choices a configuration has to make
# deliberately, and both are recorded either way.
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "max_rounds": 3,
    "analyst_steps": 10,
    "coder_steps": 15,
    "max_tokens": 8192,
    "temperature": 0.0,
    "top_p": 0.95,
}

_TOP_LEVEL_KEYS = {
    "name",
    "description",
    "run",
    "benchmark",
    "task",
    "model",
    "agent",
}
_RUN_KEYS = {"jobs_dir", "n_attempts", "n_concurrent_trials", "artifacts"}
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


def resolve_hyperparameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in defaults, validate, and reject names the runner would ignore.

    The result is the complete set of hyperparameters the run uses, so what is
    archived and what Harbor records is the effective setup rather than only
    the part that was written down.
    """
    unknown = set(values) - set(HYPERPARAMETER_TYPES)
    if unknown:
        raise ConfigurationError(
            f"Unknown hyperparameter(s): {', '.join(sorted(unknown))}. "
            f"Known hyperparameters: {', '.join(sorted(HYPERPARAMETER_TYPES))}."
        )
    return DEFAULT_HYPERPARAMETERS | {
        name: _coerce_hyperparameter(name, value) for name, value in values.items()
    }


def _coerce_hyperparameter(name: str, value: Any) -> Any:
    expected = HYPERPARAMETER_TYPES[name]
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ConfigurationError(
            f"Hyperparameter {name!r} must be a {expected.__name__}, got {value!r}."
        )
    if expected is int and value < 1:
        raise ConfigurationError(f"Hyperparameter {name!r} must be at least 1.")
    if name == "test_command" and not value.strip():
        raise ConfigurationError(
            "Hyperparameter 'test_command' must be a command; remove the key to "
            "run the Coder without the Tester."
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

    hyperparameters = resolve_hyperparameters(
        _require_mapping(
            agent.get("hyperparameters", {}), "'agent.hyperparameters'", source
        )
    )

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
        agent_import_path=_require_str(agent, "import_path", "'agent'", source),
        agent_n_concurrent=agent.get("n_concurrent"),
        agent_repository=agent.get("repository"),
        agent_commit=agent.get("commit"),
        hyperparameters=hyperparameters,
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
