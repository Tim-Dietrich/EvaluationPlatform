import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# The runner is uploaded next to its own dependencies, so its directory carries
# the shared usage accounting under a flat name. On the host, where the tests
# import this module from the package, that same directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_categories import FailureTally  # noqa: E402
from model_usage import UsageTotals, record_response_usage  # noqa: E402
from model_routing import (  # noqa: E402
    ServedProviders,
    install_client_routing,
    routing_from_env,
)

SELF_COLLABORATION_ROOT = Path("/installed-agent/self-collaboration")
WORKSPACE = Path("/workspace")
HISTORY_PATH = Path("/logs/agent/session-history.json")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")
# Directories a Python build, test run, or virtual environment leaves behind in
# the workspace. None of them is part of the generated project.
BUILD_OUTPUT = (
    "__pycache__/",
    "*.py[cod]",
    "*.egg-info/",
    ".eggs/",
    "build/",
    "dist/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "venv/",
)


def main() -> None:
    hyperparameters = _read_hyperparameters()
    instruction = _read_instruction()
    sys.path.insert(0, str(SELF_COLLABORATION_ROOT))

    agent_module = importlib.import_module("core.agent")
    config_module = importlib.import_module("core.config")
    repo_tools_module = importlib.import_module("core.repo_tools")

    # Pin which of the provider's servers may answer, before any request is
    # made. The wrapper sits on the OpenAI client rather than on this runner's
    # own call sites, so it reaches the requests the tool builds for itself as
    # well; what changes is the server, not the messages, the sampling, or the
    # model. The same seam reads back which server actually answered.
    routing = routing_from_env()
    observed = ServedProviders()
    install_client_routing(routing, observed)

    model_config = config_module.ModelConfig(
        **_model_settings(hyperparameters)
    )
    failures = FailureTally()
    # Written before the session starts, so a run that dies in its first
    # round still says what it was, and again after it, once the servers
    # that answered and the limits that were hit are known.
    _record_resolved_setup(
        hyperparameters, model_config, routing, observed, failures
    )

    usage_totals = UsageTotals()
    agent_module.call_llm_with_tools = _usage_recording_call(
        agent_module.call_llm_with_tools,
        usage_totals,
        model=model_config.model,
        failures=failures,
    )

    _initialize_repository()
    structure = repo_tools_module.get_repo_structure(str(WORKSPACE))
    task = f"""## Repository Structure
```
{structure}
```

## Task
{instruction}
"""
    session = agent_module.SelfCollabSession(
        config=model_config,
        repo_path=str(WORKSPACE),
        max_round=hyperparameters["max_rounds"],
        analyst_steps=hyperparameters["analyst_steps"],
        coder_steps=hyperparameters["coder_steps"],
        verbose=True,
    )
    try:
        # Without a test command the session skips its Tester phase entirely
        # and degrades to Analyst followed by a single Coder pass.
        history, analyst_result, coder_result = session.run(
            task,
            test_cmd=hyperparameters.get("test_command"),
        )
    finally:
        USAGE_PATH.write_text(
            json.dumps(usage_totals.to_dict(), indent=2),
            encoding="utf-8",
        )
        _record_resolved_setup(
            hyperparameters, model_config, routing, observed, failures
        )
    HISTORY_PATH.write_text(
        json.dumps(
            {
                "history": history,
                "analyst_result": analyst_result,
                "coder_result": coder_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _usage_recording_call(
        call: Callable[..., Any],
        totals: UsageTotals,
        model: str = "unknown",
        failures: FailureTally | None = None,
        sleep: Callable[[float], None] = time.sleep,
) -> Callable[..., Any]:
    """The tool's own model call, with what it spent and how it failed counted.

    The tool has one call site for its Analyst, its Coder and its Tester alike,
    so wrapping it once covers every role. Neither the retrying below nor the
    tool's own is changed by the counting: a classified failure is counted and
    then handled exactly as it was before.
    """
    failures = failures if failures is not None else FailureTally()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(3):
            try:
                response = call(*args, **kwargs)
                record_response_usage(response, totals)
                _record_finish_reasons(response, failures)
                return response
            except RuntimeError as error:
                failures.record_error(error)
                if str(error) != "Failed to call LLM API with tools":
                    raise
                if attempt == 2:
                    raise RuntimeError(
                        f"The LLM provider rate-limited model {model!r} after "
                        "9 requests. Retry later or select a model with available "
                        "capacity."
                    ) from error
                sleep(15 * (attempt + 1))
            except Exception as error:  # noqa: BLE001 - counted, then re-raised.
                failures.record_error(error)
                raise

        raise AssertionError("unreachable")

    return wrapped


def _record_finish_reasons(response: Any, failures: FailureTally) -> None:
    """Count a reply that stopped at the output ceiling rather than finishing.

    A truncated tool call is the shape this takes here: the Coder's `edit_file`
    arrives half-written, the tool cannot parse it, and the round is spent. That
    is a different finding from a Coder that made a wrong edit, and only this
    tells them apart.
    """
    for choice in getattr(response, "choices", None) or ():
        failures.record_finish_reason(getattr(choice, "finish_reason", None))


def _read_instruction() -> str:
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(
        encoding="utf-8"
    )


def _model_settings(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Model settings for this run, as the tool's own config accepts them.

    Reasoning settings are passed only when configured. The revision under
    evaluation is the authors' own and declares no reasoning field, so a
    configuration that asks for reasoning fails the run at startup rather than
    having the setting silently dropped, and a configuration that says nothing
    about it runs against that revision unmodified.
    """
    settings: dict[str, Any] = {
        "max_tokens": hyperparameters["max_tokens"],
        "temperature": hyperparameters["temperature"],
        "top_p": hyperparameters["top_p"],
    }
    if hyperparameters.get("reasoning_effort"):
        settings["reasoning_effort"] = hyperparameters["reasoning_effort"]
    if hyperparameters.get("request_extra"):
        settings["extra_body"] = hyperparameters["request_extra"]
    return settings


def _read_hyperparameters() -> dict[str, Any]:
    return json.loads(
        Path(os.environ["SELF_COLLABORATION_HYPERPARAMETERS_PATH"]).read_text(
            encoding="utf-8"
        )
    )


def _record_resolved_setup(
        hyperparameters: dict[str, Any],
        model_config: Any,
        routing: dict[str, Any] | None,
        observed: ServedProviders,
        failures: FailureTally | None = None,
) -> None:
    """Record what actually ran, next to the run's other logs.

    The experiment configuration states the intended setup; this states the
    observed one, including the commit of the code generation tool that the
    container ended up with and the model the environment resolved to.
    """
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "self_collaboration_commit": _installed_commit(),
                "model": model_config.model,
                "base_url": model_config.base_url,
                # What the run asked of the aggregator, and which server
                # answered. A model name does not name a server.
                "routing": routing,
                "providers_served": observed.names(),
                "hyperparameters": hyperparameters,
                # The sampling this run used, read off the config object the
                # tool's own requests were built from rather than off the file
                # that supplied it. `top_p` is stated as inert because this
                # revision declares it and sends it only from its non-agentic
                # entry point: the Analyst and the Coder sample at the
                # provider's default for it whatever was configured.
                "generation": {
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "max_tokens": model_config.max_tokens,
                    "source": "configs/generation.yaml",
                    "applied_via": "core.config.ModelConfig, one per role",
                    "inert_on_this_revision": ["top_p"],
                },
                # Why a round ended where it did, in categories that stay apart
                # from each other, from a timeout, and from a failing test.
                "failures": (failures or FailureTally()).to_dict(),
                "tester_enabled": bool(hyperparameters.get("test_command")),
                "reasoning_requested": bool(
                    hyperparameters.get("reasoning_effort")
                    or hyperparameters.get("request_extra")
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _installed_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SELF_COLLABORATION_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _initialize_repository() -> None:
    if (WORKSPACE / ".git").exists():
        return
    subprocess.run(["git", "init", "--quiet"], cwd=WORKSPACE, check=True)
    subprocess.run(
        ["git", "config", "user.email", "harbor@example.invalid"],
        cwd=WORKSPACE,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Harbor"], cwd=WORKSPACE, check=True
    )
    _exclude_build_output()
    subprocess.run(
        ["git", "commit", "--allow-empty", "--quiet", "-m", "Initial workspace"],
        cwd=WORKSPACE,
        check=True,
    )


def _exclude_build_output() -> None:
    """Keep build output out of the repository view the tool reads.

    The tool decides whether the Coder produced anything, and what to show it
    of its previous attempt, by reading `git diff`, which sees nothing of a
    project written from scratch because every file of it is untracked. The
    experiment's test command makes those files visible with
    `git add --intent-to-add`; left alone it would sweep up everything `pip
    install` and `pytest` leave behind as well, padding the record of the
    previous attempt and reporting work in a round where the Coder did none.

    The patterns go in `.git/info/exclude` rather than in a `.gitignore`, so
    that nothing of ours is added to the project the benchmark grades.
    """
    exclude_path = WORKSPACE / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a", encoding="utf-8") as handle:
        for pattern in BUILD_OUTPUT:
            print(pattern, file=handle)


if __name__ == "__main__":
    main()
