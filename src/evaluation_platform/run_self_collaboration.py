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

from model_usage import UsageTotals, record_response_usage  # noqa: E402

SELF_COLLABORATION_ROOT = Path("/installed-agent/self-collaboration")
WORKSPACE = Path("/workspace")
HISTORY_PATH = Path("/logs/agent/session-history.json")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")


def main() -> None:
    hyperparameters = _read_hyperparameters()
    instruction = _read_instruction()
    sys.path.insert(0, str(SELF_COLLABORATION_ROOT))

    agent_module = importlib.import_module("core.agent")
    config_module = importlib.import_module("core.config")
    repo_tools_module = importlib.import_module("core.repo_tools")

    model_config = config_module.ModelConfig(
        **_model_settings(hyperparameters)
    )
    _record_resolved_setup(hyperparameters, model_config)

    usage_totals = UsageTotals()
    agent_module.call_llm_with_tools = _usage_recording_call(
        agent_module.call_llm_with_tools,
        usage_totals,
        model=model_config.model,
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
        sleep: Callable[[float], None] = time.sleep,
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(3):
            try:
                response = call(*args, **kwargs)
                record_response_usage(response, totals)
                return response
            except RuntimeError as error:
                if str(error) != "Failed to call LLM API with tools":
                    raise
                if attempt == 2:
                    raise RuntimeError(
                        f"The LLM provider rate-limited model {model!r} after "
                        "9 requests. Retry later or select a model with available "
                        "capacity."
                    ) from error
                sleep(15 * (attempt + 1))

        raise AssertionError("unreachable")

    return wrapped


def _read_instruction() -> str:
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(
        encoding="utf-8"
    )


def _model_settings(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Model settings for this run, as the tool's own config accepts them.

    Reasoning settings are passed only when configured, so that an unmodified
    revision of the tool can still be pinned for a baseline comparison; asking
    such a revision for reasoning fails loudly rather than dropping the setting.
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
                "hyperparameters": hyperparameters,
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
    subprocess.run(
        ["git", "commit", "--allow-empty", "--quiet", "-m", "Initial workspace"],
        cwd=WORKSPACE,
        check=True,
    )


if __name__ == "__main__":
    main()
