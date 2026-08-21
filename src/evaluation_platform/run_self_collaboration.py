import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SELF_COLLABORATION_ROOT = Path("/installed-agent/self-collaboration")
WORKSPACE = Path("/workspace")
HISTORY_PATH = Path("/logs/agent/session-history.json")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")


@dataclass
class UsageTotals:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    responses: int = 0

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


def main() -> None:
    hyperparameters = _read_hyperparameters()
    instruction = _read_instruction()
    sys.path.insert(0, str(SELF_COLLABORATION_ROOT))

    agent_module = importlib.import_module("core.agent")
    config_module = importlib.import_module("core.config")
    repo_tools_module = importlib.import_module("core.repo_tools")

    model_config = config_module.ModelConfig(
        max_tokens=hyperparameters["max_tokens"],
        temperature=hyperparameters["temperature"],
        top_p=hyperparameters["top_p"],
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
                _record_response_usage(response, totals)
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


def _record_response_usage(response: Any, totals: UsageTotals) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    totals.input_tokens += _usage_value(usage, "prompt_tokens")
    totals.output_tokens += _usage_value(usage, "completion_tokens")
    details = getattr(usage, "prompt_tokens_details", None)
    totals.cached_input_tokens += _usage_value(details, "cached_tokens")
    cost = getattr(usage, "cost", None)
    if isinstance(cost, (int, float)):
        totals.cost_usd = (totals.cost_usd or 0.0) + float(cost)
    totals.responses += 1


def _usage_value(container: Any, key: str) -> int:
    if isinstance(container, dict):
        value = container.get(key)
    else:
        value = getattr(container, key, None)
    return value if isinstance(value, int) else 0


def _read_instruction() -> str:
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(
        encoding="utf-8"
    )


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
