"""Run CodeTeam inside a Harbor task container.

CodeTeam ships an entry point of its own, `app/main.py`, which reads a handful
of environment variables, builds a configuration, and generates a repository
into a timestamped directory beneath its workspace. Three of those defaults are
wrong for an evaluation harness rather than for a local run, and none of them
is reachable through the environment:

  * The generated repository must land in Harbor's `/workspace` itself, since
    that is the directory the task's verifier reads. The tool's own layout puts
    it one level down, under a name containing the time it started.
  * The tool's run artifacts — the competing design sketches, the CTO's choice,
    each QA round — belong with the job's logs, not inside the repository being
    graded. Written to the default location they become files the verifier sees.
  * `temperature` has no environment override at all, so the sampling of a run
    could not be part of its recorded setup.

This runner therefore assembles the same objects `app/main.py` does, from the
experiment configuration's hyperparameters, and leaves the workflow itself
untouched. It also wraps the model client to record what the run spent and to
survive a rate limit, neither of which the tool does on its own.
"""

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

# The runner is uploaded next to its own dependencies, so its directory carries
# the shared usage accounting under a flat name. On the host, where the tests
# import this module from the package, that same directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_usage import UsageTotals, record_response_usage  # noqa: E402

CODE_TEAM_ROOT = Path("/installed-agent/code-team")
WORKSPACE = Path("/workspace")
ARTIFACTS_DIR = Path("/logs/agent/codeteam")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")

# How long the wrapper waits before retrying a rate-limited request, in
# seconds, once CodeTeam's own three attempts have been exhausted.
RATE_LIMIT_WAITS = (15, 30)


def main() -> None:
    hyperparameters = _read_hyperparameters()
    instruction = _read_instruction()
    sys.path.insert(0, str(CODE_TEAM_ROOT))

    config_module = importlib.import_module("app.config")
    context_module = importlib.import_module("orchestrator.context")
    workflow_module = importlib.import_module("orchestrator.workflow_async")
    requirements_module = importlib.import_module("core.requirements_preprocessor")
    artifacts_module = importlib.import_module("utils.run_artifacts")

    config = _build_config(config_module, hyperparameters)
    usage_totals = UsageTotals()
    llm = _build_llm(config, hyperparameters, usage_totals)
    context = _build_context(
        context_module,
        artifacts_module,
        config=config,
        llm=llm,
        rag=_build_rag(config),
    )
    _record_resolved_setup(hyperparameters, config)

    question = instruction
    if config.preprocess_requirements:
        question = requirements_module.preprocess_requirements(question)

    workflow = workflow_module.MultiAgentCodegenWorkflowAsync(context)
    try:
        repo_path = asyncio.run(workflow.run(question=question))
    finally:
        USAGE_PATH.write_text(
            json.dumps(usage_totals.to_dict(), indent=2),
            encoding="utf-8",
        )
    print(f"Done. Repo at: {repo_path}")


def _build_config(config_module: Any, hyperparameters: dict[str, Any]) -> Any:
    """Translate the experiment's hyperparameters into CodeTeam's own config.

    The names differ in two places, deliberately. `max_qa_rounds` is
    CodeTeam's `max_rounds`, renamed because Self-Collaboration has a
    `max_rounds` of its own that counts something else entirely, and a
    configuration read beside that one should not suggest the two are
    comparable. The developer-allocation and Git settings are flattened out of
    their nested sections, so that each of the paper's three ablations is one
    key in the experiment configuration.
    """
    config = config_module.SystemConfig()

    config.workspace = str(WORKSPACE)
    config.artifacts_enabled = True
    config.artifacts_dir = str(ARTIFACTS_DIR)
    # The asynchronous workflow is the one the paper describes: developers
    # implement concurrently under the dependency-aware scheduler.
    config.async_mode = True

    config.architects = hyperparameters["architects"]
    config.architect_seed = hyperparameters.get("architect_seed")
    config.sds_retry = hyperparameters["sds_retry"]
    config.max_rounds = hyperparameters["max_qa_rounds"]
    config.preprocess_requirements = hyperparameters["preprocess_requirements"]
    config.max_wall_clock_seconds = hyperparameters.get("max_wall_clock_seconds")
    config.max_token_budget = hyperparameters.get("max_token_budget")

    config.git.enabled = hyperparameters["git_coordination"]
    allocation = config.developer_allocation
    allocation.dynamic_enabled = hyperparameters["dynamic_developer_allocation"]
    allocation.fixed_agents = hyperparameters["fixed_developer_agents"]
    allocation.assignment_seed = hyperparameters.get("developer_assignment_seed")

    config.rag.enabled = hyperparameters["rag_enabled"]
    config.rag.index_backend = hyperparameters["rag_backend"]
    config.rag.top_k = hyperparameters["rag_top_k"]

    config.llm.provider = "openai"
    config.llm.model = os.environ.get("MODEL") or config.llm.model
    config.llm.base_url = os.environ.get("BASE_URL") or config.llm.base_url
    config.llm.max_tokens = hyperparameters["max_tokens"]
    config.llm.temperature = hyperparameters["temperature"]
    config.llm.top_p = hyperparameters["top_p"]
    return config


def _build_llm(
        config: Any,
        hyperparameters: dict[str, Any],
        totals: UsageTotals,
) -> Any:
    """CodeTeam's own OpenAI client, instrumented rather than replaced.

    Two things the tool does not do are needed here. It records only a total
    token count, which is not enough to report input, cached, output, and
    reasoning tokens separately; and it has no notion of reasoning settings at
    all, so the amount of reasoning a run performs would be whatever the
    provider defaults to for that model at that moment — a variable set
    outside the experiment and absent from its record.

    Both are supplied by wrapping the one method the tool calls, so the tool's
    own request construction, retries, and JSON repair are unchanged.
    """
    llm_module = importlib.import_module("core.llm_openai")
    api_key_env = os.environ.get("API_KEY_ENV", "OPENAI_API_KEY")
    llm = llm_module.OpenAILLM(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
        base_url=config.llm.base_url,
        api_key=os.environ.get(api_key_env),
    )
    llm.client = _instrumented_client(
        llm.client,
        totals,
        model=config.llm.model,
        reasoning_effort=hyperparameters.get("reasoning_effort"),
        request_extra=hyperparameters.get("request_extra"),
    )
    return llm


def _instrumented_client(
        client: Any,
        totals: UsageTotals,
        model: str,
        reasoning_effort: str | None,
        request_extra: dict[str, Any] | None,
        sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """A stand-in for the OpenAI client that CodeTeam cannot tell apart.

    CodeTeam reaches the API through `client.chat.completions.create` and
    nowhere else, so intercepting that one call is enough. Everything else is
    forwarded to the real client untouched.
    """

    def create(**kwargs: Any) -> Any:
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if request_extra:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **request_extra}
        for wait in (*RATE_LIMIT_WAITS, None):
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as error:
                if not _is_rate_limit(error):
                    raise
                if wait is None:
                    raise RuntimeError(
                        f"The LLM provider rate-limited model {model!r} after "
                        f"{len(RATE_LIMIT_WAITS) + 1} attempts. Retry later or "
                        "select a model with available capacity."
                    ) from error
                sleep(wait)
                continue
            record_response_usage(response, totals)
            return response
        raise AssertionError("unreachable")

    return _ClientProxy(client, create)


class _ClientProxy:
    """The real client, with `chat.completions.create` replaced."""

    def __init__(self, client: Any, create: Callable[..., Any]) -> None:
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _is_rate_limit(error: Exception) -> bool:
    """Whether a failed request was throttled rather than rejected.

    Only a throttled request is worth waiting for: a rejected credential or an
    unknown model will be refused just as quickly the second time. The
    provider's status code decides it where the client exposes one, and the
    message otherwise, since the tool's paths raise plain errors of their own.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429
    return "rate limit" in str(error).lower() or "429" in str(error)


def _build_rag(config: Any) -> Any:
    """The Architect stage's retrieval client, when the run asks for one.

    Constructed here rather than by the tool's `bootstrap`, which builds the
    model client too and would undo the instrumentation above. The corpus and
    its cached embeddings travel with the tool's own checkout.
    """
    if not config.rag.enabled:
        return None
    rag_module = importlib.import_module("rag.rag_client")
    return rag_module.RAGClient(config.rag)


def _build_context(
        context_module: Any,
        artifacts_module: Any,
        config: Any,
        llm: Any,
        rag: Any,
) -> Any:
    """CodeTeam's context, with the generated repository placed in `/workspace`.

    The tool generates into a fresh `repo-<timestamp>` directory beneath its
    workspace, which suits a local run that keeps its attempts side by side.
    Harbor grades the workspace itself: a task's verifier copies what it finds
    at `/workspace` on top of the benchmark's reference tests. The one method
    that decides the location is overridden, and nothing else about the
    workflow changes.
    """

    class HarborWorkspaceContext(context_module.Context):
        def make_repo_root(self) -> str:
            WORKSPACE.mkdir(parents=True, exist_ok=True)
            return str(WORKSPACE)

    return HarborWorkspaceContext(
        cfg=config,
        llm=llm,
        rag=rag,
        artifacts=artifacts_module.RunArtifacts(
            str(ARTIFACTS_DIR), enabled=config.artifacts_enabled
        ),
    )


def _read_instruction() -> str:
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(
        encoding="utf-8"
    )


def _read_hyperparameters() -> dict[str, Any]:
    return json.loads(
        Path(os.environ["CODE_TEAM_HYPERPARAMETERS_PATH"]).read_text(
            encoding="utf-8"
        )
    )


def _record_resolved_setup(
        hyperparameters: dict[str, Any],
        config: Any,
) -> None:
    """Record what actually ran, next to the run's other logs.

    The experiment configuration states the intended setup; this states the
    observed one, including the commit of the code generation tool that the
    container ended up with and the model the environment resolved to.
    """
    RESOLVED_SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "code_team_commit": _installed_commit(),
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "hyperparameters": hyperparameters,
                "architects": config.architects,
                "qa_rounds_allowed": config.max_rounds,
                "rag_enabled": config.rag.enabled,
                "rag_backend": config.rag.index_backend,
                "git_coordination": config.git.enabled,
                "dynamic_developer_allocation": (
                    config.developer_allocation.dynamic_enabled
                ),
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
            cwd=CODE_TEAM_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
