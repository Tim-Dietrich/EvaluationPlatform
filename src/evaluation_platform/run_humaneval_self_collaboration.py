"""Run Self-Collaboration's own HumanEval entry point on one Harbor task.

The tool ships two entry points and this is the other one. Where
`run_self_collaboration.py` drives `core.agent.SelfCollabSession` — the
repository shape, an Analyst that localizes files and a Coder that edits them —
this drives `run_humaneval.run_task`, which is what `bash run.sh` invokes and
therefore the code path behind the paper's HumanEval numbers.

Nothing of the method is reimplemented here. `run_task` is imported and
called, so the Analyst's prompt, the Coder's tool loop, the Tester that writes
its own `check(candidate)` cases, and the order they run in are all the
authors'. What this file does is the three things the harness has to do around
it, and each one is recorded in `resolved-setup.json`:

  * `run_task` takes one problem as a dict, while the module's own `main()`
    iterates a HuggingFace dataset. Harbor gives out one task per container,
    so the problem comes from the task package's `problem.json` instead.

  * `run_task` returns the generated code and leaves it in a working directory
    of its own. The benchmark grades `/workspace/solution.py`, so the returned
    code is written there.

  * The tool has no accounting of its own. The one function it calls the model
    through is wrapped, in both places it is referenced, so that what a run
    spent and how its requests ended are counted without changing what is
    sent.

What is deliberately *not* here is a reasoning setting or a `top_p`. On this
revision `core.backend.call_llm_with_tools` builds a request from `model`,
`messages`, `max_tokens` and `temperature` and nothing else, for all three
roles, so those two would be dropped in transit. The experiment configuration
refuses them for that reason, and this file records `top_p` as inert rather
than implying it took effect.
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# The runner is uploaded next to its own dependencies, so its directory carries
# the shared accounting under flat names. On the host, where the tests import
# this module from the package, that same directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_categories import FailureTally  # noqa: E402
from model_usage import UsageTotals  # noqa: E402
from model_routing import (  # noqa: E402
    ServedProviders,
    install_client_routing,
    routing_from_env,
)
# The wrapper that counts what a call spent and how it ended. Shared with the
# repository-shape runner rather than written twice: it is the same tool, the
# same client and the same failure categories, and two copies would be two
# things to keep in step.
from run_self_collaboration import _usage_recording_call  # noqa: E402

SELF_COLLABORATION_ROOT = Path("/installed-agent/self-collaboration")
WORKSPACE = Path("/workspace")
# Where the benchmark looks for the answer. The task states this in its
# instruction and its verifier reads exactly this path.
SOLUTION_PATH = WORKSPACE / "solution.py"
# The problem, as the task image carries it. Not the instruction markdown: the
# tool composes its own requirement string from the raw prompt and entry point,
# and handing it those is what keeps that composition the authors'.
PROBLEM_PATH = Path("/task/problem.json")
# The tool's own scratch directory, kept out of `/workspace` so that nothing
# but the graded file is in the graded directory. It is copied into the logs
# afterwards, because the Tester's generated tests and its runs are the record
# of what the Tester actually did.
WORK_DIR = Path("/installed-agent/humaneval-work")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")
WORK_ARCHIVE = Path("/logs/agent/humaneval-work")


def main() -> None:
    hyperparameters = _read_hyperparameters()
    problem = json.loads(PROBLEM_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(SELF_COLLABORATION_ROOT))

    config_module = importlib.import_module("core.config")
    agent_module = importlib.import_module("core.agent")
    # Imports `datasets` and `tqdm` at module level for the CLI loop in its
    # `main()`, which is never called from here. Both are the tool's own
    # declared requirements and the agent installs them.
    humaneval_module = importlib.import_module("run_humaneval")

    # Pin which of the provider's servers may answer, before any request is
    # made. The wrapper sits on the OpenAI client rather than on a call site,
    # so it reaches every request the tool builds; what changes is the server,
    # not the messages, the sampling, or the model.
    routing = routing_from_env()
    observed = ServedProviders()
    install_client_routing(routing, observed)

    model_config = config_module.ModelConfig(**_model_settings(hyperparameters))
    failures = FailureTally()
    # Written before the run starts, so a run that dies in its first round
    # still says what it was, and again after it, once the servers that
    # answered and the limits that were hit are known.
    _record_resolved_setup(
        hyperparameters, model_config, routing, observed, problem, failures
    )

    usage_totals = UsageTotals()
    counted = _usage_recording_call(
        agent_module.call_llm_with_tools,
        usage_totals,
        model=model_config.model,
        failures=failures,
    )
    # Two references, one function. `core.agent` calls it for the Coder's tool
    # loop; `run_humaneval` imported it into its own namespace for the Analyst
    # and the Tester. Patching one and not the other would count half a run.
    agent_module.call_llm_with_tools = counted
    humaneval_module.call_llm_with_tools = counted

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        code = humaneval_module.run_task(
            problem,
            model_config,
            str(WORK_DIR),
            max_round=hyperparameters["max_rounds"],
            max_steps=hyperparameters["max_steps"],
            verbose=True,
        )
    finally:
        USAGE_PATH.write_text(
            json.dumps(usage_totals.to_dict(), indent=2), encoding="utf-8"
        )
        _archive_work_dir()

    _write_solution(code)
    _record_resolved_setup(
        hyperparameters, model_config, routing, observed, problem, failures, code
    )


def _write_solution(code: str) -> None:
    """Put the generated code where the benchmark grades.

    `run_task` returns the contents of the `solution.py` it had its Coder
    write, and an empty string when the Coder wrote nothing at all. Writing an
    empty file in that case would turn 'no answer' into 'an answer that does
    not parse', which the verifier reports differently, so nothing is written
    and the record says so.
    """
    if not code.strip():
        print("Self-Collaboration produced no code; nothing to grade.")
        return
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SOLUTION_PATH.write_text(code, encoding="utf-8")
    print(f"Wrote {len(code)} characters to {SOLUTION_PATH}.")


def _archive_work_dir() -> None:
    """Keep the Tester's generated tests and runs with the trial's logs."""
    if not WORK_DIR.exists():
        return
    if WORK_ARCHIVE.exists():
        shutil.rmtree(WORK_ARCHIVE, ignore_errors=True)
    WORK_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WORK_DIR, WORK_ARCHIVE, dirs_exist_ok=True)


def _model_settings(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Model settings for this run, as the tool's own config accepts them.

    All three sampling values are set, because `ModelConfig` carries all three;
    only two of them are sent, because this entry point's requests are built by
    `call_llm_with_tools`, which does not read `top_p`. The resolved setup
    records that rather than leaving it to be inferred.
    """
    return {
        "max_tokens": hyperparameters["max_tokens"],
        "temperature": hyperparameters["temperature"],
        "top_p": hyperparameters["top_p"],
    }


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
        problem: dict[str, Any],
        failures: FailureTally,
        code: str | None = None,
) -> None:
    """Record what actually ran, next to the run's other logs.

    The experiment configuration states the intended setup; this states the
    observed one, including the commit of the tool the container ended up with
    and which of the configured knobs this entry point could not use.
    """
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "self_collaboration_commit": _installed_commit(),
                "entry_point": "run_humaneval.run_task",
                "entry_point_note": (
                    "The authors' own HumanEval entry point, as 'bash run.sh' "
                    "invokes it. Imported and called; not reimplemented."
                ),
                "task_id": problem.get("task_id"),
                "declared_entry_point": problem.get("entry_point"),
                "model": model_config.model,
                "base_url": model_config.base_url,
                # What the run asked of the aggregator, and which server
                # answered. A model name does not name a server.
                "routing": routing,
                "providers_served": observed.names(),
                "hyperparameters": hyperparameters,
                "generation": {
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "max_tokens": model_config.max_tokens,
                    "source": "configs/generation.yaml",
                    "applied_via": "core.config.ModelConfig, one for all roles",
                    # `call_llm_with_tools` sends model, messages, max_tokens
                    # and temperature. Everything else on the config object is
                    # carried and not sent.
                    "inert_on_this_revision": ["top_p"],
                },
                # Knobs this shape has no use for. They keep their defaults so
                # that the repository shape is unaffected, and are named here
                # so no reader takes them for settings that decided something.
                "inert_on_this_task_shape": ["analyst_steps", "coder_steps"],
                "harness_compensations": [
                    "run_task is called per task; the module's own main() "
                    "iterates a HuggingFace dataset and is not used.",
                    "The problem is read from the task package's problem.json "
                    "rather than from load_dataset.",
                    "The returned code is written to /workspace/solution.py, "
                    "where the benchmark grades.",
                ],
                # Why a round ended where it did, in categories that stay apart
                # from each other, from a timeout, and from a failing test.
                "failures": failures.to_dict(),
                "reasoning_requested": False,
                "solution_written": bool(code and code.strip()),
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


if __name__ == "__main__":
    main()
