"""Self-Collaboration's session history, as a Harbor trajectory (ATIF).

Harbor's own agent writes a `trajectory.json` beside every run it does, and
that file is what the results view and every downstream trajectory tool read.
Four of this platform's five arms write nothing of the kind, so a comparison
that is careful about tokens, revisions and hyperparameters is, at the level of
*what the run actually did*, between one arm that can be inspected and four
that cannot. This is the first of the four.

The conversion happens on the host, after the trial has synced its logs back,
from files the runner already writes. Nothing about the run itself changes, and
a run that is already on disk converts the same way as one finishing now.

What the conversion cannot invent is granularity. Self-Collaboration keeps one
entry per *phase*: the Analyst's own tool-calling turns, and each Coder's, are
collapsed by the tool into the single string that phase returned before
anything is written down. The steps here are therefore phases, not inferences,
and they carry no per-step metrics -- what the run spent is reported once, in
`final_metrics`, from the same `model-usage.json` the rest of the platform
counts with. The trajectory says so in its own `notes`, because a document that
looks turn-level and is not would be read as one.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    Trajectory,
)

SESSION_HISTORY_FILENAME = "session-history.json"
USAGE_FILENAME = "model-usage.json"
RESOLVED_SETUP_FILENAME = "resolved-setup.json"
TRAJECTORY_FILENAME = "trajectory.json"

# What the tool names the one phase that runs before the Coder rounds, and the
# prefix it gives each round. Both are its own keys, read rather than chosen.
ANALYST_KEY = "analyst"
ROUND_PREFIX = "round_"

NOTES = (
    "Converted from Self-Collaboration's session history. That record keeps "
    "one entry per phase, so each step here is a phase rather than a model "
    "turn: the Analyst's internal tool-calling turns, and each Coder's, are "
    "collapsed by the tool into the single result string the phase returned, "
    "and are not recoverable from what it writes down. Steps therefore carry "
    "no per-step metrics; what the run spent is reported once in "
    "final_metrics, counted by model_usage.py around the OpenAI client. The "
    "Tester phase runs a test command and no model, and appears as a system "
    "step whose observation is that command's output, truncated by the tool "
    "to its first 2000 characters."
)
MISSING_INSTRUCTION_NOTE = (
    " The task instruction was not available at conversion time, so this "
    "trajectory opens on the Analyst rather than on the user step that "
    "prompted it."
)


def write_trajectory(
        logs_dir: Path,
        instruction: str | None = None,
        session_id: str | None = None,
        logger: logging.Logger | None = None,
) -> Path | None:
    """Convert a finished run's logs into `trajectory.json`, or explain why not.

    Returns the path written, or None. A trial that died before its session
    history existed, or one whose history cannot be read, is reported and left
    alone: the trajectory is a record of the run, and a run that has already
    finished must not fail again over the record of it.
    """
    logger = logger or logging.getLogger(__name__)
    history_path = logs_dir / SESSION_HISTORY_FILENAME
    if not history_path.exists():
        logger.debug(f"No session history at {history_path}; no trajectory written")
        return None

    try:
        session_history = _read_json(history_path) or {}
        trajectory = build_trajectory(
            session_history.get("history") or {},
            instruction=instruction,
            usage=_read_json(logs_dir / USAGE_FILENAME),
            resolved_setup=_read_json(logs_dir / RESOLVED_SETUP_FILENAME),
            session_id=session_id or str(uuid.uuid4()),
        )
    except Exception as error:  # noqa: BLE001 - see the docstring above.
        logger.warning(f"Failed to convert session history to ATIF: {error}")
        return None
    if trajectory is None:
        logger.debug("Session history recorded no phases; no trajectory written")
        return None

    trajectory_path = logs_dir / TRAJECTORY_FILENAME
    trajectory_path.write_text(
        json.dumps(trajectory.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    return trajectory_path


def build_trajectory(
        history: dict[str, Any],
        instruction: str | None = None,
        usage: dict[str, Any] | None = None,
        resolved_setup: dict[str, Any] | None = None,
        session_id: str | None = None,
) -> Trajectory | None:
    """Assemble the ATIF document, or None where there is nothing to record.

    Every argument but the history is optional, because each comes from a file
    the runner writes at a different moment and a run can stop between any two
    of them. A missing input leaves its fields absent rather than zeroed, which
    is the rule `model_usage.py` already applies to what a run spent.
    """
    setup = resolved_setup or {}
    model = setup.get("model")
    steps: list[Step] = []

    if instruction is not None:
        steps.append(Step(step_id=1, source="user", message=instruction))
    analysis = (history.get(ANALYST_KEY) or {}).get("analysis")
    if analysis is not None:
        steps.append(
            Step(
                step_id=1,
                source="agent",
                model_name=model,
                message=analysis,
                extra={"phase": "analyst"},
            )
        )
    for index, round_record in _rounds(history):
        steps.extend(_round_steps(index, round_record, model))

    if not steps:
        return None
    # The tool's own keys decide the order above; the ids are assigned here so
    # that ATIF's requirement that they run sequentially from 1 cannot be
    # broken by a phase being added to or dropped from the sequence.
    for step_id, step in enumerate(steps, start=1):
        step.step_id = step_id

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name="self-collaboration",
            # The revision that ran, as the container reported it. This is the
            # same pin `resolved-setup.json` records, and it is the only honest
            # version for an arm whose subject is somebody else's repository.
            version=setup.get("self_collaboration_commit") or "unknown",
            model_name=model,
            # No credential reaches this file: `resolved-setup.json` records
            # the endpoint and the routing directive, never the key.
            extra={"resolved_setup": setup} if setup else None,
        ),
        steps=steps,
        notes=NOTES if instruction is not None else NOTES + MISSING_INSTRUCTION_NOTE,
        final_metrics=_final_metrics(usage, len(steps)),
    )


def _rounds(history: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """The Coder/Tester rounds, in the order they ran.

    The tool numbers them in the key rather than ordering them in a list, so
    they are sorted numerically here: `round_10` follows `round_9` and does not
    sort between `round_1` and `round_2`.
    """
    rounds = []
    for key, record in history.items():
        if not key.startswith(ROUND_PREFIX) or not isinstance(record, dict):
            continue
        suffix = key[len(ROUND_PREFIX):]
        if suffix.isdigit():
            rounds.append((int(suffix), record))
    return sorted(rounds, key=lambda pair: pair[0])


def _round_steps(
        index: int,
        record: dict[str, Any],
        model: str | None,
) -> list[Step]:
    """One Coder phase, and the Tester phase that judged it where there was one.

    The Tester runs a test command and no model, so it is a system step rather
    than an agent one, and its `llm_call_count` of zero says as much to a
    consumer counting inferences. The last round has no Tester, and neither has
    any round of a run configured without a test command.
    """
    steps = [
        Step(
            step_id=1,
            source="agent",
            model_name=model,
            message=record.get("coder") or "(no response)",
            extra={"phase": "coder", "round": index},
        )
    ]
    if "test_passed" not in record:
        return steps

    # The command itself is a property of the setup, not of the round: it is
    # the same one every round, and `agent.extra` already records it once.
    passed = bool(record.get("test_passed"))
    steps.append(
        Step(
            step_id=1,
            source="system",
            message=f"Tester phase: tests {'passed' if passed else 'failed'}.",
            observation=Observation(
                results=[ObservationResult(content=record.get("test_output") or "")]
            ),
            llm_call_count=0,
            extra={"phase": "tester", "round": index, "test_passed": passed},
        )
    )
    return steps


def _final_metrics(usage: dict[str, Any] | None, total_steps: int) -> FinalMetrics:
    """What the run spent, under ATIF's names for it.

    `model-usage.json` counts one field ATIF has no place for. Reasoning tokens
    are billed inside the completion tokens and are recorded separately so a
    run shows how much of its budget went to thinking rather than to answering;
    dropping them here to fit the schema would lose a distinction the platform
    went to the trouble of drawing, so they go in `extra` instead.
    """
    usage = usage or {}
    reasoning_tokens = usage.get("reasoning_tokens")
    return FinalMetrics(
        total_prompt_tokens=usage.get("input_tokens"),
        total_completion_tokens=usage.get("output_tokens"),
        total_cached_tokens=usage.get("cached_input_tokens"),
        total_cost_usd=usage.get("cost_usd"),
        total_steps=total_steps,
        extra=(
            {"reasoning_tokens": reasoning_tokens}
            if isinstance(reasoning_tokens, int)
            else None
        ),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    """One of the runner's records, or None where the run never wrote it."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
