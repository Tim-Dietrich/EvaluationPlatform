"""What the session history and the trajectory written from it promise.

The conversion runs after the trial is over, on files the runner already wrote,
so its failures are silent by construction: a broken converter does not fail a
run, it just leaves the run unrecorded. These tests are the only thing standing
between that and a sweep that finishes with nothing to inspect.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from harbor.models.agent.context import AgentContext
from harbor.utils.trajectory_validator import TrajectoryValidator

from evaluation_platform.self_collaboration_agent import (
    SELF_COLLABORATION_COMMIT,
    SelfCollaborationAgent,
)
from evaluation_platform.self_collaboration_trajectory import (
    build_trajectory,
    write_trajectory,
)

# One Analyst phase, then two Coder rounds each judged by the Tester, then a
# third Coder round that the tool does not test because it is the last. This is
# the shape of every recorded run of the tool that reached its round limit.
HISTORY = {
    "analyst": {"analysis": '{"reasoning": "the workspace is empty", "files": []}'},
    "round_0": {
        "coder": "wrote src/library/__init__.py",
        "test_passed": False,
        "test_output": "no tests ran in 0.00s",
    },
    "round_1": {
        "coder": "added the parser",
        "test_passed": False,
        "test_output": "1 failed, 3 passed",
    },
    "round_2": {"coder": "fixed the parser"},
}
USAGE = {
    "input_tokens": 943207,
    "cached_input_tokens": 671488,
    "output_tokens": 14620,
    "reasoning_tokens": 320,
    "cost_usd": 0.0298,
}
RESOLVED_SETUP = {
    "self_collaboration_commit": SELF_COLLABORATION_COMMIT,
    "model": "deepseek/deepseek-v4-flash-0731",
    "base_url": "https://openrouter.ai/api/v1",
    "hyperparameters": {"max_rounds": 3, "test_command": "python -m pytest -q"},
}


def build(history=None, **overrides):
    settings = {
        "instruction": "Build the library",
        "usage": USAGE,
        "resolved_setup": RESOLVED_SETUP,
        "session_id": "a-session",
    }
    settings.update(overrides)
    return build_trajectory(HISTORY if history is None else history, **settings)


def test_the_phases_appear_in_the_order_the_tool_ran_them():
    trajectory = build()

    assert [step.source for step in trajectory.steps] == [
        "user", "agent", "agent", "system", "agent", "system", "agent",
    ]
    assert [(step.extra or {}).get("phase") for step in trajectory.steps] == [
        None, "analyst", "coder", "tester", "coder", "tester", "coder",
    ]
    # ATIF requires the ids to run sequentially from 1, and the phases are read
    # out of a dictionary rather than a list, so this is not free.
    assert [step.step_id for step in trajectory.steps] == [1, 2, 3, 4, 5, 6, 7]


def test_a_tenth_round_follows_the_ninth_rather_than_the_first():
    """The tool numbers rounds inside the key, where sorting is by string."""
    history = {f"round_{index}": {"coder": f"round {index}"} for index in range(11)}

    trajectory = build(history, instruction=None)

    assert [step.message for step in trajectory.steps][-2:] == ["round 9", "round 10"]


def test_the_tester_is_a_system_step_that_called_no_model():
    """It runs a test command, and nothing about it is a model turn.

    Read as an agent step it would inflate every count of what the solution's
    roles did by one per round, which is the comparison this platform exists to
    make.
    """
    tester = build().steps[3]

    assert tester.source == "system"
    assert tester.llm_call_count == 0
    assert tester.model_name is None
    assert tester.metrics is None
    assert tester.observation.results[0].content == "no tests ran in 0.00s"
    assert tester.extra == {"phase": "tester", "round": 0, "test_passed": False}


def test_the_last_round_has_no_tester_because_the_tool_does_not_run_one():
    trajectory = build()

    assert (trajectory.steps[-1].extra or {})["phase"] == "coder"
    assert trajectory.steps[-1].extra["round"] == 2


def test_what_the_run_spent_is_reported_once_and_not_per_step():
    """The session history records no per-response usage, so neither does this.

    Spreading a total across steps, or defaulting the steps to zero, would both
    read as a measurement. The total is the measurement; the steps are silent.
    """
    trajectory = build()

    assert all(step.metrics is None for step in trajectory.steps)
    assert trajectory.final_metrics.total_prompt_tokens == 943207
    assert trajectory.final_metrics.total_cached_tokens == 671488
    assert trajectory.final_metrics.total_completion_tokens == 14620
    assert trajectory.final_metrics.total_cost_usd == 0.0298
    assert trajectory.final_metrics.total_steps == 7
    # ATIF has no field for these, and they are the platform's own distinction
    # between budget spent thinking and budget spent answering.
    assert trajectory.final_metrics.extra == {"reasoning_tokens": 320}


def test_the_evaluated_revision_is_the_trajectorys_agent_version():
    """A trajectory that cannot be traced to a revision is not a record."""
    trajectory = build()

    assert trajectory.agent.version == SELF_COLLABORATION_COMMIT
    assert trajectory.agent.model_name == "deepseek/deepseek-v4-flash-0731"
    assert trajectory.agent.extra["resolved_setup"] == RESOLVED_SETUP


def test_a_run_that_never_wrote_its_setup_is_recorded_without_one():
    """Absent, rather than zeroed or guessed, as everywhere else here."""
    trajectory = build(usage=None, resolved_setup=None)

    assert trajectory.agent.version == "unknown"
    assert trajectory.agent.model_name is None
    assert trajectory.agent.extra is None
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert all(step.model_name is None for step in trajectory.steps)


def test_a_conversion_without_the_instruction_says_so_in_its_notes():
    """Runs already on disk have no instruction to recover, and open blind."""
    trajectory = build(instruction=None)

    assert trajectory.steps[0].source == "agent"
    assert "not available at conversion time" in trajectory.notes


def test_the_notes_say_the_steps_are_phases_rather_than_model_turns():
    """The one thing a reader must not get wrong about this document."""
    trajectory = build()

    assert "phase rather than a model turn" in trajectory.notes


def test_a_history_with_no_phases_produces_no_trajectory():
    """ATIF requires at least one step; a run that recorded nothing has none."""
    assert build({}, instruction=None) is None


def test_the_result_validates_against_harbors_own_validator():
    validator = TrajectoryValidator()

    assert validator.validate(build().to_json_dict()) is True
    assert validator.errors == []


def test_no_credential_reaches_the_trajectory():
    """Harbor's own Terminus writes its API key into `agent.extra`.

    This arm's setup record names the endpoint and the credential's variable,
    never its value, and the trajectory copies that record rather than the
    agent's keyword arguments. The check is here because the file is the one
    this platform would share.
    """
    setup = {**RESOLVED_SETUP, "api_key_env": "OPENROUTER_API_KEY"}

    serialised = json.dumps(build(resolved_setup=setup).to_json_dict())

    assert "OPENROUTER_API_KEY" in serialised
    assert "sk-or-v1-" not in serialised


def test_write_trajectory_reads_the_files_the_runner_left_behind(tmp_path):
    (tmp_path / "session-history.json").write_text(
        json.dumps({"history": HISTORY}), encoding="utf-8"
    )
    (tmp_path / "model-usage.json").write_text(json.dumps(USAGE), encoding="utf-8")
    (tmp_path / "resolved-setup.json").write_text(
        json.dumps(RESOLVED_SETUP), encoding="utf-8"
    )

    written = write_trajectory(tmp_path, instruction="Build the library")

    assert written == tmp_path / "trajectory.json"
    assert TrajectoryValidator().validate(
        json.loads(written.read_text(encoding="utf-8"))
    ) is True


def test_a_trial_that_died_before_writing_a_history_writes_no_trajectory(tmp_path):
    """The common case, and not an error: the run failed, not the record."""
    assert write_trajectory(tmp_path) is None
    assert not (tmp_path / "trajectory.json").exists()


def test_an_unreadable_history_is_reported_rather_than_raised(tmp_path):
    """A finished run must not fail a second time over the record of it."""
    (tmp_path / "session-history.json").write_text("{not json", encoding="utf-8")
    warnings = []
    logger = cast(
        object, SimpleNamespace(debug=lambda message: None, warning=warnings.append)
    )

    assert write_trajectory(tmp_path, logger=logger) is None
    assert not (tmp_path / "trajectory.json").exists()


def test_the_agent_writes_the_trajectory_after_the_run(tmp_path):
    """The whole point, end to end: `run` remembers the task, the post-run hook
    converts, and Harbor's usage fields are still populated as before."""
    (tmp_path / "session-history.json").write_text(
        json.dumps({"history": HISTORY}), encoding="utf-8"
    )
    (tmp_path / "model-usage.json").write_text(
        json.dumps({"input_tokens": 90, "cached_input_tokens": 15,
                    "output_tokens": 25, "cost_usd": 0.0}),
        encoding="utf-8",
    )
    agent = SelfCollaborationAgent(logs_dir=tmp_path)
    agent.instruction = "Build the library"
    context = AgentContext()

    agent.populate_context_post_run(context)

    trajectory = json.loads(
        (tmp_path / "trajectory.json").read_text(encoding="utf-8")
    )
    assert agent.SUPPORTS_ATIF is True
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["steps"][0]["message"] == "Build the library"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 90
    assert context.n_input_tokens == 90


def test_the_agent_records_no_trajectory_when_the_run_left_no_history(tmp_path):
    agent = SelfCollaborationAgent(logs_dir=tmp_path)

    agent.populate_context_post_run(AgentContext())

    assert not (tmp_path / "trajectory.json").exists()


def test_every_recorded_run_on_disk_still_converts():
    """The ten runs already in `jobs/` are the only real fixtures there are.

    Skipped where none is present, so a clean checkout does not fail; where
    they exist, a change to the tool's history format shows up here first.
    """
    histories = sorted(Path("jobs").glob("*/*/agent/session-history.json"))
    if not histories:
        return

    for history_path in histories:
        # The earliest runs predate `resolved-setup.json`, which is the point:
        # they convert without one rather than failing over it.
        setup_path = history_path.parent / "resolved-setup.json"
        trajectory = build_trajectory(
            json.loads(history_path.read_text(encoding="utf-8")).get("history") or {},
            instruction="Build the library",
            resolved_setup=(
                json.loads(setup_path.read_text(encoding="utf-8"))
                if setup_path.exists()
                else None
            ),
            session_id="a-session",
        )
        assert trajectory is not None, history_path
        assert TrajectoryValidator().validate(trajectory.to_json_dict()) is True
