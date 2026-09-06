"""The generated HumanEval tasks, checked against what actually consumes them.

The benchmark is generated rather than depended on, so the checks a registry
would have applied have to live here instead. Two kinds of thing go wrong with
a generated task and neither shows up until a run is already spending money:
the package can disagree with what Harbor reads, and the grader can disagree
with the pipeline it reproduces.
"""

import json
from pathlib import Path

import pytest

from harbor.models.trial.paths import EnvironmentPaths

from evaluation_platform.benchmark import local_tree_digest


ROOT = Path(__file__).parents[1]
TASKS_DIR = ROOT / "benchmarks" / "humaneval" / "tasks"

# The tasks are generated and git-ignored, so a checkout that has not run the
# generator has nothing to check. That is a state to skip in, not to fail in.
pytestmark = pytest.mark.skipif(
    not (TASKS_DIR / "HumanEval_0").is_dir(),
    reason=(
        "HumanEval tasks are generated; run "
        "'python scripts/build_humaneval_tasks.py' first."
    ),
)


def grade_source() -> str:
    return (TASKS_DIR / "HumanEval_0" / "tests" / "grade.py").read_text(
        encoding="utf-8"
    )


def test_the_grader_writes_the_reward_file_harbor_actually_reads():
    """The bug this test exists for, and why it was expensive.

    Harbor reads exactly two paths, `reward.txt` and `reward.json`, and its own
    task template's comment misnames the second as `rewards.json`. A task that
    follows the comment runs, calls the model, grades correctly, writes its
    rewards — and is then failed for having no reward file, after the whole
    cost of the trial has been paid. Pinned against Harbor's own constant
    rather than against a literal, so a rename upstream fails here rather than
    in a run.
    """
    # The full in-container path Harbor reads, not just its basename.
    expected = EnvironmentPaths.reward_json_path
    assert str(expected) == "/logs/verifier/reward.json"
    assert f'pathlib.Path("{expected}")' in grade_source()


def test_both_rewards_are_written_under_the_key_harbor_treats_as_primary():
    """`reward` is what Harbor reports as the trial's reward.

    `_parse_reward_text` wraps a bare float as `{"reward": ...}`, so a JSON
    reward file has to use the same name for the figure that means the same
    thing. HumanEval-ET travels beside it under its own key.
    """
    source = grade_source()
    assert '("reward", "test")' in source
    assert '("humaneval_et", "test_et")' in source


def test_the_hidden_tests_are_not_in_anything_the_agent_can_read():
    """The one failure mode that would invalidate every number at once.

    Harbor uploads `tests/` only after the agent has finished, so the tests are
    hidden by where they live. This checks the other half: that nothing in the
    task's own image carries them in.
    """
    for task_dir in sorted(TASKS_DIR.iterdir())[:20]:
        hidden = json.loads(
            (task_dir / "tests" / "problem.json").read_text(encoding="utf-8")
        )
        visible = json.loads(
            (task_dir / "environment" / "problem.json").read_text(encoding="utf-8")
        )

        assert set(visible) == {"task_id", "prompt", "entry_point"}, task_dir.name
        assert hidden["test"] not in json.dumps(visible), task_dir.name

        for path in (task_dir / "environment").rglob("*"):
            if path.is_file():
                assert "def check(" not in path.read_text(
                    encoding="utf-8", errors="replace"
                ), path


def test_the_graded_program_is_assembled_the_way_the_authors_assemble_it():
    """`prompt + completion + "\\n" + test + "\\n" + check(entry_point)`.

    From `evaluate/execute/_execution.py`, with `all_evaluate.py` supplying an
    empty prompt and a completion of preamble plus generation. The entry point
    is re-derived from the generated code rather than taken from the dataset,
    which is more permissive than the reference harness and is the behaviour
    the published figure was measured under.
    """
    source = grade_source()
    assert 'generation = PROBLEM["preamble"] + "\\n" + solution' in source
    assert 'entry_point = find_method_name(generation) or "candidate"' in source
    assert (
        'generation + "\\n" + PROBLEM[test] + "\\n" + f"check({entry_point})"'
        in source
    )


def test_the_preamble_stops_before_the_prompts_last_def():
    """A HumanEval prompt ends in an unclosed signature.

    Prepending it whole would be a syntax error, which is why the authors take
    only the part before its last `def`. Checked on the two problems where the
    prompt defines a helper before the target function, since those are the
    ones where truncating at the wrong `def` would still parse.
    """
    for name in ("HumanEval_32", "HumanEval_38", "HumanEval_50"):
        problem = json.loads(
            (TASKS_DIR / name / "tests" / "problem.json").read_text(encoding="utf-8")
        )
        preamble = problem["preamble"]
        assert "def " not in preamble.split("\n")[-1]
        assert not preamble.rstrip().endswith('"""')


def test_every_task_carries_an_oracle_and_a_verifier():
    """A task whose canonical solution is missing cannot be checked at all."""
    tasks = sorted(path for path in TASKS_DIR.iterdir() if path.is_dir())
    assert len(tasks) == 164

    for task_dir in tasks:
        for relative in (
                "task.toml",
                "instruction.md",
                "environment/Dockerfile",
                "environment/problem.json",
                "tests/test.sh",
                "tests/grade.py",
                "tests/problem.json",
                "solution/solve.sh",
        ):
            assert (task_dir / relative).is_file(), f"{task_dir.name}/{relative}"


def test_the_digest_is_the_one_the_launcher_will_record():
    """The generator prints what the launcher computes, or the pin means little."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_humaneval_tasks

    assert build_humaneval_tasks.tree_digest is local_tree_digest
    assert len(local_tree_digest(TASKS_DIR)) == 64
