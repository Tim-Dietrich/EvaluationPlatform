"""The generated SketchEval tasks, checked against what actually consumes them.

The benchmark is generated rather than depended on, so the checks a registry
would have applied have to live here instead. Two kinds of thing go wrong with
a generated task and neither shows up until a run is already spending money:
the package can disagree with what Harbor reads, and the grader can disagree
with the metric it reproduces.

SketchEval adds a third, which HumanEval does not have. Its reference *is* the
answer — a whole repository of real source — and it has to travel inside the
task package so the verifier can score against it. Every path that reaches the
agent is therefore checked for it here.
"""

import json
import tomllib
from pathlib import Path

import pytest

from harbor.models.task.config import TaskConfig
from harbor.models.trial.paths import EnvironmentPaths

from evaluation_platform.benchmark import local_tree_digest


ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = ROOT / "benchmarks" / "sketcheval"
TASKS_DIR = BENCHMARK_DIR / "tasks"
SOURCES_PATH = BENCHMARK_DIR / "data" / "SOURCES.json"

# The nineteen repositories the CodeS repository vendors as SketchEval's Python
# half. Named rather than counted, because a generator that silently dropped
# one would still produce a plausible benchmark.
REPOSITORIES = (
    "CVE-2023-44487",
    "EVM_inscription",
    "EasyLiterature",
    "django-tui",
    "easier-docker",
    "epubhv",
    "every-breath-you-take",
    "fastui-chat",
    "flameshow",
    "kanban-python",
    "libgen_to_txt",
    "mactop",
    "pitch-visualizer",
    "pygraft",
    "pyobd",
    "sim-web-visualizer",
    "smol-podcaster",
    "van-gonography",
    "web.Monitor",
)

# The tasks are generated and git-ignored, so a checkout that has not run the
# generator has nothing to check. That is a state to skip in, not to fail in.
pytestmark = pytest.mark.skipif(
    not (TASKS_DIR / "epubhv").is_dir(),
    reason=(
        "SketchEval tasks are generated; run "
        "'python scripts/build_sketcheval_tasks.py' first."
    ),
)


def grade_source() -> str:
    return (TASKS_DIR / "epubhv" / "tests" / "grade.py").read_text(encoding="utf-8")


def test_the_grader_writes_the_reward_file_harbor_actually_reads():
    """Harbor reads `reward.json`, and its own task template misnames it.

    A task that follows that comment runs, calls the model, grades correctly,
    writes its rewards — and is then failed for having no reward file, after
    the whole cost of the trial has been paid. Pinned against Harbor's own
    constant rather than against a literal, so a rename upstream fails here
    rather than in a run.
    """
    expected = EnvironmentPaths.reward_json_path
    assert str(expected) == "/logs/verifier/reward.json"
    assert f'pathlib.Path("{expected}")' in grade_source()


def test_the_reference_repository_reaches_no_path_the_agent_can_read():
    """The failure mode that would invalidate every number at once.

    The reference is the answer. Harbor uploads `tests/` only after the agent
    has finished and `solution/` only when the oracle agent runs, so both
    copies are hidden by where they live. What is checked here is the third
    path — the task's own image — which is built before the agent starts and
    is the one place a reference file could reach it.
    """
    for name in REPOSITORIES:
        task_dir = TASKS_DIR / name
        reference = task_dir / "tests" / "reference"
        sources = sorted(reference.rglob("*.py"))
        assert sources, f"{name}: no reference Python files"

        environment = sorted(
            path for path in (task_dir / "environment").rglob("*") if path.is_file()
        )
        # The image carries a Dockerfile and the metric's build script, and
        # nothing else at all.
        assert {path.name for path in environment} == {
            "Dockerfile",
            "build_metric.py",
        }, name

        blob = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in environment
        )
        for source in sources:
            body = source.read_text(encoding="utf-8", errors="replace").strip()
            if len(body) > 200:
                assert body[:200] not in blob, f"{name}: {source.name} in the image"


def test_the_instruction_is_the_repositorys_readme_and_nothing_else():
    """CodeS quotes the instruction into all three of its phases, verbatim.

    Its driver reads `README.md` and interpolates it as `{readme}`, so anything
    appended here would be a paragraph inside a prompt the paper's numbers were
    not produced with. The benchmark keeps the file unchanged and pays for that
    in what it does not tell an arm with no such convention; the trade is
    documented in benchmarks/sketcheval/README.md.
    """
    for name in REPOSITORIES:
        instruction = (TASKS_DIR / name / "instruction.md").read_bytes()
        reference = (
            TASKS_DIR / name / "tests" / "reference" / "README.md"
        ).read_bytes()
        assert instruction == reference, name


def test_the_reward_file_holds_numbers_and_nothing_else():
    """The bug this test exists for, and why it was expensive.

    Harbor parses the whole of `reward.json` into `VerifierResult.rewards`,
    typed `dict[str, float | int]`. One string anywhere in that file fails the
    trial in pydantic — after every model call has been paid for, and with a
    message about float parsing that never mentions the repository it was
    grading. The first SketchEval trial died on `"repo": "CVE-2023-44487"`.

    So the grader writes numbers to `reward.json` and prose to
    `sketchbleu.json` beside it. Pinned against Harbor's own annotation rather
    than a literal, so a widening upstream shows up here rather than in a run.
    """
    from harbor.models.verifier.result import VerifierResult

    annotation = VerifierResult.model_fields["rewards"].annotation
    assert "float" in str(annotation) and "str" in str(annotation)

    source = grade_source()
    # The rewards leave through one function, and it drops anything that is not
    # a number before the file is written.
    assert "isinstance(value, (int, float)) and not isinstance(value, bool)" in source
    assert "REWARDS_PATH.write_text(json.dumps(numbers" in source
    # The prose has somewhere else to be.
    assert 'DIAGNOSTICS_PATH = pathlib.Path("/logs/verifier/sketchbleu.json")' in source
    # And `graded` travels as 0/1, since pydantic is not asked to read a
    # boolean as a number.
    assert '"graded": 0' in source and '"graded": 1' in source
    assert '"graded": True' not in source and '"graded": False' not in source


def test_every_reward_the_grader_can_write_validates_as_a_verifier_result():
    """The grader's own payloads, run through the model that rejected them.

    Checking the shape of the source is not the same as checking the file, so
    the two payload shapes are built here and validated. A reward file that
    parses is the difference between a graded trial and a trial that spent its
    whole budget and recorded nothing.
    """
    from harbor.models.verifier.result import VerifierResult

    components = {
        "ngram_match_score": 1.0,
        "weighted_ngram_match_score": 1.0,
        "syntax_match_score": 0.999814,
        "dataflow_match_score": 0.942857,
    }
    graded = {
        "reward": 0.985668,
        "graded": 1,
        "generated_python_files": 7,
        "generated_functions": 35,
        **components,
    }
    refused = {"reward": 0.0, "graded": 0, **dict.fromkeys(components, 0.0)}

    for payload in (graded, refused):
        result = VerifierResult(rewards=payload)
        assert result.rewards is not None
        assert result.rewards["reward"] == payload["reward"]


def test_the_reward_is_sketchbleu_under_the_key_harbor_treats_as_primary():
    """`reward` is what Harbor reports as the trial's reward.

    `_parse_reward_text` wraps a bare float as `{"reward": ...}`, so a JSON
    reward file has to use the same name for the figure that means the same
    thing. SketchBLEU's four components travel beside it under their own keys,
    which is what lets a result say *why* a repository scored what it did.
    """
    source = grade_source()
    assert '"reward": result["codebleu"]' in source
    assert 'rewards.update({name: result[name] for name in COMPONENTS})' in source
    assert (
        '"ngram_match_score",\n    "weighted_ngram_match_score",\n'
        '    "syntax_match_score",\n    "dataflow_match_score",' in source
    )


def test_a_workspace_the_metric_cannot_score_is_a_zero_and_not_a_crash():
    """The ordinary outcome that would otherwise cost a trial its whole price.

    `calc_repobleu` builds a sparse matrix over pairs of functions with
    matching dataflow. When the generated tree has no functions, that matrix
    has zero-sized index arrays and scipy raises before the metric reaches its
    own division — which is not an exotic case here but a pipeline whose first
    phase failed. The grader has to write a reward file anyway, and has to
    distinguish scoring nothing from being unable to score.
    """
    source = grade_source()
    for reason in (
            '"no_workspace"',
            '"no_python_files"',
            '"unreadable_python_files"',
            '"no_functions"',
            '"metric_failed"',
    ):
        assert reason in source, reason
    assert '"reward": 0.0, "graded": 0' in source
    # Every one of them leaves through the one helper that writes the file, so
    # none of these paths can return without a reward: five call sites, plus
    # the definition.
    assert source.count("ungraded(") == 1 + 5
    # And the guard must not swallow a real score: `graded` is 1 on exactly one
    # path, the one that actually called the metric.
    assert source.count('"graded": 1') == 1


def test_the_metric_the_grader_calls_is_the_authors_own():
    """SketchBLEU is `calc_repobleu`, and the tokenizer is not the default.

    The package named `codebleu` on PyPI has no `calc_repobleu` at all; the
    function exists only in the CodeS authors' fork, which the image clones at
    the same commit this project pins for the tool under test. And
    `get_metric.py` passes a Python tokenizer rather than the whitespace split
    `calc_repobleu` defaults to, which changes both n-gram components.
    """
    source = grade_source()
    assert "from codebleu import calc_repobleu" in source
    assert "tokenizer=tokenize_code" in source

    commit = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["metric"]["commit"]
    for name in REPOSITORIES:
        dockerfile = (TASKS_DIR / name / "environment" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert commit in dockerfile, name
        # The two pins that are not cosmetic: see SOURCES.json for why each
        # upper bound exists.
        assert "'tree-sitter==0.21.3'" in dockerfile, name
        assert "'setuptools==75.8.0'" in dockerfile, name


def test_one_image_serves_every_task():
    """Nineteen task images that share every layer, or nineteen real builds.

    Nothing repository-specific belongs in the image: the specification arrives
    as Harbor's instruction and the reference arrives under `tests/` once the
    agent has finished. A Dockerfile that differed per task would rebuild the
    tree-sitter grammars nineteen times for no gain.
    """
    dockerfiles = {
        (TASKS_DIR / name / "environment" / "Dockerfile").read_bytes()
        for name in REPOSITORIES
    }
    assert len(dockerfiles) == 1


def test_every_task_carries_an_oracle_a_verifier_and_a_difficulty():
    """A task with no oracle cannot be checked, and this benchmark needs one.

    The ceiling is below 1.0 and differs per repository, so the oracle is not a
    formality here: it is the only way to learn what a given task's best
    attainable reward actually is.
    """
    tasks = sorted(path.name for path in TASKS_DIR.iterdir() if path.is_dir())
    assert tasks == sorted(REPOSITORIES)

    for name in REPOSITORIES:
        task_dir = TASKS_DIR / name
        for relative in (
                "task.toml",
                "instruction.md",
                "environment/Dockerfile",
                "environment/build_metric.py",
                "tests/test.sh",
                "tests/grade.py",
                "tests/metric.json",
                "solution/solve.sh",
        ):
            assert (task_dir / relative).is_file(), f"{name}/{relative}"

        # The oracle writes the reference where the verifier reads, and the
        # verifier is told where both of those are.
        metric = json.loads((task_dir / "tests" / "metric.json").read_text("utf-8"))
        assert metric["prediction_path"] == "/workspace"
        assert metric["reference_path"] == "/tests/reference"
        assert metric["repo"] == name
        assert metric["lang"] == "python"
        assert metric["weights"] == [0.25, 0.25, 0.25, 0.25]

        solve = (task_dir / "solution" / "solve.sh").read_text(encoding="utf-8")
        assert "/solution/reference/. /workspace/" in solve


def test_every_task_validates_against_harbors_own_task_model():
    """Harbor parses `task.toml` with pydantic and rejects what it dislikes.

    Worth its own check because one of these names contains a dot —
    `sketcheval/web.Monitor` — and Harbor's package-name pattern is strict
    enough that a task could be generated, committed and only rejected at
    launch.
    """
    for name in REPOSITORIES:
        raw = tomllib.loads(
            (TASKS_DIR / name / "task.toml").read_text(encoding="utf-8")
        )
        config = TaskConfig.model_validate(raw)
        assert config.task.name == f"sketcheval/{name}"
        assert config.metadata["difficulty"] in {"easy", "medium", "hard"}
        assert config.metadata["repo"] == name
        # Measured worst case for the verifier was 136 seconds; the budget is
        # an order of magnitude above it, and the agent budget matches
        # NL2RepoBench's so a CodeS configuration means one thing either way.
        assert config.verifier.timeout_sec == 1800
        assert config.agent.timeout_sec == 3600


def test_the_difficulty_rule_is_the_one_the_authors_stated():
    """`validation/repos/README.md`: >10 files or >2500 lines is Hard, and so on.

    Checked against the recorded counts rather than recomputed, so a generator
    that wrote the wrong level beside the right counts is caught.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_sketcheval_tasks

    for name in REPOSITORIES:
        raw = tomllib.loads(
            (TASKS_DIR / name / "task.toml").read_text(encoding="utf-8")
        )["metadata"]
        files, lines = raw["reference_python_files"], raw["reference_code_lines"]
        if files > 10 or lines > 2500:
            expected = "hard"
        elif files > 5 or lines > 500:
            expected = "medium"
        else:
            expected = "easy"
        assert raw["difficulty"] == expected, name

        contents = {
            str(path.relative_to(TASKS_DIR / name / "tests" / "reference")): (
                path.read_bytes()
            )
            for path in (TASKS_DIR / name / "tests" / "reference").rglob("*")
            if path.is_file()
        }
        level, counted_files, counted_lines = build_sketcheval_tasks.difficulty_of(
            contents
        )
        assert (level, counted_files, counted_lines) == (expected, files, lines), name


def test_the_generated_tree_carries_no_windows_line_endings():
    """The digest pins the commit, or it pins the checkout that generated it.

    The CodeS submodule is checked out with `core.autocrlf=true` on this
    machine, so 221 of the 228 source files carry CRLF on disk. The generator
    reads blobs with `git archive` under `core.autocrlf=false` for exactly this
    reason, and every file it writes itself uses `newline="\\n"`.
    """
    for name in REPOSITORIES:
        for path in (TASKS_DIR / name).rglob("*"):
            if path.is_file():
                assert b"\r\n" not in path.read_bytes(), path


def test_the_digest_is_the_one_the_launcher_will_record():
    """The generator prints what the launcher computes, or the pin means little."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_sketcheval_tasks

    assert build_sketcheval_tasks.tree_digest is local_tree_digest
    assert len(local_tree_digest(TASKS_DIR)) == 64
