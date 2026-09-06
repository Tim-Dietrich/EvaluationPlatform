"""Generate SketchEval as Harbor task packages.

SketchEval is the repository-oriented benchmark the CodeS paper introduces, and
it is not published as a dataset anywhere — not in Harbor's registry, not on a
hub, not as an archive. It exists as a directory inside the CodeS repository:
nineteen Python projects under `validation/cleaned_repos/`, each one a
`README.md` that is the whole of the task's specification and a source tree
that is the reference a generated repository is scored against. This project
already pins that repository as a submodule, because it is the tool under test.
So the tasks are generated from the pinned commit, by this script:

    python scripts/build_sketcheval_tasks.py

This is the second generated benchmark here, after HumanEval, and the exception
is worth the same defence: the alternative was publishing somebody else's
benchmark to a shared registry under our own account. What a registry `ref`
pins for a dataset, a digest over the generated tree pins here; the launcher
computes it and archives it beside the job.

Three things about this benchmark differ from HumanEval's generator and are the
parts worth reading.

**The source is a commit, not a file.** `git archive` reads the blobs at the
pinned revision under `core.autocrlf=false`, which reproduces the stored bytes
exactly. The working tree is deliberately not read: this checkout has
`core.autocrlf=true`, so 221 of the 228 files carry CRLF on disk, and a
benchmark generated from them would pin the checkout rather than the commit.

**The reward is a similarity score, not a passing fraction.** Nothing is
executed and there are no hidden tests. SketchBLEU — `calc_repobleu` in the
CodeS authors' fork of codebleu — stacks every `.py` file in the generated tree
and every `.py` file in the reference, and returns a quarter each of n-gram
match, keyword-weighted n-gram match, AST subtree match and dataflow match. A
SketchEval reward and an NL2RepoBench reward are different quantities and do
not pool.

**The attainable ceiling is below 1.0, and differs per repository.** Scoring a
reference tree against itself returns 1.0 for the first three components and
less than that for dataflow, because tree-sitter extracts no dataflow at all
from some functions; those pairs score zero, drop out of the sparse matrix, and
the assignment leaves them unmatched. The shortfall is exactly an integer
number of such functions. `benchmarks/sketcheval/README.md` carries the
measured figures. This is a property of the reference trees, in the way
HumanEval-ET's 96.3% ceiling is a property of its test cases.

`instruction.md` is the repository's README and nothing else, which is what
CodeS's own driver feeds RepoSketcher. See the benchmark README for what that
costs an arm that has no such convention.
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The digest is computed by the same function the launcher uses, so the figure
# printed here and the figure archived beside a job cannot drift apart.
sys.path.insert(0, str(ROOT / "src"))
from evaluation_platform.benchmark import local_tree_digest as tree_digest  # noqa: E402

BENCHMARK_DIR = ROOT / "benchmarks" / "sketcheval"
DATA_DIR = BENCHMARK_DIR / "data"
TASKS_DIR = BENCHMARK_DIR / "tasks"
SOURCES_PATH = DATA_DIR / "SOURCES.json"
SUBMODULE = ROOT / "code_generation" / "CodeS"

# Where the agent leaves the repository it generated, and where the verifier
# looks for it. Both belong to the task rather than to any one solution.
WORKSPACE = "/workspace"
REFERENCE_IN_TESTS = "/tests/reference"
REFERENCE_IN_SOLUTION = "/solution/reference"
METRIC_ROOT = "/metric"

# SketchBLEU's own defaults, stated rather than inherited so that the weights a
# run scored with are in the task package and in the digest over it.
LANGUAGE = "python"
WEIGHTS = (0.25, 0.25, 0.25, 0.25)

# Measured: reference against reference, one CPU and 2 GB, worst of nineteen
# was 136 seconds (sim-web-visualizer, 240 functions). Dataflow match is
# O(functions_reference x functions_generated) tree-sitter parses, so a
# generated repository larger than the reference costs more than the oracle
# does; this is that worst case with an order of magnitude of headroom.
VERIFIER_TIMEOUT_SEC = 1800
# NL2RepoBench's own figure, so that the CodeS budgets in an experiment
# configuration mean the same thing against either benchmark.
AGENT_TIMEOUT_SEC = 3600


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=TASKS_DIR,
        help=f"Where to write the task packages (default: {TASKS_DIR}).",
    )
    arguments = parser.parse_args(argv)

    source = _read_sources()
    files = _read_verified(source)
    repositories = _group_by_repository(files, source["path"])

    output = arguments.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name, contents in sorted(repositories.items()):
        _write_task(output, name, contents, source["commit"])

    print(f"Wrote {len(repositories)} task(s) to {output}")
    print(f"Task tree digest: sha256:{tree_digest(output)}")
    return 0


def _read_sources() -> dict:
    document = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return document["sources"][0]


def _read_verified(source: dict) -> dict[str, bytes]:
    """Read the reference trees at the pinned commit, refusing anything else.

    `core.autocrlf=false` and `core.eol=lf` are passed explicitly because
    `git archive` runs blobs through the same conversion a checkout does. On a
    Windows clone with `autocrlf` on, the default output differs from the
    stored bytes and hashes to something no other machine would reproduce.
    """
    commit = source["commit"]
    if not SUBMODULE.is_dir():
        raise SystemExit(
            f"No CodeS checkout at {SUBMODULE}. SketchEval is generated from "
            "the submodule; run 'git submodule update --init "
            "code_generation/CodeS' first."
        )

    raw = _git(
        "-c", "core.autocrlf=false", "-c", "core.eol=lf",
        "archive", commit, source["path"],
    )
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                handle = archive.extractfile(member)
                assert handle is not None
                files[member.name] = handle.read()

    digest = _digest_of(files)
    if digest != source["sha256"]:
        raise SystemExit(
            f"{source['path']} at {commit[:12]} hashes to {digest}, but "
            f"{SOURCES_PATH.name} pins {source['sha256']}. The benchmark is "
            "generated from pinned data or not at all."
        )
    return files


def _git(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(SUBMODULE), *arguments],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(
            f"git {' '.join(arguments)} failed in {SUBMODULE}: {detail}"
        ) from error
    return completed.stdout


def _digest_of(files: dict[str, bytes]) -> str:
    """The same accumulation `local_tree_digest` performs, over a mapping.

    Path, then length, then bytes, in sorted order. Kept identical to the
    launcher's function so that "the digest" means one thing in this project
    whether it is taken over a directory or over what a commit holds.
    """
    accumulator = hashlib.sha256()
    for name in sorted(files):
        accumulator.update(name.encode("utf-8"))
        accumulator.update(str(len(files[name])).encode("utf-8"))
        accumulator.update(files[name])
    return accumulator.hexdigest()


def _group_by_repository(
        files: dict[str, bytes],
        prefix: str,
) -> dict[str, dict[str, bytes]]:
    """Split the flat archive into one mapping per repository.

    A repository with no README is skipped, which is the tool's own rule: both
    of its drivers open `README.md` and `continue` past a repository that has
    none, because the README is the entire input to the first phase.
    """
    grouped: dict[str, dict[str, bytes]] = {}
    for name, content in files.items():
        relative = name[len(prefix):].lstrip("/")
        if "/" not in relative:
            continue
        repository, _, path = relative.partition("/")
        grouped.setdefault(repository, {})[path] = content

    without_readme = sorted(n for n, c in grouped.items() if "README.md" not in c)
    for name in without_readme:
        print(
            f"Skipping {name}: no README.md, which is the whole of the task's "
            "specification.",
            file=sys.stderr,
        )
        del grouped[name]
    return grouped


def _write_task(
        output: Path,
        repository: str,
        contents: dict[str, bytes],
        commit: str,
) -> None:
    task_dir = output / repository
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()

    # What the agent is given: the README, and nothing about how it is graded.
    # Verbatim, because CodeS's own driver reads exactly this file and quotes
    # it into every one of its three phases; a contract paragraph appended here
    # would be a paragraph inside the prompt the paper's numbers were not
    # produced with.
    _write_bytes(task_dir / "instruction.md", contents["README.md"])
    _write(task_dir / "environment" / "Dockerfile", _dockerfile(commit))
    _write(task_dir / "environment" / "build_metric.py", _BUILD_METRIC_PY)

    # The reference, which the agent never sees: Harbor uploads `tests/` only
    # once the agent has finished. The whole tree goes in rather than only the
    # `.py` files the metric reads, so that "the reference" means the
    # repository the CodeS authors vendored.
    for path, content in sorted(contents.items()):
        _write_bytes(task_dir / "tests" / "reference" / path, content)
    _write(task_dir / "tests" / "metric.json", _metric_json(repository))
    _write(task_dir / "tests" / "grade.py", _GRADE_PY)
    _write(task_dir / "tests" / "test.sh", _TEST_SH)

    # The oracle, so the task has a demonstrable ceiling. `solution/` is
    # uploaded by Harbor's oracle agent and by nothing else, so the second copy
    # of the reference is not a leak: a real trial never sees it.
    for path, content in sorted(contents.items()):
        _write_bytes(task_dir / "solution" / "reference" / path, content)
    _write(task_dir / "solution" / "solve.sh", _SOLVE_SH)

    _write(task_dir / "task.toml", _task_toml(repository, contents, commit))


def _write(path: Path, content: str) -> None:
    """Write one generated text file with newlines that do not depend on the host.

    The tree is hashed, and a digest that changes between Windows and Linux
    would be a digest that pins nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_bytes(path: Path, content: bytes) -> None:
    """Write one file of the reference exactly as the commit stores it.

    Byte-for-byte: this is somebody else's source code, it is what the metric
    scores against, and re-encoding it would change the score.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def difficulty_of(contents: dict[str, bytes]) -> tuple[str, int, int]:
    """The level, file count and code-line count for one repository.

    The rule is the CodeS authors' own, from `validation/repos/README.md`:
    Hard is more than ten Python files or more than 2500 Python code lines,
    Medium is more than five files or more than 500 lines, and everything else
    is Easy. It is reproduced rather than read from a table because the
    repository ships no such table — the paper has one, the tree does not.

    "Code lines" is read as non-blank, non-comment, which is the reading that
    reproduces the paper's three-way split.
    """
    python_files = sorted(name for name in contents if name.endswith(".py"))
    code_lines = 0
    for name in python_files:
        text = contents[name].decode("utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines += 1

    if len(python_files) > 10 or code_lines > 2500:
        level = "hard"
    elif len(python_files) > 5 or code_lines > 500:
        level = "medium"
    else:
        level = "easy"
    return level, len(python_files), code_lines


def _metric_json(repository: str) -> str:
    """What the grader is told, and the only thing that differs between tasks."""
    alpha, beta, gamma, theta = WEIGHTS
    return json.dumps(
        {
            "repo": repository,
            "lang": LANGUAGE,
            "weights": [alpha, beta, gamma, theta],
            "reference_path": REFERENCE_IN_TESTS,
            "prediction_path": WORKSPACE,
        },
        indent=2,
    )


def _dockerfile(commit: str) -> str:
    """One image, byte-identical for every task, so Docker builds it once.

    Nothing repository-specific is in it: the specification arrives as Harbor's
    instruction and the reference arrives under `tests/` after the agent has
    finished. That is what lets nineteen task images share every layer.
    """
    return f"""FROM python:3.11-slim

# `git` for the metric checkout below and for the agent's own installer;
# `build-essential` because tree-sitter compiles the grammars into a shared
# object at image-build time.
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        git ca-certificates build-essential \\
    && rm -rf /var/lib/apt/lists/*

# The container has no credentials and nobody to ask for them.
ENV GIT_TERMINAL_PROMPT=0

# SketchBLEU is `calc_repobleu`, which exists only in the CodeS authors' fork
# of codebleu — the package of that name on PyPI has no such function. It is
# taken from the same commit this project pins for the tool under test, so the
# metric and the tool cannot drift apart. The clone is partial and sparse: the
# repository carries two hundred vendored projects that no run reads.
RUN git clone --quiet --filter=blob:none --sparse \\
        https://github.com/NL2Code/CodeS.git /tmp/codes \\
    && git -C /tmp/codes sparse-checkout set \\
        validation/evaluation_scripts/codebleu \\
    && git -C /tmp/codes checkout --quiet {commit} \\
    && mkdir -p {METRIC_ROOT} \\
    && mv /tmp/codes/validation/evaluation_scripts/codebleu/codebleu \\
        {METRIC_ROOT}/codebleu \\
    && mv /tmp/codes/validation/evaluation_scripts/codebleu/tree_sitter_languages \\
        {METRIC_ROOT}/tree_sitter_languages \\
    && rm -rf /tmp/codes

# `tree-sitter` below 0.22: the fork calls `Language.build_library` and the
# two-argument `Language(so_path, name)`, both removed there.
#
# `setuptools` exactly 75.8.0, and not a later one: `build_library` constructs
# its compiler with `new_compiler()` and never calls `customize_compiler()`, so
# the link flags are whatever `UnixCCompiler`'s class defaults carry.
# setuptools 79 restructured that module and dropped `-shared` from the C++
# linker default, which links the nine grammars as an executable and fails on a
# missing `main`.
RUN python -m pip install --no-cache-dir \\
    'tree-sitter==0.21.3' \\
    'scipy==1.14.1' \\
    'numpy==2.1.3' \\
    'setuptools==75.8.0'

# Build the grammars the fork's own `setup.py` builds, from the sources the
# pinned commit carries rather than from the nine GitHub archives that
# `setup.py` re-downloads. The grammars are then dropped: the shared object is
# what the metric loads, and the sources are seventy megabytes.
COPY build_metric.py {METRIC_ROOT}/build_metric.py
RUN python {METRIC_ROOT}/build_metric.py \\
    && rm -rf {METRIC_ROOT}/tree_sitter_languages

# Deliberately not on `PYTHONPATH`: `tests/test.sh` puts it there for the
# verifier, so the metric is not sitting on the agent's import path during the
# phase where the agent is the one running.
RUN mkdir -p {WORKSPACE}
WORKDIR {WORKSPACE}
"""


_BUILD_METRIC_PY = '''"""Build codebleu's `my-languages.so` from the grammars the commit carries.

This is `build_tree_sitter_languages` out of the fork's own `setup.py` with the
download removed. That function deletes `tree_sitter_languages/` and refetches
nine grammar archives from GitHub before compiling them — but the sources are
already tracked in the CodeS repository at the pinned commit, so compiling
those pins the parser to the commit instead of to whatever those tags serve
today. Nothing about the metric changes; only where the grammar sources come
from.
"""

import time
from pathlib import Path

from tree_sitter import Language

ROOT = Path(__file__).parent
GRAMMARS = ROOT / "tree_sitter_languages"
TARGET = ROOT / "codebleu" / "my-languages.so"

folders = sorted(path for path in GRAMMARS.iterdir() if path.is_dir())
if not folders:
    raise SystemExit(f"No grammar sources under {GRAMMARS}.")

print(f"Building {TARGET} from {len(folders)} grammars.")
started = time.monotonic()
Language.build_library(str(TARGET), [str(folder) for folder in folders])
print(
    f"Built in {time.monotonic() - started:.1f}s "
    f"({TARGET.stat().st_size / 1e6:.1f} MB)"
)
'''


_GRADE_PY = '''"""Score the repository in /workspace against the reference with SketchBLEU.

The metric is the CodeS authors' `calc_repobleu`, called the way their own
`batch_eval/get_metric.py` calls it — including its tokenizer, which is not the
default whitespace split and which changes the two n-gram components.

What is added around it is a guard. `calc_repobleu` assumes both trees contain
functions with extractable dataflow; when the generated tree has none, it fails
inside scipy with `cannot infer dimensions from zero sized index arrays`,
before it reaches its own division. That is not a rare case here — it is a run
that wrote nothing, or wrote files with no functions in them, which is an
ordinary outcome for a pipeline whose first phase can fail. Unguarded it costs
a trial the whole price of its model calls and then records no reward at all,
so the reward file says 0.0 and why. The metric itself is not touched.
"""

import json
import pathlib
import traceback
from io import BytesIO
from tokenize import tokenize

CONFIG = json.loads(pathlib.Path("/tests/metric.json").read_text(encoding="utf-8"))
# `reward.json`, singular. Harbor reads exactly two paths — `reward.txt` and
# `reward.json` — and its own task template's comment misnames the second as
# `rewards.json`, which is a trial that runs, grades, and is then failed for
# having no reward file. `tests/test_sketcheval_benchmark.py` pins this name
# against Harbor's own constant.
#
# Numbers only. Harbor parses this whole file into
# `VerifierResult.rewards`, typed `dict[str, float | int]`, so one string
# anywhere in it fails the trial in pydantic — after the model calls have been
# paid for, and with a message about float parsing that says nothing about the
# repository it was grading. Which repository, and why a score is zero, are
# facts about the grading rather than rewards, so they go next door.
REWARDS_PATH = pathlib.Path("/logs/verifier/reward.json")
DIAGNOSTICS_PATH = pathlib.Path("/logs/verifier/sketchbleu.json")
REFERENCE = pathlib.Path(CONFIG["reference_path"])
PREDICTION = pathlib.Path(CONFIG["prediction_path"])

COMPONENTS = (
    "ngram_match_score",
    "weighted_ngram_match_score",
    "syntax_match_score",
    "dataflow_match_score",
)


def tokenize_code(code):
    """`batch_eval/get_metric.py`'s tokenizer, reproduced exactly.

    The bare `except` is the original's own control flow: source that does not
    tokenize yields whatever was collected before it stopped, rather than an
    error. A generated repository routinely contains such a file.
    """
    tokens = []
    try:
        for tok in tokenize(BytesIO(code.encode("utf-8")).readline):
            if tok.type == 57:  # NEWLINE
                tokens.append("\\n")
            elif tok.type == 58:  # INDENT
                tokens.append("    ")
            elif tok.type == 59:  # DEDENT
                pass
            else:
                tokens.append(tok.string)
    except:  # noqa: E722 - the original's own control flow.
        pass
    return tokens


def write(rewards, **diagnostics):
    """Write the numeric rewards Harbor reads, and the prose beside them.

    The filter is not defensive padding. It is the last thing standing between
    a future edit that adds a helpful string and a trial that dies in pydantic
    with its model calls already paid for, so a non-number is dropped from the
    rewards rather than allowed to fail the run — it is still recorded in full
    next door.
    """
    numbers = {
        name: value
        for name, value in rewards.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    REWARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARDS_PATH.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    DIAGNOSTICS_PATH.write_text(
        json.dumps({"repo": CONFIG["repo"], **diagnostics, **rewards}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"repo": CONFIG["repo"], **diagnostics, **rewards}, indent=2))


def ungraded(reason, detail=""):
    """A reward of 0.0, and the reason it is 0.0 rather than a score.

    `graded` separates "the generated repository scored nothing" from "the
    generated repository could not be scored", which are different findings and
    would otherwise look identical in the results. It is written as 0 or 1
    rather than as a boolean because it travels in the rewards, and pydantic is
    not asked to read `true` as a number.
    """
    write(
        {"reward": 0.0, "graded": 0} | {name: 0.0 for name in COMPONENTS},
        reason=reason,
        detail=detail,
    )


def main():
    from codebleu import calc_repobleu
    from codebleu.codebleu import (
        extract_functions,
        get_file_list,
        stack_source_code,
    )

    if not PREDICTION.is_dir():
        ungraded("no_workspace", f"{PREDICTION} does not exist.")
        return

    generated = get_file_list(PREDICTION, ".py")
    if not generated:
        ungraded("no_python_files", f"No .py files under {PREDICTION}.")
        return

    try:
        source = stack_source_code(generated)
    except (OSError, UnicodeDecodeError) as error:
        ungraded("unreadable_python_files", f"{type(error).__name__}: {error}")
        return

    functions = extract_functions(source)
    if not functions:
        ungraded(
            "no_functions",
            f"{len(generated)} .py file(s), none containing a function. "
            "SketchBLEU's dataflow component has nothing to match.",
        )
        return

    try:
        result = calc_repobleu(
            [REFERENCE],
            [PREDICTION],
            CONFIG["lang"],
            weights=tuple(CONFIG["weights"]),
            tokenizer=tokenize_code,
        )
    except Exception as error:  # noqa: BLE001 - any failure is a zero, with the reason.
        ungraded(
            "metric_failed",
            f"{type(error).__name__}: {error}\\n{traceback.format_exc()}",
        )
        return

    rewards = {
        "reward": result["codebleu"],
        "graded": 1,
        "generated_python_files": len(generated),
        "generated_functions": len(functions),
    }
    rewards.update({name: result[name] for name in COMPONENTS})
    write(rewards, reason="", detail="")


if __name__ == "__main__":
    main()
'''


_TEST_SH = """#!/bin/bash
# Score the repository the agent generated. `PYTHONPATH` is set here rather
# than in the image so the metric is not on the agent's import path while the
# agent is the one running.
set -uo pipefail

mkdir -p /logs/verifier
PYTHONPATH=/metric python3 /tests/grade.py 2>&1 | tee /logs/verifier/grade.log
"""


_SOLVE_SH = f"""#!/bin/bash
# The oracle: the reference repository, written where the verifier looks.
#
# Harbor uploads `solution/` only when the oracle agent runs, so this copy of
# the reference never reaches a real trial. What it establishes is the task's
# ceiling, which for this benchmark is not 1.0 — SketchBLEU's dataflow
# component leaves functions with no extractable dataflow unmatched, even
# against an identical tree.
set -euo pipefail

mkdir -p {WORKSPACE}
cp -a {REFERENCE_IN_SOLUTION}/. {WORKSPACE}/
"""


def _task_toml(
        repository: str,
        contents: dict[str, bytes],
        commit: str,
) -> str:
    level, python_files, code_lines = difficulty_of(contents)
    return f"""schema_version = "1.4"

[task]
name = "sketcheval/{repository}"
description = "SketchEval {repository}: generate the whole repository from its README, scored against the reference with SketchBLEU."
keywords = ["code-generation", "sketcheval", "nl2repo", "python"]

[metadata]
category = "code-generation"
source = "SketchEval"
repo = "{repository}"
# The CodeS authors' own size rule, from validation/repos/README.md.
difficulty = "{level}"
reference_python_files = {python_files}
reference_code_lines = {code_lines}
source_commit = "{commit}"
# Generated by scripts/build_sketcheval_tasks.py from the digest-pinned commit
# in benchmarks/sketcheval/data. Do not edit a task by hand: regenerate it.
generated = true

[verifier]
# Dataflow match is O(functions_reference x functions_generated) tree-sitter
# parses. Measured worst case across the nineteen references, scored against
# themselves on one CPU, was 136 seconds.
timeout_sec = {VERIFIER_TIMEOUT_SEC}

[agent]
# NL2RepoBench's own figure, so that a CodeS budget means the same thing
# against either benchmark.
timeout_sec = {AGENT_TIMEOUT_SEC}

[environment]
build_timeout_sec = 1800.0
cpus = 1
memory_mb = 4096
storage_mb = 8192
"""


if __name__ == "__main__":
    raise SystemExit(main())
