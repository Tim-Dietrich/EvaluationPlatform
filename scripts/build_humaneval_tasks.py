"""Generate the HumanEval benchmark as Harbor task packages.

Every other benchmark in this project enters as a dependency: Harbor's
registry carries it as a digest-pinned dataset and `benchmark.py` resolves it.
HumanEval is the exception, because Harbor's registry does not carry it —
neither the public git registry nor the package registry has anything but
`humanevalfix` and `evoeval`, which are different benchmarks. So it is built
here instead, from its published source, by this script.

What that buys back is the property the registry was giving us. The two data
files under `data/` are pinned by digest and verified before anything is
written, the generated tree is itself hashed, and a run records that hash
beside its results — so a HumanEval result is traceable to an exact set of 164
tasks in the way an NL2RepoBench result is traceable to a dataset ref.

The tasks are generated rather than checked in. 164 packages of seven files
each is not source, and a directory of a thousand generated files invites
hand-edits that no digest would survive. Run this once before the first run:

    python scripts/build_humaneval_tasks.py

Grading is the part worth reading closely. It reproduces what the authors'
own `evaluate/all_evaluate.py` does, quirks included, because the experiment
asks whether this platform reproduces their published number — and a stricter
grader would answer a different question. See `_GRADE_PY` below.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
# The digest is computed by the same function the launcher uses, so that the
# figure printed here and the figure archived beside a job cannot drift apart.
# The path insert is for running this from a checkout that has not been
# installed; where it has been, the import resolves to the same module.
sys.path.insert(0, str(ROOT / "src"))
from evaluation_platform.benchmark import local_tree_digest as tree_digest  # noqa: E402

BENCHMARK_DIR = ROOT / "benchmarks" / "humaneval"
DATA_DIR = BENCHMARK_DIR / "data"
TASKS_DIR = BENCHMARK_DIR / "tasks"
SOURCES_PATH = DATA_DIR / "SOURCES.json"

# Where the agent finds the problem, and where the verifier looks for its
# answer. Both belong to the task rather than to any one solution: an arm that
# writes somewhere else scores zero for a reason the record explains.
WORKSPACE = "/workspace"
SOLUTION_FILENAME = "solution.py"
PROBLEM_PATH = "/task/problem.json"

# The authors evaluate with `timeout=10` per problem in `all_evaluate.py`.
EXECUTION_TIMEOUT_SEC = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=TASKS_DIR,
        help=f"Where to write the task packages (default: {TASKS_DIR}).",
    )
    arguments = parser.parse_args(argv)

    problems = _read_verified("HumanEval.jsonl.gz")
    extended = {
        record["task_id"]: record
        for record in _read_verified("HumanEval_test_case_ET.jsonl.gz")
    }

    missing = sorted(p["task_id"] for p in problems if p["task_id"] not in extended)
    if missing:
        print(
            f"{len(missing)} problem(s) have no HumanEval-ET test cases, "
            f"starting with {missing[0]}. The ET reward cannot be graded for "
            "them, so the generation is refused rather than recording a zero "
            "that means 'no cases' instead of 'failed'.",
            file=sys.stderr,
        )
        return 2

    output = arguments.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for problem in problems:
        _write_task(output, problem, extended[problem["task_id"]])

    print(f"Wrote {len(problems)} task(s) to {output}")
    print(f"Task tree digest: sha256:{tree_digest(output)}")
    return 0


def _read_verified(filename: str) -> list[dict[str, Any]]:
    """Read one pinned data file, refusing content that is not what was pinned."""
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["sources"]
    try:
        expected = next(s for s in sources if s["file"] == filename)
    except StopIteration:
        raise SystemExit(f"{filename} is not recorded in {SOURCES_PATH}.") from None

    raw = gzip.decompress((DATA_DIR / filename).read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"]:
        raise SystemExit(
            f"{filename} hashes to {digest}, but {SOURCES_PATH.name} pins "
            f"{expected['sha256']}. The benchmark is generated from pinned "
            "data or not at all."
        )
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _write_task(
        output: Path,
        problem: dict[str, Any],
        extended: dict[str, Any],
) -> None:
    task_dir = output / task_directory_name(problem["task_id"])
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()

    prompt = problem["prompt"]
    entry_point = problem["entry_point"]

    # What the agent may see: the problem, and nothing about how it is graded.
    # The hidden tests live under `tests/`, which Harbor uploads only once the
    # agent has finished.
    _write(
        task_dir / "environment" / "problem.json",
        json.dumps(
            {
                "task_id": problem["task_id"],
                "prompt": prompt,
                "entry_point": entry_point,
            },
            indent=2,
        ),
    )
    _write(task_dir / "environment" / "Dockerfile", _DOCKERFILE)

    # What the verifier needs, and the agent never sees. `preamble` is the part
    # of the prompt before its last `def`, which is what the authors prepend to
    # a generation before executing it; the prompt itself ends in an unclosed
    # signature and could not be concatenated whole.
    _write(
        task_dir / "tests" / "problem.json",
        json.dumps(
            {
                "task_id": problem["task_id"],
                "preamble": prompt[: prompt.rfind("def ")],
                "entry_point": entry_point,
                "test": problem["test"],
                "test_et": build_test_method(
                    extended["test_case_list"], "", extended["entry_point"]
                ),
                "timeout_sec": EXECUTION_TIMEOUT_SEC,
                "solution_path": f"{WORKSPACE}/{SOLUTION_FILENAME}",
            },
            indent=2,
        ),
    )
    _write(task_dir / "tests" / "grade.py", _GRADE_PY)
    _write(task_dir / "tests" / "test.sh", _TEST_SH)

    _write(task_dir / "instruction.md", _instruction(prompt, entry_point))
    _write(
        task_dir / "solution" / "solve.sh",
        _solve_sh(prompt + problem["canonical_solution"]),
    )
    _write(task_dir / "task.toml", _task_toml(problem["task_id"]))


def _write(path: Path, content: str) -> None:
    """Write one generated file with newlines that do not depend on the host.

    The tree is hashed, and a digest that changes between Windows and Linux
    would be a digest that pins nothing.
    """
    path.write_text(content, encoding="utf-8", newline="\n")


def task_directory_name(task_id: str) -> str:
    """`HumanEval/0` as a directory name.

    Harbor takes a local task's name from its directory, and that name is what
    `benchmark.task_names` filters on, so it is worth being predictable:
    `HumanEval/0` becomes `HumanEval_0`.
    """
    return task_id.replace("/", "_")


def build_test_method(
        test_list: list[str],
        test_imports: str,
        method_name: str,
) -> str:
    """`core.utils.build_test_method`, reproduced rather than imported.

    This is how the authors turn HumanEval-ET's list of assertions into a
    `check` function, and the ET reward is only comparable with their published
    figure if it is assembled the same way. It is copied here — including its
    discarding of `test_imports`, which the third assignment below overwrites —
    because the benchmark must not import from the solution it grades.
    `all_evaluate.py` passes no imports, so the discard changes nothing about
    what runs.
    """
    if test_imports:
        test_imports = "\n".join(test_imports)
        test_method = test_imports + "\n"
    else:
        test_method = ""
    test_method = "def check(" + method_name + "):\n"
    if len(test_list) == 0:
        return test_method + "\treturn True" + "\n"
    for test in test_list:
        test_method += "\t" + test + "\n"
    return test_method.strip("\n")


def _instruction(prompt: str, entry_point: str) -> str:
    """What an agent is told, for any solution that reads the instruction.

    Self-Collaboration does not: its HumanEval entry point builds its own
    requirement string from the prompt and the entry point, and the runner
    hands it those from `problem.json` so that the tool composes exactly what
    it composes upstream. This file is what makes the task answerable by an arm
    that has no such entry point, which is what keeps the benchmark neutral
    between them.
    """
    return f"""# HumanEval: implement `{entry_point}`

Implement the following Python function.

```python
{prompt.rstrip()}
```

## What to produce

Write the complete implementation to `{WORKSPACE}/{SOLUTION_FILENAME}`.

- The file must be self-contained and importable: it may include any imports
  and helper functions it needs.
- Define the function `{entry_point}` with exactly the signature above.
- Do not write tests into `{SOLUTION_FILENAME}`; the grading tests are hidden
  and are supplied separately.

The problem is also available as JSON at `{PROBLEM_PATH}`, with the keys
`task_id`, `prompt` and `entry_point`.
"""


def _solve_sh(reference_solution: str) -> str:
    """The oracle, so the task is demonstrably solvable as it is graded.

    Harbor runs this in place of an agent to check a task end to end. It is the
    canonical solution from the pinned data, written where the verifier looks,
    and a task whose oracle does not score 1.0 is a broken task rather than a
    hard one.
    """
    return f"""#!/bin/bash
set -euo pipefail

mkdir -p {WORKSPACE}
cat > {WORKSPACE}/{SOLUTION_FILENAME} <<'HUMANEVAL_REFERENCE_SOLUTION'
{reference_solution.rstrip()}
HUMANEVAL_REFERENCE_SOLUTION
"""


def _task_toml(task_id: str) -> str:
    return f"""schema_version = "1.4"

[task]
name = "humaneval/{task_directory_name(task_id)}"
description = "HumanEval {task_id}: implement one Python function from its signature and docstring."
keywords = ["code-generation", "humaneval", "python"]

[metadata]
category = "code-generation"
source = "HumanEval"
task_id = "{task_id}"
# Generated by scripts/build_humaneval_tasks.py from the digest-pinned data in
# benchmarks/humaneval/data. Do not edit a task by hand: regenerate it.
generated = true

[verifier]
# Each of the two graded programs is bounded at ten seconds by the grader
# itself, which is the authors' own figure. This is the outer bound on the
# verifier as a whole.
timeout_sec = 300

[agent]
timeout_sec = 1800

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
"""


# The task image. Deliberately plain: a Python interpreter, git for whatever an
# agent's own installation step needs, and the problem statement. No solution's
# dependencies are baked in here — those are installed by the arm that needs
# them, so the same image serves every arm.
_DOCKERFILE = """FROM python:3.11-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# The problem, in the form a runner can read. The hidden tests are not here:
# Harbor uploads `tests/` only after the agent has finished.
COPY problem.json /task/problem.json

RUN mkdir -p /workspace
WORKDIR /workspace
"""


# The verifier hook. Harbor copies `tests/` into the container after the agent
# finishes and runs this from the working directory.
_TEST_SH = """#!/bin/bash
# Grade the agent's solution.py exactly as the HumanEval authors do, and write
# both rewards for Harbor to record.
set -uo pipefail

mkdir -p /logs/verifier
python3 /tests/grade.py 2>&1 | tee /logs/verifier/grade.log
"""


# The grader, reproducing `evaluate/all_evaluate.py` and
# `evaluate/execute/_execution.py` from the tool under evaluation.
#
# Three details of the authors' pipeline are load-bearing, and all three are
# reproduced rather than improved on:
#
#   1. The graded program is `preamble + solution + test + check(entry_point)`,
#      where the preamble is the part of the HumanEval prompt before its last
#      `def`. The prompt is not prepended whole; it ends in an unclosed
#      signature.
#   2. The entry point is *re-derived from the generated code* by
#      `find_method_name`, not taken from the dataset. That is more permissive
#      than the reference harness: a model that renames the function is still
#      graded. Using the dataset's entry point instead would score this arm
#      below the published figure for a reason that has nothing to do with the
#      orchestrator under test.
#   3. Ten seconds per program.
#
# The authors sandbox with `reliability_guard()` inside a forked process. Here
# the whole task is already a disposable container, so a subprocess with a
# timeout is the equivalent, and the separate interpreter is what keeps one
# graded program from disturbing the other.
_GRADE_PY = '''"""Grade /workspace/solution.py against HumanEval and HumanEval-ET."""

import ast
import json
import pathlib
import subprocess
import sys
import tempfile


PROBLEM = json.loads(
    pathlib.Path("/tests/problem.json").read_text(encoding="utf-8")
)
# `reward.json`, singular. Harbor reads exactly two paths — `reward.txt` and
# `reward.json` — and its own task template's comment misnames the second as
# `rewards.json`, which is a trial that runs, grades, and is then failed for
# having no reward file. `tests/test_humaneval_benchmark.py` pins this name
# against Harbor's own constant.
REWARDS_PATH = pathlib.Path("/logs/verifier/reward.json")


def find_method_name(code, lang="python"):
    """`core.utils.find_method_name`, reproduced exactly.

    The last top-level function in the generated code, or the second to last
    when the last one is called `main`. The bare `except` is on purpose: the
    original returns None for code that does not parse, and a solution that
    does not parse has to fail as a wrong answer rather than as a crash here.
    """
    try:
        parsed = ast.parse(code)
        function_defs = [
            node for node in parsed.body if isinstance(node, ast.FunctionDef)
        ]
        if function_defs:
            if len(function_defs) == 1:
                method_name = function_defs[0].name
            else:
                method_name = (
                    function_defs[-1].name
                    if function_defs[-1].name != "main"
                    else function_defs[-2].name
                )
        else:
            method_name = None
    except:  # noqa: E722 - the original's own control flow.
        method_name = None

    return method_name


def run(program, timeout):
    """Execute one check program. Returns (passed, detail)."""
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "check_program.py"
        path.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            return False, "timed out"
    if result.returncode == 0:
        return True, "passed"
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    return False, f"failed: {detail[-1] if detail else 'no output'}"


def main():
    solution_path = pathlib.Path(PROBLEM["solution_path"])
    report = {"task_id": PROBLEM["task_id"], "solution_path": str(solution_path)}

    if not solution_path.exists():
        report["error"] = f"{solution_path} was not written"
        finish({"reward": 0.0, "humaneval_et": 0.0}, report)
        return

    solution = solution_path.read_text(encoding="utf-8", errors="replace")
    generation = PROBLEM["preamble"] + "\\n" + solution
    entry_point = find_method_name(generation) or "candidate"
    report["entry_point_graded"] = entry_point
    report["entry_point_declared"] = PROBLEM["entry_point"]

    rewards = {}
    for key, test in (("reward", "test"), ("humaneval_et", "test_et")):
        program = (
            generation + "\\n" + PROBLEM[test] + "\\n" + f"check({entry_point})"
        )
        passed, detail = run(program, PROBLEM["timeout_sec"])
        rewards[key] = 1.0 if passed else 0.0
        report[key] = detail

    finish(rewards, report)


def finish(rewards, report):
    REWARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARDS_PATH.write_text(json.dumps(rewards, indent=2), encoding="utf-8")
    print(json.dumps({**report, "rewards": rewards}, indent=2))


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    raise SystemExit(main())
