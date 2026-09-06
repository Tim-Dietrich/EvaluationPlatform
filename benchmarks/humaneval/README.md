# HumanEval

164 Python function-completion problems, and the one benchmark in this project
that is not a dependency.

Every other benchmark here is resolved from Harbor's registry and pinned to a
digest. Harbor's registry has no HumanEval: the git registry's 80 datasets and
the package registry between them offer `humanevalfix` (164 *repair* tasks from
HumanEvalPack) and `evoeval` (100 mutated problems), which are different
benchmarks measuring different things. So this one is built here, from
HumanEval's published data, by `scripts/build_humaneval_tasks.py`.

```bash
python scripts/build_humaneval_tasks.py
```

That writes 164 Harbor task packages to `tasks/`, which is generated and
git-ignored — regenerate it rather than editing it. The script prints a digest
over the tree; the launcher computes the same digest and archives it in
`jobs/<job>/benchmark.json`, which is what a registry `ref` does for every
other benchmark and what makes a result traceable to the tasks that produced
it.

## What it is generated from

`data/SOURCES.json` pins both inputs by the sha256 of their decompressed
contents, and the generator verifies them before writing anything.

| File | Origin |
| --- | --- |
| `HumanEval.jsonl.gz` | `openai/human-eval`, MIT. The canonical 164 problems. |
| `HumanEval_test_case_ET.jsonl.gz` | The copy in Self-collaboration-Code-Generation at `a6490a9d`, which is what its authors evaluate ET against. |

The ET cases are vendored rather than read from the tool at run time, so that
the benchmark does not depend on the solution it grades.

## What a task looks like

```
HumanEval_0/
  task.toml                     schema 1.4; 1 CPU, 2 GB, 30-minute agent budget
  instruction.md                the problem, and the contract: write /workspace/solution.py
  environment/Dockerfile        python:3.11-slim, plus git
  environment/problem.json      task_id, prompt, entry_point — and nothing about grading
  tests/problem.json            the hidden tests: HumanEval's and HumanEval-ET's
  tests/grade.py                the grader
  tests/test.sh                 the verifier hook Harbor runs
  solution/solve.sh             the canonical solution, so the task has an oracle
```

`tests/` is uploaded into the container by Harbor only after the agent has
finished, so nothing under it is visible while the task is being solved.

## How it is graded

Faithfully to the authors' own `evaluate/all_evaluate.py`, because the
experiment this benchmark exists for asks whether this platform reproduces
their published figure — and a grader that is stricter than theirs answers a
different question. The graded program is assembled exactly as
`evaluate/execute/_execution.py` assembles it:

```
preamble + solution.py + "\n" + test + "\n" + check(entry_point)
```

Three details are load-bearing and are reproduced rather than improved on:

1. **`preamble`, not the whole prompt.** A HumanEval prompt ends in an
   unclosed signature and cannot be concatenated whole; the authors prepend
   only the part before its last `def`.
2. **The entry point is re-derived from the generated code** by
   `find_method_name` — the last top-level function, or the second to last when
   the last is named `main` — rather than taken from the dataset. This is more
   permissive than the reference harness: a model that renames the function is
   still graded. Grading against the declared entry point instead would score
   an arm below the published figure for a reason unrelated to the orchestrator
   under test.
3. **Ten seconds per program**, the authors' own `timeout=10`.

Two rewards are written to `/logs/verifier/reward.json`. `reward` is HumanEval,
and it is what Harbor reports as the trial's reward; `humaneval_et` is
HumanEval-ET, which the paper reports alongside it. Both are binary, so pass@1
over a run is the mean.

The filename is `reward.json`, singular, and it is worth saying so because
Harbor's own task template misnames it. Its `tests/test.sh` comment offers
`/logs/verifier/rewards.json` for multiple rewards, but the only two paths
Harbor reads are `reward.txt` and `reward.json`. A task that follows the
comment runs to completion, calls the model, grades correctly, writes its
rewards — and is then failed for having no reward file, with the whole cost of
the trial already paid. `tests/test_humaneval_benchmark.py` pins the name
against Harbor's own `EnvironmentPaths` constant.

## A ceiling on the ET number

Six of the 164 canonical solutions fail HumanEval-ET, so **the highest
attainable ET score against this data is 158/164 = 96.3%**. This is a property
of the ET cases as the authors ship and use them, not of this harness — their
pipeline assembles the same `check` function and would hit the same errors.
Verified by running the generated grader against every canonical solution: all
164 pass HumanEval, and these six do not pass ET.

| Task | Cause |
| --- | --- |
| `HumanEval/2` | Strict float equality: `assert truncate_number(3.952) == 0.952` |
| `HumanEval/38` | `assert decode_cyclic(encoded_str) == str` — `encoded_str` is undefined |
| `HumanEval/44` | `assert change_base(x, x + 1) == str(x)` — `x` is undefined |
| `HumanEval/50` | `assert decode_shift(copy.deepcopy(encoded_str)) == str` — `copy` is not imported |
| `HumanEval/53` | `assert add(x, y) == x + y` — `x` and `y` are undefined |
| `HumanEval/151` | `assert double_the_difference(lst) == odd_sum` — `lst` is undefined |

Five of the six are prose examples from the problem docstrings that were
collected as though they were executable cases. The paper's reported ET figures
sit well below this ceiling, so it does not affect the comparison — but any ET
number from this benchmark should be read against 96.3% rather than 100%.

## Neutrality

The task packages privilege no solution. The image carries a Python
interpreter, git and the problem statement, and no arm's dependencies; the
contract is stated in `instruction.md`; and grading reads one file at one path.
Self-Collaboration happens to ignore the instruction — its HumanEval entry
point composes its own requirement string from `problem.json`'s raw prompt and
entry point, which is what keeps that composition the authors' — but any arm
that reads instructions can answer these tasks unchanged.
