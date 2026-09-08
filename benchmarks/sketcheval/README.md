# SketchEval

Nineteen Python repositories, generated whole from their READMEs and scored
against the real thing by similarity. It is the benchmark the CodeS paper
introduced, and the second benchmark in this project that is built here rather
than depended on.

Every benchmark resolved from Harbor's registry is pinned to a digest. Harbor's
registry has no SketchEval, and neither does anywhere else: it is not published
as a dataset at all. It exists as a directory inside the CodeS repository —
`validation/cleaned_repos/` — which this project already pins as a submodule,
because CodeS is one of the solutions under test. So the tasks are generated
from that commit by `scripts/build_sketcheval_tasks.py`:

```bash
python scripts/build_sketcheval_tasks.py
```

That writes 19 Harbor task packages to `tasks/`, which is generated and
git-ignored — regenerate it rather than editing it. The script prints a digest
over the tree; the launcher computes the same digest and archives it in
`jobs/<job>/benchmark.json`, which is what a registry `ref` does for every
other benchmark.

Generating a benchmark rather than depending on one is the exception and should
stay one. It is worth the exception twice now for the same reason: the
alternative was publishing somebody else's benchmark to a shared registry under
our own account.

## Read this before reading any number from it

**The reward is a similarity score, not a passing fraction.** Nothing is
executed. There are no hidden tests. A SketchEval reward and an NL2RepoBench
reward are different quantities — one is how much a repository *resembles* a
reference, the other is how many of its tests *pass* — and a figure that puts
them on one axis is measuring nothing.

**The attainable ceiling is below 1.0, and differs per repository.** See the
table below. Read a reward against its repository's own ceiling.

**This is CodeS's own benchmark.** SketchEval was introduced by the CodeS paper
and ships inside the CodeS repository. An arm evaluated here is on ground its
authors chose, which is worth stating whenever the result sits beside one from
NL2RepoBench, which belongs to nobody in this comparison.

## What it is generated from

`data/SOURCES.json` pins the source by the sha256 of the blobs at one commit,
and the generator verifies it before writing anything.

| | |
| --- | --- |
| Repository | `https://github.com/NL2Code/CodeS.git` |
| Commit | `0b624ab4ef22b0d9d223f274a986eb27fe090c88` |
| Path | `validation/cleaned_repos` |
| Contents | 228 files, 1 411 079 bytes, 19 repositories |

Read with `git archive` under `core.autocrlf=false` and `core.eol=lf`, which
reproduces the stored blob bytes exactly. The working tree is deliberately not
read: this checkout has `core.autocrlf=true`, so 221 of the 228 files carry
CRLF on disk, and a benchmark generated from them would pin the checkout rather
than the commit.

Only the Python half of the paper's benchmark is here, because only the Python
half is in the repository. `validation/multilingual-repos/` holds a table of
Java, C++ and Go project links and no code.

## What a task looks like

```
epubhv/
  task.toml                     schema 1.4; 1 CPU, 4 GB, 60-minute agent budget
  instruction.md                the repository's README.md, byte for byte
  environment/Dockerfile        python:3.11-slim, git, and the metric
  environment/build_metric.py   builds codebleu's parser from vendored grammars
  tests/reference/              the reference repository — the answer
  tests/metric.json             which repo, which weights, which paths
  tests/grade.py                the grader
  tests/test.sh                 the verifier hook Harbor runs
  solution/reference/           the same tree again, for the oracle
  solution/solve.sh             copies it to /workspace
```

`tests/` is uploaded into the container by Harbor only after the agent has
finished, and `solution/` only when the oracle agent runs, so neither copy of
the reference is visible while the task is being solved. The image is the third
path and the only one built beforehand; it carries a Dockerfile and a build
script and nothing else, which
`tests/test_sketcheval_benchmark.py::test_the_reference_repository_reaches_no_path_the_agent_can_read`
holds.

Every task's Dockerfile is byte-identical, so Docker builds the image once and
the other eighteen tasks hit the layer cache.

### The instruction is the README, and nothing is appended to it

CodeS's own driver opens `README.md` and interpolates it as `{readme}` into all
three of its phases. Anything appended here — a "write your repository to
`/workspace`" contract, say — would become a paragraph inside a prompt the
paper's numbers were not produced with. So the instruction is the README
verbatim.

What that costs is stated plainly: the task never says where to write. Every
arm integrated in this project already writes its repository to `/workspace`,
so all of them can answer these tasks unchanged, but an arm that relies on
being told would need the contract added — and adding it would change what
CodeS is being asked. If that trade ever needs making differently, make it in
the generator and say so here.

## How it is graded

SketchBLEU, which is `calc_repobleu` in the CodeS authors' fork of `codebleu`.
The package named `codebleu` on PyPI is a different implementation and has no
such function, so the fork is what the image installs — cloned at the same
commit this project pins for the tool under test, so the metric and the tool
cannot drift apart.

It stacks every `.py` file under the generated tree and every `.py` file under
the reference, and returns a quarter each of:

| Component | What it compares |
| --- | --- |
| `ngram_match_score` | BLEU over the stacked source |
| `weighted_ngram_match_score` | the same, weighting Python keywords 5× |
| `syntax_match_score` | shared AST subtrees, as s-expressions |
| `dataflow_match_score` | per-function dataflow graphs, matched pairwise by the Hungarian algorithm |

Called the way `batch_eval/get_metric.py` calls it, including its Python
tokenizer — not the whitespace split `calc_repobleu` defaults to, which changes
both n-gram components.

Two build pins are load-bearing and are not cosmetic:

- **`tree-sitter==0.21.3`.** The fork calls `Language.build_library` and the
  two-argument `Language(so_path, name)`, both removed in 0.22.
- **`setuptools==75.8.0`.** `build_library` constructs its compiler with
  `new_compiler()` and never calls `customize_compiler()`, so the link flags
  are whatever `UnixCCompiler`'s class defaults carry. setuptools 79
  restructured that module and dropped `-shared` from the C++ linker default,
  which links the nine grammars as an executable and fails on a missing `main`.

The parser is built from the grammar sources the pinned commit carries, rather
than by the fork's `setup.py`, which deletes them and re-downloads nine
archives from GitHub. Same compilation, pinned inputs.

### A workspace the metric cannot score

`calc_repobleu` assumes both trees contain functions with extractable dataflow.
When the generated tree has none, it fails inside scipy with `cannot infer
dimensions from zero sized index arrays` — before it reaches its own division.

That is not an exotic case here. It is a run whose first phase failed, or one
that wrote files with no functions in them, which is an ordinary outcome for a
pipeline whose length the model chooses. Unguarded it costs a trial the whole
price of its model calls and then records no reward at all. So the grader
checks first and writes `0.0` with a reason:

| `reason` | |
| --- | --- |
| `no_workspace` | `/workspace` does not exist |
| `no_python_files` | nothing under it ends in `.py` |
| `unreadable_python_files` | a generated file could not be decoded |
| `no_functions` | `.py` files, none containing a function |
| `metric_failed` | anything else, with the traceback attached |

`graded` — `1` or `0` — separates "scored nothing" from "could not be scored",
which are different findings and would otherwise look identical in the results.
The metric itself is not modified.

### Two files, because one of them may hold only numbers

Harbor parses the whole of `reward.json` into `VerifierResult.rewards`, typed
`dict[str, float | int]`. A single string anywhere in that file fails the trial
in pydantic — after every model call has been paid for, and with a message
about float parsing that never names the repository it was grading. The first
SketchEval trial died exactly that way, on a `"repo"` field.

So the grader writes two files:

| | |
| --- | --- |
| `/logs/verifier/reward.json` | numbers only: `reward`, `graded`, the four components, and the two counts |
| `/logs/verifier/sketchbleu.json` | the same, plus `repo`, `reason` and `detail` |

`write()` filters non-numbers out of the rewards rather than trusting the call
sites, so a later edit that adds a helpful string costs nothing instead of
costing a run. `graded` travels as `1`/`0` rather than `true`/`false` for the
same reason: pydantic is never asked to read a boolean as a number.

## The ceiling is below 1.0

Scoring a reference tree against **itself** returns 1.0 for the first three
components and less than that for dataflow. The cause is exact: tree-sitter
extracts no dataflow at all from some functions, those pairs score zero, they
drop out of the sparse matrix, and the assignment leaves them unmatched. The
shortfall is an integer number of such functions every time — epubhv's 0.9429
is 33/35, kanban-python's 0.9320 is 96/103.

This is a property of the reference trees and of the metric, not of any arm.
Their pipelines would hit the same thing. **Read every reward against the
ceiling for its repository**, not against 1.0.

Measured on one CPU with 2 GB, reference against reference:

| Repository | Difficulty | Functions | Ceiling | Dataflow | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| CVE-2023-44487 | easy | 4 | **1.0000** | 1.0000 | 0.1 |
| EVM_inscription | easy | 1 | **1.0000** | 1.0000 | 0.0 |
| EasyLiterature | hard | 51 | **0.9804** | 0.9216 | 5.9 |
| django-tui | medium | 37 | **0.9932** | 0.9730 | 3.6 |
| easier-docker | medium | 17 | **1.0000** | 1.0000 | 0.3 |
| epubhv | medium | 35 | **0.9857** | 0.9429 | 3.0 |
| every-breath-you-take | medium | 56 | **0.9955** | 0.9821 | 6.9 |
| fastui-chat | medium | 9 | **1.0000** | 1.0000 | 0.2 |
| flameshow | hard | 135 | **0.9796** | 0.9185 | 19.9 |
| kanban-python | medium | 103 | **0.9830** | 0.9320 | 13.5 |
| libgen_to_txt | medium | 27 | **1.0000** | 1.0000 | 1.1 |
| mactop | hard | 221 | **0.9955** | 0.9819 | 51.3 |
| pitch-visualizer | easy | 9 | **1.0000** | 1.0000 | 0.2 |
| pygraft | hard | 118 | **0.9958** | 0.9831 | 36.1 |
| pyobd | hard | 268 | **0.9879** | 0.9515 | 129.4 |
| sim-web-visualizer | hard | 240 | **0.9979** | 0.9917 | 135.6 |
| smol-podcaster | easy | 10 | **1.0000** | 1.0000 | 0.2 |
| van-gonography | medium | 13 | **0.9808** | 0.9231 | 1.0 |
| web.Monitor | easy | 9 | **1.0000** | 1.0000 | 0.3 |

Eight of the nineteen do reach 1.0; the lowest ceiling is flameshow's 0.9796.

Two caveats on this table. It is a *self*-comparison — the same directory
passed as both reference and prediction — which makes the syntax component
exactly 1.0. A real oracle run passes two different paths, and one
s-expression, the whole-tree one, then depends on `os.walk` enumeration order:
running the oracle for epubhv through the task package gives `syntax` 0.999814
rather than 1.0, which moves the reward by 5e-5. Second, the figures are the
*oracle's* cost; a generated repository with more functions than the reference
costs more, since dataflow match is
O(functions_reference × functions_generated).

The verifier budget is 1800 seconds against a measured worst case of 136.

## Difficulty

`task.toml` records a level per repository under the CodeS authors' own rule,
from `validation/repos/README.md`: Hard is more than ten Python files or more
than 2500 Python code lines, Medium is more than five files or more than 500
lines, everything else Easy. Code lines are read as non-blank and
non-comment. The split is 5 easy, 8 medium, 6 hard.

The rule is reproduced rather than read from a table because the repository
ships no table — the paper has one, the tree does not.

## Neutrality

The task packages privilege no solution. The image carries a Python
interpreter, git, and the metric the verifier needs; no arm's dependencies are
in it. Grading reads one directory at one path, `/workspace`, and every
integrated arm already writes its repository there. The one place this
benchmark is *not* neutral is the instruction, and that is stated above rather
than hidden: it is a README with no contract attached, which favours an arm
that expects a README.
