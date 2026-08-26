# The baselines

Three code generation solutions are integrated, and each of them claims — in
its paper, in one form or another — to improve on prompting a model directly.
A table of the three can rank them against each other. It cannot say whether
the scaffolding or the model is doing the work, and that is the question the
research topic actually asks.

Two arms answer it, and they answer different halves of it.

| Arm | What it is | What it isolates |
| --- | --- | --- |
| **Single-Shot** | One request. The specification goes in, a repository comes back as text, a deterministic writer puts the files on disk. | Whether *any* process beats no process. |
| **Terminus 2** | Harbor's reference agent: one model, one shell, one loop. No roles, no plan, no review. | Whether *role structure* beats the same model iterating on its own. |

The second is the more searching of the two. "Multi-agent beats a single
prompt" is a claim from 2023 that nobody now disputes; the claim a reader will
want tested is that four named roles beat one agent with a terminal and the
same budget. A solution that clears Single-Shot but not Terminus has not shown
that its roles are what did the work.

## There is no unscaffolded option

It is worth being precise about what "just run an LLM" can mean here, because
it does not mean what it means on HumanEval.

NL2RepoBench grades files in `/workspace`, copied out by a tester sidecar. A
model emits text. Something has to turn one into the other, and that something
is scaffolding however small it is. On a single-function benchmark you avoid
this because the harness is a string substitution into a function body; on a
repository benchmark you cannot.

Harbor does not close the gap either. Its built-in agents are `nop`, `oracle`,
`terminus-2`, `dspy-rlm` and around thirty installed third-party CLI agents.
There is no raw-completion agent, so "Harbor runs a model against a benchmark"
means Terminus, which is a harness. Single-Shot is the minimum scaffolding that
makes a repository benchmark answerable at all, and it is written here rather
than borrowed so that its every part is on the record.

## What is held constant

The `benchmark` and `model` blocks of all five `math-verify-*.yaml`
configurations are identical, and a test asserts it. That now includes which of
the aggregator's servers may answer: providers serve different quantizations of
the same weights, so an unpinned comparison can run one arm at fp4 and another
at fp8. `model.routing` fixes it for every arm, and each run records the server
that actually answered — the README's *Which server answers* has the argument
in full. So is the reasoning
setting, everywhere it can be stated: `request_extra` carries the same
provider-level value to Single-Shot, CodeTeam, CodeS and Terminus, whether it
travels through the OpenAI client or through LiteLLM. Self-Collaboration is the
one documented exception — the revision under evaluation sends no reasoning
field at all on its tool-calling path, so stating one would mean patching the
tool.

## What is not held constant, and must therefore be reported

Cost. CodeS spends on the order of seventy requests on a task; Single-Shot
spends one. Comparing them at equal task count and unequal token spend is not
wrong, but presenting it as a single column is. Every arm records uncached
input, cached input, output and reasoning tokens through the same fields, so
report the arms as a **cost–quality frontier** rather than as a leaderboard.
The objection that a reader would otherwise raise is answered by the axis.

Terminus's turn budget is the same argument seen from the other side.
`max_turns: 75` is not a round number: Self-Collaboration's ceiling on this
benchmark is three Coder rounds of ten Analyst and fifteen Coder steps, and a
Terminus turn is one model call. Set far higher, the baseline gets an advantage
the solutions do not have; far lower, it is handicapped. Either way the number
belongs in the write-up.

One more difference is structural rather than chosen. Terminus runs on the
host and reaches the provider through LiteLLM, so what it spent is counted by
Harbor's own accounting rather than by `model_usage.py`. The figures mean the
same thing — tokens in, tokens out — but a different counter produced them.
Each run says so in its `resolved-setup.json`.

## Reading a Single-Shot result

A single reward figure cannot distinguish a baseline that did not know the
answer from one that ran out of room to write it down, and the difference
decides how much the other arms' gains are worth. Four fields in
`<trial>/agent/resolved-setup.json` settle it, and they are worth checking
before a low score is read as a weak baseline:

- **`truncated`** — the reply hit the token ceiling. The library is cut off
  mid-file and the score says nothing about the model. Raise `max_tokens`.
- **`stripped_root`** — the reply put every file under `workspace/`, and the
  runner wrote them one level up. This is a correction for the apparatus rather
  than a helping hand, and it is narrow: NL2RepoBench draws the project as a
  tree rooted at the directory the task mounts it in, so a model writing paths
  as *text* can reproduce that root as though it were part of the project. The
  three solutions never face it, because they write through tools into the
  working directory and the root never materializes. Left uncorrected it would
  score a correct library zero over a difference in emitting filenames.
- **`common_top_level_directory`** — every file arrived nested inside one
  *other* directory, the project's own name most likely, so the package is not
  where the tester looks. That is the model's choice rather than the
  benchmark's phrasing, so the runner records it and moves nothing: rewriting
  output would be the runner deciding what the model meant. If it turns out to
  be common it is a finding, not a bug to paper over.
- **`parse_warnings`** — a section that could not be read as a file, a file
  named twice, a fence left open.

The reply itself is kept verbatim at `<trial>/agent/single-shot/response.md`,
next to the exact prompt that produced it.

The prompt is not configurable, and that is deliberate. Its only content beyond
the benchmark's own specification is the contract for naming files. A control
whose wording is a knob gets tuned, and a tuned control is a fourth solution
rather than a baseline; its digest is recorded with every run so that a result
can be checked against the prompt behind it.

## Why NOP is available and Oracle is not

Two further arms would be free, and they are sanity controls rather than
comparisons: NOP leaves the workspace empty and establishes the reward floor,
and Oracle writes the reference solution and establishes the ceiling — an
Oracle run that does not score ~1.0 means the grading path is broken, which is
worth knowing before publishing numbers that depend on it.

NOP would work. `harbor.agents.nop:NopAgent` needs nothing from a task at
all, so the only work is a one-line entry in `AGENT_HYPERPARAMETERS` with an
empty schema — this platform rejects an agent module it does not know rather
than passing hyperparameters to something that would ignore them.

Oracle does not, and the reason is in how NL2RepoBench is built. Harbor's
`OracleAgent` uploads a task's `solution/` directory and runs its `solve.sh`.
An NL2RepoBench task as packaged for Harbor contains `instruction.md`,
`task.toml`, `environment/` and `tests/test.sh` — and no solution of any kind.
The reference implementation is not distributed with the task; what the
benchmark ships is the specification and the hidden tests that grade an attempt
at it. Oracle would fail with `Solution script not found` on every task.

The ceiling is therefore not available from this benchmark, and a run of it
should say so rather than leave the absence to be noticed. What can stand in
its place is weaker but real: a benchmark whose tasks are derived from existing
libraries has a known-good implementation upstream, and installing that library
into the tester and running its tests would establish the same ceiling outside
Harbor's agent path. That is a separate piece of work, and worth it only if a
result turns out to hinge on where the ceiling is.
