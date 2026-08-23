# Scientific Codegen Evaluation MVP

This repository runs code generation solutions against whole benchmarks using
Harbor as the orchestrator, and keeps the setup of every run on record so
results from different solutions can be compared fairly.

A benchmark enters the repository as a dependency, not as checked-in task
files. Harbor's registry carries NL2RepoBench as a digest-pinned dataset of 104
tasks; an experiment configuration names the dataset, the version, and which of
its tasks to run, and Harbor downloads and pins the rest. Scaling from one task
to the full benchmark is a filter in the configuration, not an integration
effort.

Most Harbor benchmarks need nothing beyond that. NL2RepoBench is the exception:
every one of its tasks names its tester image on a private mirror that needs GCP
credentials this project does not have, while the same images are public on
`ghcr.io/multimodal-art-projection/nl2repobench`. A configuration can therefore
declare an optional `image_mirror`, and before a run the launcher resolves
exactly the tasks the job will execute, reads the images they name, and makes
each one available locally under the name the task expects, pulled from its
public home; Docker Compose then uses the local image without contacting a
registry. One rule covers all 104 NL2RepoBench tasks, and the benchmark's own
files are never modified.

## Prerequisites

- Python 3.12 or newer
- Docker with Linux containers enabled
- An OpenRouter API key (or credentials for another OpenAI-compatible endpoint)

Create the project environment and install the pinned Harbor release:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Each code generation solution under evaluation is checked out beneath
`code_generation/` as a submodule, for reading and reference:

```powershell
git submodule update --init
```

A run does not use these working copies: the agent clones the revision named in
the experiment configuration into the task container, so a clone without them
still runs.

## Experiment configurations

`configs/` is the home for run setup. One file describes one experiment: the
benchmark and task selection, the model backend, the code generation solution
and the revision of it, and the hyperparameters its roles receive. Nothing that
changes what a run does lives outside it — `.env` holds credentials only.

Which hyperparameters a file may state depends on the solution it names.
`agent.import_path` selects it, and each solution declares its own set: a name
belonging to a different one, or to none, is rejected with the known names
listed rather than accepted and then ignored by an agent that has no use for
it.

Copy the tracked template and set `API_KEY` to your credential. The local
`.env` file is ignored by Git:

```powershell
Copy-Item .env.example .env
```

Run the default configuration:

```powershell
.venv\Scripts\python.exe main.py
```

Run a different one, or name the job directory yourself:

```powershell
.venv\Scripts\python.exe main.py --config configs/math-verify-self-collaboration.yaml
```

The launcher loads `.env` before starting Harbor so every setting, including
`PYTHONUTF8`, is in place. Values in a configuration may use `${VAR}` and
`${VAR:-default}`, which resolve against the environment; the shipped
configuration uses this for the model backend, so `MODEL`, `MODEL_PROVIDER`,
and `BASE_URL` in `.env` still override it without editing tracked files. A
configuration is validated before Docker starts: unknown keys, unknown
hyperparameters, an unset credential, and DeepSeek's `/api/v1` path are all
rejected with an explanation rather than a failed run. DeepSeek's official
OpenAI-compatible endpoint has no `/api/v1` path:

```dotenv
MODEL_PROVIDER=deepseek
MODEL=deepseek-v4-flash
BASE_URL=https://api.deepseek.com
```

`MODEL_PROVIDER` is Harbor's reporting label and is not sent to the API.
DeepSeek also accepts `https://api.deepseek.com/v1`. Free OpenRouter models can
be temporarily rate-limited even with a valid key. The adapter waits and retries
after the tool has exhausted its own attempts; if every one is throttled, the
job log reports the rate limit and the affected model explicitly.

## Benchmarks

The `benchmark` block of a configuration is what makes a run comparable:

```yaml
benchmark:
  dataset: nl2repobench/nl2repobench
  ref: sha256:b0d58e327ee30a6e6584bd4843a53db52a3442e230630369da095ec564542712
  task_names:
    - nl2repobench/math-verify
  image_mirror:
    - expects: us-docker.pkg.dev/.../nl2repobench/
      pull_from: ghcr.io/multimodal-art-projection/nl2repobench/
```

`image_mirror` is optional and benchmark-specific: it exists because
NL2RepoBench publishes references to images it cannot itself pull. A benchmark
whose images are reachable needs no rule, and omitting the key leaves the
preparation step to resolving and pinning tasks.

`ref` pins the benchmark version; a floating reference is resolved once at
launch and re-pinned, so every trial of a run sees one version. Omit
`task_names` to run the entire benchmark, or narrow it with glob patterns
(org-qualified), `exclude_task_names`, and `n_tasks` — Harbor's own dataset
filtering does the selecting, so the run and its preparation can never disagree
about scope. `configs/nl2repobench-self-collaboration.yaml` runs all 104 tasks;
`configs/math-verify-self-collaboration.yaml` is the same setup narrowed to one
task for iterating cheaply.

Comparing a further code generation solution means copying a configuration,
changing only the `agent` section, and leaving everything else byte-identical.
`configs/math-verify-codeteam.yaml` and `configs/math-verify-codes.yaml` are
that: the same task, the same pinned benchmark version, and the same model as
the Self-Collaboration configuration beside them, given to CodeTeam and to
CodeS instead. A test asserts that those blocks agree across all three, since
that agreement is the entire basis for comparing their results.

Integrating a solution that is not yet here is a larger job than copying a
configuration, and `docs/adding-a-solution.md` describes it: what the platform
requires of a tool, the five files an integration adds, and the mistakes the
earlier integrations made.

A configuration may instead point at a single local task directory with
`task: {path: ...}`, for a task authored by hand rather than taken from a
benchmark. Exactly one of `benchmark` and `task` is required.

## Running a whole benchmark

A NL2RepoBench task takes around six minutes, so 104 of them in sequence is
about ten hours. Harbor runs trials concurrently, and `run.n_concurrent_trials`
is how many at once:

```yaml
run:
  n_concurrent_trials: 4
  retry:
    max_retries: 2
    wait_multiplier: 2
    min_wait_sec: 5
    max_wait_sec: 120
```

Four concurrent trials turns ten hours into roughly two and a half. There are
two limits to weigh, and they are not the same limit:

- **`run.n_concurrent_trials`** is the machine's. Every task declares what it
  needs — NL2RepoBench asks for 2 CPUs and 8 GB per trial — and each trial runs
  both its agent container and its tester sidecar. Harbor passes the declared
  memory to Docker as a ceiling, so a trial that reaches it is killed and the
  task is scored as an error rather than merely slowed down. Before a run
  starts, the launcher reads what the selected tasks ask for, compares it with
  what Docker reports it has, and says so. On Windows and macOS that figure is
  the Docker VM's allocation, not the host's hardware, and raising it is often
  the cheapest way to run more tasks at once.
- **`agent.n_concurrent`** is the provider's. It caps how many of those trials
  may be calling the model at the same time, while container setup,
  installation, and verification stay fully parallel. Set it equal to
  `n_concurrent_trials` for no extra limit, and lower it to stay under a rate
  limit. It can never be the larger of the two; a configuration that makes it
  so is rejected before Docker starts.

Because tasks spend most of their time waiting on the model rather than on this
machine's processors, concurrency well above the core count still pays. Memory
is the ceiling that bites first.

`run.retry` is the other half of making a long run finish. Across a hundred
tasks a provider hiccup or a dropped connection is close to certain, and
without a retry the affected task is simply missing from the results. Harbor's
own exclusions still apply, so failures a retry cannot fix — a timeout, an
exhausted usage limit, a rejected credential — fail once rather than three
times.

Tester images are prepared concurrently too. Each NL2RepoBench image is around
two gigabytes and every task has its own, so the whole benchmark is a large
one-time download; pulling them one at a time would cost more than the run
itself. They are cached, so only the first run pays.

### Continuing an interrupted run

Harbor keeps every trial that already has a result, so a run that dies at task
80 of 104 costs the remaining tasks rather than all of them:

```powershell
.venv\Scripts\python.exe main.py --resume jobs\2026-08-21__16-48-46
```

The setup comes from the `experiment-config.yaml` archived beside the job, so a
resume continues the run that was configured rather than offering a chance to
change it — `--resume` cannot be combined with `--config` or `--job-name`. Any
tester images the remaining tasks need are made available again first, since a
resume may happen days later on a machine whose images have since been pruned.
Trials that were cancelled are discarded and run again; trials that finished,
including failed ones, are kept.

A job is resumable once Harbor has written its `config.json`. A run that failed
before that point — a configuration error, an unreachable image — has to be
launched again.

## The code generation solutions

Three are integrated. Each receives the task's natural-language specification
and an empty workspace, and each is driven by an adapter in
`src/evaluation_platform/` that installs the tool in the task container, hands
it the specification and the configured hyperparameters, and records what it
spent. No adapter changes how a method works.

### Self-Collaboration

A team of three: the Analyst localizes the work, the Coder writes it, and the
Tester runs the tests and reports failures back to the Coder for the next
round. The Tester runs *between* Coder rounds, so it needs both halves of its
setup to exist:

- `test_command` — what it runs in the agent's workspace. Without it the
  session degrades to Analyst followed by a single Coder pass.
- `max_rounds` greater than one — with a single round the loop ends before the
  Tester is ever reached, whatever the test command says.

The Tester only ever runs the code and tests the agent wrote itself. A task's
reference tests stay in its tester sidecar and are never visible to the agent.

### CodeTeam

A team of four kinds. `architects` Architects each propose a software design
sketch — the file tree, the public interfaces, the dependencies between files,
and how many Developers the plan needs — a CTO selects one and normalizes it,
the Developers implement the files they own under a dependency-aware scheduler,
and a QA agent tests the result and hands failures back for repair for up to
`max_qa_rounds` rounds.

Three keys are the ablations the paper reports, each isolating one component:

- `rag_enabled` — whether the Architects are grounded with design references
  retrieved from a corpus of public repositories. Off by default here, because
  the paper's vector backend downloads an embedding model at first use inside
  the task container; `rag_backend: lexical` grounds from the same corpus
  without that dependency, and the retrieval stack is installed only for a run
  that asks for it.
- `dynamic_developer_allocation` — whether the selected design decides the
  number of Developers and who owns which file, or the files are dealt
  round-robin to `fixed_developer_agents` of them.
- `git_coordination` — whether Developers propagate interface changes to each
  other as commits carrying a structured update reason.

QA writes its own throwaway tests, runs them in the workspace, and deletes
them before the repository is returned; as with Self-Collaboration's Tester,
the benchmark's reference tests stay in the sidecar and are never visible.

Unlike Self-Collaboration, CodeTeam has no bound of its own on what a task may
cost: the width of the architect search, the number of files the chosen design
names, and the QA rounds each cost what they cost. `max_wall_clock_seconds`
and `max_token_budget` are the bound. Reaching either stops the run and is
recorded as a failed agent run, and Harbor still grades whatever the workspace
holds at that point.

### CodeS

A pipeline rather than a team, and the only one of the three that is not a
conversation between agents. It writes the repository in three layers of
sketch: RepoSketcher proposes the file tree from the specification,
FileSketcher writes each Python file the tree names as signatures with empty
bodies, and SketchFiller implements one function per request from that file's
sketch and the sketches of the files it imports. A final stage parses each
sketch, substitutes the bodies into it, and writes the result to the workspace.

Two things about the tool decide how it is configured here.

The first is that CodeS is published as a *fine-tuned model* together with the
framework that prompts it, and its own inference driver runs that model locally
through `transformers`, on a GPU, reaching no API at all. The tool also ships a
driver that runs the same three phases against an OpenAI-compatible endpoint,
and that is the one integrated: this platform supplies one credential and one
base URL, and a comparison across solutions depends on all of them reaching the
same provider the same way. What is measured here is therefore CodeS's
multi-layer sketch driven by the experiment's model, not the fine-tuned model
of the paper — a distinction worth keeping in view when reading a result.

The second is that the length of a run is chosen by the model rather than by
the configuration. There are no rounds and no roles to size: the cost of a task
is one request, plus one for each Python file the first response named, plus
one for every function those files declared. A specification that invites a
wide design costs several times what a narrow one does, and nothing in the tool
notices. Two settings follow from that:

- `max_wall_clock_seconds` and `max_token_budget` are the bound, as for
  CodeTeam, and they matter more here because there is no number of rounds to
  lower instead. Unlike CodeTeam, which builds the repository as it goes, CodeS
  assembles at the very end — so a run that stops early still writes what it
  has, and a repository of correct interfaces with some bodies left as `pass`
  is graded rather than discarded.
- `concurrent_requests` is how many of one phase's requests are in flight at
  once. The published pipeline is strictly sequential, which is the default of
  `1`, and seventy-odd requests in sequence exceeds the task's agent timeout
  before the pipeline can finish. Raising it is safe because of what the phases
  are: every file sketch reads the one repository sketch, and every function
  body reads the completed set of file sketches, so no request in a phase can
  see another request in the same phase whatever the order. Note that it
  multiplies with `agent.n_concurrent` rather than being capped by it — the
  requests reaching the provider are the product of the two.

## Results and provenance

Harbor writes each job beneath `jobs/`. Alongside Harbor's own records, each
run keeps the setup that produced it:

- `jobs/<job>/experiment-config.yaml`: the resolved experiment configuration,
  written before the run starts, with templates expanded and defaults filled
  in. It names the credential's environment variable, never its value.
- `jobs/<job>/benchmark.json`: the benchmark version the run resolved to, the
  tasks it selected, and the content digest of every tester image used.
- `jobs/<job>/config.json`: Harbor's record, including the hyperparameters it
  passed to the agent.
- `<trial>/agent/resolved-setup.json`: what actually ran in the container —
  the commit of the code generation tool, the resolved model, the
  hyperparameters, and whether the phases that a configuration can switch off
  were in fact on.
- `<trial>/artifacts/workspace/`: the generated workspace.
- `<trial>/agent/`: the solution's console log, plus what it recorded about its
  own reasoning. Self-Collaboration writes `session-history.json`, the
  structured session including each round's test result; CodeTeam writes
  `codeteam/`, holding every architect's candidate design, the CTO's choice and
  its rationale, the normalized plan, and the result of each QA round; CodeS
  writes `codes/`, holding the prompt and answer of every request it made, one
  file per phase. These are kept out of the workspace deliberately: they are
  evidence about the run, not part of the repository being graded.
- `<trial>/verifier/`: NL2RepoBench pytest output and `reward.txt`, the
  fraction of the task's reference tests that passed, consumed by Harbor.

To inspect results, stop any viewer started from an old clone and launch it
from this repository with the current `jobs` directory:

```powershell
.venv\Scripts\harbor.exe view .\jobs --jobs
```

The viewer's jobs path is independent of the experiment runner. Seeing an old
clone in the viewer is therefore harmless to runs, but that viewer will not
show jobs created in this repository until it is restarted with the path above.

The Harbor results view records the provider, model, and dataset label for each
run, and aggregates uncached input, cached input, and output tokens across
every model call a run made. All three solutions report this through the same
accounting, so the totals mean the same thing for each. Cost is recorded when the
OpenAI-compatible API includes a `cost` value in its usage response; otherwise
Harbor leaves Cost USD empty rather than estimating it from a potentially stale
pricing table. These fields apply to new runs and do not retrofit existing job
directories.

## How a task is evaluated

Each NL2RepoBench task runs two containers: `main`, a generic Python/Node image
where the agent generates the project under `/workspace`, and a `tester`
sidecar built from NL2RepoBench's own evaluator image, which holds the hidden
benchmark tests. Once the agent finishes, Harbor's verifier hook signals the
sidecar over the shared workspace volume; the sidecar strips any test files the
agent generated, copies the remaining code on top of its own reference tests,
installs the package, runs pytest, and reports the passing fraction as the
reward. This mirrors NL2RepoBench's own upstream evaluation flow rather than
reimplementing it.
