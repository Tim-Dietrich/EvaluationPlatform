# Scientific Codegen Evaluation MVP

This repository runs code generation solutions against whole benchmarks with
Harbor as the orchestrator, and records the setup of every run so results from
different solutions can be compared. This file is the operating manual: how to
set the platform up, configure an experiment, run it, and find what it wrote.
The reasoning behind the design and the results themselves are in the
accompanying paper.

## Prerequisites

- Python 3.12 or newer
- Docker with Linux containers enabled
- An OpenRouter API key, or credentials for another OpenAI-compatible endpoint

## Setup

Create the project environment and install the pinned Harbor release:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Copy the tracked template and set `API_KEY`. `.env` is ignored by Git and holds
credentials and backend defaults only; nothing that changes what a run does
lives there:

```powershell
Copy-Item .env.example .env
```

```dotenv
API_KEY=
MODEL_PROVIDER=openrouter
MODEL=moonshotai/kimi-k2.5
BASE_URL=https://openrouter.ai/api/v1
PYTHONUTF8=1
```

`MODEL_PROVIDER` is Harbor's reporting label and is not sent to the API.
`PYTHONUTF8=1` is required on Windows so task instructions are read as UTF-8.
For DeepSeek, set `MODEL_PROVIDER=deepseek`, `MODEL=deepseek-v4-flash` and
`BASE_URL=https://api.deepseek.com`; a `/api/v1` path is rejected at
validation because that endpoint does not have one.

The submodules are checked out for reading and reference only. A run clones
the revision named in the experiment configuration into the task container, so
a clone without them still runs:

```powershell
git submodule update --init
```

- `code_generation/` holds the three integrated solutions.
- `benchmarks/nl2repobench/upstream` holds NL2RepoBench's published source at
  the commit the subset selection reads its task metadata from. Tasks come
  from Harbor's registry, not from this checkout.

## Running an experiment

```powershell
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py --config configs/nl2repobench-self-collaboration.yaml
.venv\Scripts\python.exe main.py --config configs/nl2repobench-codes.yaml --job-name codes-full
```

`--config` defaults to `configs/math-verify-self-collaboration.yaml`;
`--job-name` defaults to the launch timestamp. The launcher loads `.env`,
validates the configuration, resolves and pins the benchmark, makes the tester
images available, and then starts Harbor. Unknown keys, unknown
hyperparameters, an unset credential and an invalid endpoint path are rejected
before Docker starts.

`scripts/run-smoke-tests.ps1` runs the `math-verify-*` configurations once
each, as separate jobs under a shared `smoke__<timestamp>` prefix, as an
end-to-end check of the harness against the real model backend. `-DryRun`
prints the commands instead of running them; `-StopOnFailure` stops at the
first failing arm.

### Concurrency

```yaml
run:
  n_concurrent_trials: 4
  retry:
    max_retries: 2
    wait_multiplier: 2
    min_wait_sec: 5
    max_wait_sec: 120
```

Two limits apply, and they are not the same:

- `run.n_concurrent_trials` is the machine's. Each trial runs an agent
  container and a tester sidecar, and Harbor passes the memory a task declares
  (2 CPUs and 8 GB per NL2RepoBench trial) to Docker as a ceiling: a trial
  that reaches it is killed and scored as an error. Before a run starts the
  launcher compares what the selected tasks ask for with what Docker reports.
  On Windows and macOS that figure is the Docker VM's allocation, not the
  host's.
- `agent.n_concurrent` is the provider's. It caps how many trials call the
  model at once while container setup, installation and verification stay
  fully parallel. It cannot exceed `n_concurrent_trials`; a configuration
  that makes it larger is rejected.

Tasks spend most of their time waiting on the model, so concurrency above the
core count still pays; memory is the limit that bites first. A NL2RepoBench
task takes around six minutes, so 104 in sequence is about ten hours and four
at a time roughly two and a half.

`run.retry` re-runs trials that failed on a provider hiccup or a dropped
connection. Harbor's own exclusions still apply, so a timeout, an exhausted
usage limit or a rejected credential fails once.

Tester images are pulled concurrently and cached. Each NL2RepoBench image is
around two gigabytes and every task has its own, so the first run of the whole
benchmark is a large one-time download.

### Continuing an interrupted run

```powershell
.venv\Scripts\python.exe main.py --resume jobs\2026-08-21__16-48-46
```

The setup comes from the `experiment-config.yaml` archived beside the job;
`--resume` cannot be combined with `--config` or `--job-name`. Tester images
the remaining tasks need are made available again first. Trials that
finished, including failed ones, are kept; cancelled trials are discarded and
run again. A job is resumable once Harbor has written its `config.json`; a
run that failed before that point has to be launched again.

## Experiment configurations

One file under `configs/` describes one experiment: the benchmark and task
selection, the model backend, the code generation solution and its revision,
and the hyperparameters it receives. Files are named
`<benchmark>-<solution>.yaml`; the `math-verify-*` files are the single-task
versions of each arm, for iterating cheaply, and
`nl2repobench-subset-single-shot.yaml` is the 30-task subset described under
[Benchmarks](#benchmarks).

Values may use `${VAR}` and `${VAR:-default}`, resolved against the
environment; the shipped configurations use this for the model backend, so
`MODEL`, `MODEL_PROVIDER` and `BASE_URL` in `.env` override it without editing
tracked files. Comparing two solutions means two files that differ only in
their `agent` section; `tests/test_experiment_config.py` asserts that the
`benchmark` and `model` blocks agree across the arms of a comparison.

### `benchmark`

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

- `dataset` and `ref` name a Harbor registry dataset and pin its version. A
  floating reference is resolved once at launch and re-pinned, so every trial
  of a run sees one version.
- Omit `task_names` to run the entire benchmark, or narrow it with glob
  patterns (org-qualified), `exclude_task_names` and `n_tasks`. Harbor's own
  dataset filtering does the selecting.
- `image_mirror` is optional. Before a run, the launcher reads the tester
  images named by the selected tasks and makes each one available locally
  under the name the task expects, pulled from the mirror; Docker Compose then
  uses the local image. NL2RepoBench needs it because its tasks name images on
  a private registry that are public on `ghcr.io`.
- `path` instead of `dataset` names a generated benchmark directory (HumanEval
  and SketchEval below). The launcher computes a digest over the tree and
  archives it in `jobs/<job>/benchmark.json`; a `ref` beside a `path` is
  refused. The task filters work the same way.
- `task: {path: ...}` in place of `benchmark` runs a single hand-written task
  directory. Exactly one of `benchmark` and `task` is required.

### `model`

```yaml
model:
  provider: ${MODEL_PROVIDER:-openrouter}
  name: ${MODEL:-moonshotai/kimi-k2.5}
  base_url: ${BASE_URL:-https://openrouter.ai/api/v1}
  api_key_env: API_KEY
  routing:
    order:
      - baidu/fp8
      - siliconflow/fp8
    allow_fallbacks: false
```

`api_key_env` names the environment variable holding the credential; its
value is never written to a job. `routing` pins which OpenRouter endpoints may
answer, since one model is served by many providers at different
quantizations and prices. It reaches every arm as `MODEL_ROUTING` and enters
each request as its `provider` field: merged into the body where this platform
builds the request, injected through a wrapped OpenAI client where the tool
builds its own, and through Harbor's `extra_body` for Terminus. An unknown
routing key is rejected. Each run records which endpoints actually answered as
`providers_served` in its `resolved-setup.json`. Endpoint tags belong to the
configured model; `GET /api/v1/models/<model>/endpoints` lists them with
quantization, price and uptime.

### `sampling`

`temperature`, `top_p` and `max_tokens` are set once per comparison in a
sampling file, `configs/generation.yaml` by default:

```yaml
temperature: 0.0
top_p: 0.95
max_tokens: 32768
```

A configuration may name a different file with `sampling: {file: ...}`;
`humaneval-self-collaboration.yaml` reads `configs/generation-humaneval.yaml`.
The three values are not hyperparameters of any solution, and stating one
under `agent.hyperparameters` is rejected. The four solutions that run in the
task container put them in the request body; Terminus takes `temperature`
directly and receives the other two as `llm_kwargs`. Every run writes the
values it actually used, and the route each took, into its
`resolved-setup.json`, and the archived `experiment-config.yaml` pins them for
`--resume`.

### `agent`

```yaml
agent:
  import_path: evaluation_platform.self_collaboration_agent:SelfCollaborationAgent
  n_concurrent: 10
  repository: https://github.com/YihongDong/Self-collaboration-Code-Generation.git
  commit: a6490a9d0d32f3238cc5b776d2de8d2134d2b138
  hyperparameters:
    max_rounds: 3
```

`import_path` selects the adapter in `src/evaluation_platform/`, which
installs the tool at `commit` in the task container, hands it the
specification and the hyperparameters, and records what it spent. Each adapter
declares the hyperparameters it accepts — `AGENT_HYPERPARAMETERS` in
`src/evaluation_platform/experiment_config.py` is the schema of record — and a
name belonging to another solution, or to none, is rejected with the known
names listed. The ones that decide the shape of a run:

- **Self-Collaboration** (`self_collaboration_agent`): `max_rounds`,
  `analyst_steps`, `coder_steps`, `test_command`. The Tester runs
  `test_command` in the agent's workspace between Coder rounds, so it needs
  both a command and `max_rounds` greater than one. `task_shape: humaneval`
  selects the authors' HumanEval entry point instead, which takes `max_rounds`
  and `max_steps` and refuses `test_command`.
- **CodeTeam** (`code_team_agent`): `architects`, `max_qa_rounds`,
  `rag_enabled` and `rag_backend`, `dynamic_developer_allocation` and
  `fixed_developer_agents`, `git_coordination`. The tool has no bound of its
  own on the cost of a task, so `max_wall_clock_seconds` and
  `max_token_budget` are the bound; reaching either is recorded as a failed
  agent run and the workspace is graded as it stands.
- **CodeS** (`codes_agent`): `max_wall_clock_seconds` and `max_token_budget`
  as for CodeTeam, and `concurrent_requests`, how many of one phase's
  requests are in flight at once. The default `1` is the published sequential
  pipeline, which does not finish within a NL2RepoBench task's agent timeout.
  It multiplies with `agent.n_concurrent`. The integrated driver is the tool's
  OpenAI-compatible one, so the experiment's model runs the CodeS sketch
  pipeline; the fine-tuned model of the CodeS paper is not used.
- **Single-Shot** (`single_shot_agent`): one request, the reply parsed into
  files by a deterministic writer. It is not a published tool: `repository`
  and `commit` are rejected, and the run records digests of its prompt and
  runner instead. `truncated`, `common_top_level_directory` and
  `parse_warnings` in `resolved-setup.json` say whether the reply was cut off
  or misparsed.
- **Terminus 2** (`terminus_agent`): Harbor's own reference agent, run on the
  host against a tmux session in the container and reaching the provider
  through LiteLLM. `max_turns` is its budget; the Harbor release is its pin.

`docs/adding-a-solution.md` describes what the platform requires of a tool and
the files an integration adds; `docs/baselines.md` describes what the two
baselines are for and how `max_turns` was chosen.

## Benchmarks

**NL2RepoBench** is a Harbor registry dataset of 104 tasks.
`configs/nl2repobench-*.yaml` run all of them; `configs/math-verify-*.yaml` run
the single task `math-verify`. `configs/nl2repobench-subset-single-shot.yaml`
runs a stratified subset of 30 (8 / 13 / 9 across the benchmark's three
difficulty levels). The subset is generated, not hand-written:
`notebooks/nl2repobench-subset-selection.ipynb` derives it from the task
metadata cached in `benchmarks/nl2repobench/task_metadata.csv`, writes
`benchmarks/nl2repobench/subset.csv` and the configuration, and keeps the
generated file byte-identical to `configs/nl2repobench-single-shot.yaml`
outside its name, description and `task_names`. Re-run the notebook rather
than editing the task list.

**HumanEval** is not in Harbor's registry (`humanevalfix` and `evoeval` are
different benchmarks), so its 164 tasks are generated from the pinned inputs
in `benchmarks/humaneval/data/` and named by directory:

```bash
python scripts/build_humaneval_tasks.py
```

```yaml
benchmark:
  path: benchmarks/humaneval/tasks
```

`benchmarks/humaneval/README.md` documents the inputs, their pins and the
grading pipeline. `configs/humaneval-self-collaboration.yaml` runs the
authors' own entry point against it.

**SketchEval** is published nowhere as a dataset; it exists as nineteen Python
projects under `validation/cleaned_repos/` in the CodeS repository, which is
pinned as a submodule. Its tasks are generated from that commit:

```bash
python scripts/build_sketcheval_tasks.py
```

```yaml
benchmark:
  path: benchmarks/sketcheval/tasks
```

Its reward is SketchBLEU, the CodeS authors' similarity score between the
generated and the reference repository. Nothing is executed and there are no
hidden tests, so it is not comparable to a NL2RepoBench reward, and the
attainable ceiling is below 1.0 and differs per repository.
`benchmarks/sketcheval/README.md` documents the ceilings and the metric's
pins. `configs/sketcheval-codes.yaml` runs CodeS against it.

Generated task trees (`benchmarks/*/tasks/`) are ignored by Git; regenerate
them rather than committing them.

## Results

Harbor writes each job beneath `jobs/`. Alongside Harbor's own records, each
run keeps the setup that produced it:

- `jobs/<job>/experiment-config.yaml`: the resolved configuration, written
  before the run starts, with templates expanded and defaults filled in. It
  names the credential's environment variable, never its value.
- `jobs/<job>/benchmark.json`: the benchmark version the run resolved to, the
  tasks it selected, and the content digest of every tester image used.
- `jobs/<job>/config.json`: Harbor's record, including the hyperparameters it
  passed to the agent.
- `<trial>/agent/resolved-setup.json`: what actually ran in the container —
  the commit of the tool, the resolved model, the hyperparameters, the
  `generation` block with the sampling used and the route each value took,
  `providers_served`, the `failures` block below, and which switchable phases
  were on. Single-Shot records the digests of its prompt and runner, Terminus
  the Harbor release.
- `<trial>/agent/`: the solution's console log and its own record of the run:
  `session-history.json` for Self-Collaboration; `codeteam/` (every
  architect's design, the CTO's choice, the plan, each QA round) for CodeTeam;
  `codes/` (prompt and answer of every request, one file per phase) for CodeS;
  `single-shot/` (the prompt and the verbatim reply) for Single-Shot. Terminus
  keeps its trajectory and terminal recording as it does under Harbor
  anywhere. These stay out of the workspace so they are not graded.
- `<trial>/artifacts/workspace/`: the generated workspace.
- `<trial>/verifier/`: the tester's pytest output and `reward.txt`.

The `failures` block of `resolved-setup.json` counts, under the same names in
every arm, the requests that did not complete normally. Every category is
written even at zero:

- `output_limit_exhausted`: generation began and stopped at the token cap; the
  reply exists and is truncated.
- `context_exhausted`: the request was refused before generation because the
  prompt plus `max_tokens` did not fit the context window; there is no reply.
- `request_timeout`: the request was still being answered when its clock ran
  out.

Whether the generated code passes the benchmark's tests is the verifier's
finding and never appears there.

To inspect results, start Harbor's viewer from this repository with the
current `jobs` directory (a viewer started from another clone keeps showing
that clone's jobs until restarted):

```powershell
.venv\Scripts\harbor.exe view .\jobs --jobs
```

The viewer lists each job with the agent that produced it, the provider, model
and dataset, and aggregates uncached input, cached input and output tokens
across every model call. All arms report tokens through the same accounting.
Cost is filled in only when the API's usage response includes a `cost` value;
otherwise it is left empty rather than estimated.

## How a task is evaluated

Each NL2RepoBench task runs two containers: `main`, a generic Python/Node
image where the agent generates the project under `/workspace`, and a
`tester` sidecar built from NL2RepoBench's own evaluator image, which holds
the hidden reference tests. Once the agent finishes, Harbor's verifier hook
signals the sidecar over the shared workspace volume; the sidecar strips any
test files the agent generated, copies the remaining code on top of its own
reference tests, installs the package, runs pytest, and reports the passing
fraction as the reward. A solution's own test loop — Self-Collaboration's
Tester, CodeTeam's QA — only ever runs the code and tests it wrote itself; the
reference tests are never visible to the agent.

## Tests and analysis

```powershell
.venv\Scripts\python.exe -m pytest
```

The notebooks under `notebooks/` read the job directories, select the subset,
and produce the figures under `docs/figures/` and the LaTeX tables under
`docs/tables/`; `job_analysis.py`, `job_figures.py` and `job_tables.py` are
the shared code behind them. They need the analysis stack:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[analysis]"
```

`docs/architecture/` holds the PlantUML sources and renders of the platform's
architecture diagrams.
