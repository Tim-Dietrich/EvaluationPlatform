# Adding a code generation solution

Three solutions are integrated: Self-Collaboration, CodeTeam and CodeS. Two
baselines are evaluated beside them, and `baselines.md` describes what they are
for. This describes how to add the next solution, and — more usefully — which parts of the job
are not obvious until you have done it three times.

The platform's contract with a solution is narrow. It hands the solution a
natural-language specification and an empty `/workspace` inside a task
container, and expects a repository to exist at `/workspace` when the solution
finishes. Everything else — how many agents, what they say to each other, how
many rounds — belongs to the solution. What the platform requires in return is
that the run be *recorded*: which revision of the tool ran, which model, which
hyperparameters, and what it spent. A result nobody can trace back to a setup
is not a result.

## What has to be true of the tool

Before starting, check three things. Each has failed at least once.

- **The repository is clonable without credentials.** The task container has no
  GitHub credentials and no terminal to ask for any. A private repository fails
  the clone, whatever works on your own machine — where a credential helper is
  quietly answering for you. Confirm with `curl -s -o /dev/null -w "%{http_code}"
  https://api.github.com/repos/<owner>/<name>`; anything but 200 unauthenticated
  will fail in the container.
- **It talks to an OpenAI-compatible endpoint**, or can be made to. The platform
  supplies one credential and one base URL, and the comparison across solutions
  depends on all of them reaching the same provider the same way. A tool whose
  method includes a model of its own is the awkward case: CodeS is a framework
  *and* a fine-tuned checkpoint, and its principal driver loads that checkpoint
  locally and reaches no API at all. It ships a second driver for the same
  pipeline against an API, which is what makes it integrable — and the choice
  narrows what a result is about, so it is reported rather than assumed.
- **It generates into a directory you can redirect.** Most tools write into a
  subdirectory of their own choosing. Harbor grades `/workspace` itself, so the
  location has to be reachable from configuration or from one overridable
  method. If it is hardcoded several layers down, that is a patch to the fork
  rather than a setting, and the patch has to be reported.

## The shape of an integration

Six pieces, five of them new files:

| Where | What it does |
| --- | --- |
| `src/evaluation_platform/experiment_config.py` | one `AgentHyperparameters` block and one entry in `AGENT_HYPERPARAMETERS` |
| `src/evaluation_platform/<tool>_agent.py` | the host side: install the tool in the container, hand it the task, collect what it spent |
| `src/evaluation_platform/run_<tool>.py` | the container side: drive the tool, place its output, record the run |
| `configs/<task>-<tool>.yaml` | one experiment, comparable to its counterpart for the other solutions |
| `tests/test_<tool>_agent.py` | what the two sides promise each other |
| `code_generation/<Tool>` | the source as a submodule, for reading |

The split between the two Python files is the container boundary. The agent runs
on the host with Harbor available; the runner runs inside the task image and has
only what the agent installed. They communicate through uploaded files and
environment variables, never through imports.

## Step 1: read the tool first

Do not start with the adapter. Read the tool's entry point and answer these,
because every one of them decides something in the runner:

- **Is the entry point a library or a script?** CodeTeam's is a program whose
  objects the runner assembles. CodeS's is a research script: its prompts, its
  arguments and its loop are all statements at module scope, so importing it
  runs it, and there is nothing to call. That decides how much of the runner is
  sequencing and how much is configuration, and it is the first thing to know.
- **Where does it write the generated project?** This becomes `/workspace`.
  Find the single place that decides it — CodeTeam has one method,
  `Context.make_repo_root`, which the runner overrides. Where there is no such
  place, as in CodeS, call the functions its output loop is built from and
  write the files yourself; do not reimplement what they do.
- **Where does it write its own logs and artifacts?** If that is inside the
  directory it generates into, it is inside the repository being graded, and
  the verifier will copy the tool's planning records onto the benchmark's tests
  as though they were part of the library. Redirect them to `/logs/agent/`.
- **How is it configured, and which knobs have no override?** CodeTeam reads
  fifteen environment variables and `temperature` is not among them. That one
  gap is why the runner builds the tool's configuration object directly rather
  than setting environment variables and calling its `main()`. A setting you
  cannot reach is a setting that cannot be part of the recorded setup.
- **Which single function reaches the model API?** That is the instrumentation
  point for usage accounting and reasoning parameters. If there is more than
  one, wrap the client rather than the call sites.
- **Which of its requirements does a run actually import?** Tools ship one
  `requirements.txt` covering the runtime, the paper's analysis scripts, and
  optional extras. Installing all of it means downloading a scientific stack
  into every trial. Install what the run imports, pinned.
- **Does it bound its own cost?** Some do and leave it unset; some do not at
  all. Without a bound, the only limit is the task's agent timeout, which is
  reached with nothing recorded about why. Ask also what a bound *costs*: a
  tool that writes as it goes leaves a gradeable workspace when it stops, and
  one that assembles at the end leaves nothing unless the runner assembles
  anyway.

## Step 2: declare the hyperparameters

Add an `AgentHyperparameters` to `experiment_config.py` and register the agent's
module in `AGENT_HYPERPARAMETERS`. The module is the identity, not the class.

```python
NEW_TOOL = AgentHyperparameters(
    solution="NewTool",
    types={"max_rounds": int, "reviewers": int, "review_enabled": bool},
    defaults={"max_rounds": 3, "reviewers": 2, "review_enabled": True},
)
```

Four rules, each with a reason:

- **Defaults are the tool's own.** A configuration that states nothing should
  run the published method, so that "we changed nothing" is the readable
  default rather than an accident.
- **A hyperparameter with no default is a deliberate choice.** Reasoning
  settings, budgets, and seeds have none: leaving one out means the provider's
  default applies, or the run is unbounded, or the draw is not pinned — each a
  thing a configuration should have to say out loud.
- **Rename a name that collides and means something else.** CodeTeam calls its
  QA repair rounds `max_rounds`, which is also Self-Collaboration's name for
  Coder rounds. It is exposed here as `max_qa_rounds`, because two
  configurations sitting side by side in `configs/` should not appear to set
  the same thing. The runner translates back.
- **Every ablation is one key.** If the paper reports an ablation, it should be
  one boolean in the configuration, not a combination the reader has to
  assemble.

Types are enforced by `_coerce_hyperparameter`. Integers must be at least 1,
since zero means the phase does not happen and a configuration should say that
by omitting the key — unless the integer is a seed, which names a draw rather
than bounding one, in which case add it to `_SEEDS`. A string with a fixed set
of values gets a check by name there too, as `rag_backend` does.

## Step 3: the agent, on the host

Subclass `BaseInstalledAgent`. Copy `code_team_agent.py` and change what
differs; the structure is the same each time.

`__init__` pops the hyperparameters out of Harbor's `kwargs`, pops `repository`
and `commit`, calls `reject_foreign(kwargs)` so another solution's
hyperparameter cannot arrive here and be silently accepted, then resolves.

`install` does four things in order: ensure `git`, install the pinned runtime
packages as root, clone and check out the pinned commit, and upload the runner
*together with* `model_usage.py`. The clone must pass
`env=GIT_NON_INTERACTIVE`; without it, an unreadable repository reaches Git as
an authentication challenge and Git answers by prompting, turning a two-second
failure into a trial that stalls until its own timeout.

`run` uploads the instruction and the hyperparameters as files and passes their
paths in the environment. Neither may go on the command line: instructions run
to tens of kilobytes, and anything on a command line is visible in process
listings and in Harbor's own logs. Execute with `cwd` at the workspace and pipe
the output through `tee` into `/logs/agent/<tool>.log`.

`populate_context_post_run` calls `populate_usage_context`, which reads the file
the runner wrote and gives Harbor the token and cost figures its results view
aggregates.

Harbor puts four things in the container's environment, from
`ExperimentConfig.to_harbor_config`: the credential under the name
`model.api_key_env` gives, that name again as `API_KEY_ENV`, plus `BASE_URL` and
`MODEL`. The runner reads the credential indirectly —
`os.environ[os.environ["API_KEY_ENV"]]` — so the configuration decides what the
variable is called.

## Step 4: the runner, in the container

The runner is uploaded as a loose file, so it imports `model_usage` by flat name
after putting its own directory on the path. That one line makes the same module
importable in the container, where the directory is `/installed-agent`, and on
the host, where it is the package — which is how the tests reach it.

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_usage import UsageTotals, record_response_usage
```

Its job, in order: read the hyperparameters and instruction from the paths named
in the environment; put the tool's checkout on `sys.path`; build the tool's own
configuration object with the workspace and artifact locations corrected; build
and instrument the model client; record the resolved setup; run the workflow;
and write the usage totals in a `finally`, so a crashed run still reports what
it spent up to the crash.

Two files go to `/logs/agent/` and both matter more than they look:

- `model-usage.json` — what the run spent, through `UsageTotals` and
  `record_response_usage`. Use the shared module rather than counting tokens
  yourself. A token, a cached token and a dollar have to mean the same thing
  for every solution, and two implementations of that is how they stop meaning
  it.
- `resolved-setup.json` — what actually ran: the commit the container ended up
  with, the model the environment resolved to, the hyperparameters, and whether
  the phases a configuration can switch off were in fact on. The experiment
  configuration states the intent; this states the observation, and they are not
  always the same thing.

Instrument the model client by wrapping, not replacing. `run_code_team.py`
substitutes a proxy for the tool's OpenAI client that intercepts
`chat.completions.create` and forwards everything else, which adds three things
without touching the tool: reasoning fields on the request, usage accumulation,
and a wait when the provider throttles. The tool's own request construction,
retries and response parsing stay exactly as published.

## Step 5: the configuration

Copy the counterpart configuration for the same task and change only the `agent`
section. The `benchmark` and `model` blocks must stay byte-identical — that
agreement is the entire basis for comparing the results, and a test in
`tests/test_experiment_config.py` asserts it. The `run` block is deliberately
not part of that claim: how many attempts to make and how many to run at once is
how much of the experiment to do rather than what it measures.

Configurations carry their reasoning in comments. A future reader should be able
to learn from the file why `temperature` is 0.0 rather than the tool's own
default, and what turning a given key off would mean. This is the most-read
artefact of an integration; write it for someone who has not read the code.

## Step 6: tests

`tests/test_<tool>_agent.py` covers what the two sides promise each other:
install pins the revision and uploads both modules; the clone is
non-interactive; the instruction and credential never reach a command line; the
uploaded hyperparameters are complete, defaults included; hyperparameters the
runner cannot use are rejected; the hyperparameter-to-tool-config mapping is
right; the client wrapper adds what it should and counts what it should.

Tests must not require the submodule to be checked out. Use a small stand-in for
the tool's configuration object, as `FakeSystemConfig` does, so the mapping is
under test on a machine that has only cloned this repository.

## Before the first real run

In this order, because each step is cheaper than the one after it:

1. `.venv\Scripts\python.exe -m pytest -q`.
2. Drive the runner end-to-end against the real checkout with the tool's own
   mock or offline model client, if it has one. This catches workspace
   placement, artifact placement, and configuration mapping for free. CodeTeam's
   integration was fully exercised this way before a single token was spent.
3. Load the configuration and render `to_harbor_config`, then re-load the
   snapshot it archives and check it round-trips — that is what `--resume`
   depends on.
4. One task, one attempt, one concurrent trial. Raise them once it works.

## Traps

The ones that have actually cost time:

- A fork that is private. The clone fails in the container and nowhere else.
- The tool generating one directory below where Harbor grades, which scores an
  empty repository rather than failing.
- The tool's own artefacts landing inside the graded workspace.
- A hyperparameter name that means something different in another solution.
- A setting with no environment override, which quietly falls back to the
  tool's default and is absent from the record.
- Installing the whole of `requirements.txt` into every trial.
- Reimplementing usage accounting instead of importing it.
- A tool whose retry loop does not cover its JSON path, so a rate limit costs
  the trial on the majority of its calls.
- A tool that retries a rate limit without waiting, which spends every attempt
  in the same few seconds; and a wait added *outside* the tool's attempt count,
  which multiplies the requests the configuration asked for.
- A tool that assembles its output only at the end, so a run stopped by its own
  budget is scored as an empty repository.
- One unguarded loop over every file, where a single malformed response ends
  the loop and takes the other files with it.
- A prompt copied out of the tool into a configuration or a runner, which is
  then free to disagree with the revision the configuration pins.
- A repository that is enormous for what a run reads. Every trial clones it;
  a partial, sparse clone pins the same revision and fetches a fraction of it.

## Write down what you changed

`docs/latex/tool-modifications.tex` is the scientific record: every intervention
in a tool, and every default the harness overrides, is reported there with its
reason. The bar is that a reader should be able to tell what was measured — the
published method, or something adjacent to it. A change that seems too small to
report is usually one whose effect is easy to underestimate; report it and say
it is small.

Include, for each solution: what was modified in the tool and why, what the
harness supplies around it, which defaults the configuration overrides, and
which alternatives were rejected. The rejected alternatives matter. They are
what stops the next person from trying them again.
