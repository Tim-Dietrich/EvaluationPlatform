# Scientific Codegen Evaluation MVP

This repository runs one Self-Collaboration generation attempt against
NL2RepoBench's `math-verify` task using Harbor as the orchestrator. The task
itself comes from Harbor's own `nl2repobench/nl2repobench` registry dataset
(pinned by content digest, filtered to `math-verify`), not a vendored copy of
the benchmark. Harbor keeps the generated `/workspace` under the trial's
`artifacts/` directory and retains Self-Collaboration logs, verifier output,
and the fractional benchmark score under the trial logs.

## Prerequisites

- Python 3.12 or newer
- Docker with Linux containers enabled
- The Self-Collaboration git submodule initialized
- An OpenRouter API key (or credentials for another OpenAI-compatible endpoint)

Create the project environment and install the pinned Harbor release:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Run the experiment

Create a local environment file from the tracked template, then set `API_KEY`
to your credential. The local `.env` file is ignored by Git:

```powershell
Copy-Item .env.example .env
```

Run the project launcher. It loads `.env` before starting Harbor so every
setting, including `PYTHONUTF8`, is in place, and passes the selected provider
and model to Harbor's experiment history:

```powershell
.venv\Scripts\python.exe main.py
```

Harbor assigns each launch a timestamped job name. This preserves earlier runs
and avoids trying to resume a saved trial whose configuration no longer matches
the current experiment. To inspect results, stop any viewer started from an old
clone and launch it from this repository with the current `jobs` directory:

```powershell
.venv\Scripts\harbor.exe view .\jobs --jobs
```

The viewer's jobs path is independent of the experiment runner. Seeing an old
clone in the viewer is therefore harmless to runs, but that viewer will not show
jobs created in this repository until it is restarted with the path above.

`experiment.yaml` explicitly forwards only `API_KEY`, `MODEL`, and `BASE_URL`
to Self-Collaboration. Harbor keeps the credential as an environment reference
in its saved configuration rather than writing the value into tracked files.
`PYTHONUTF8=1` is required on Windows because NL2RepoBench's instruction
contains Unicode characters; it is harmless on UTF-8-native systems.

The default Self-Collaboration model is `moonshotai/kimi-k2.5` through
OpenRouter. To use another OpenAI-compatible endpoint, change `MODEL`,
`MODEL_PROVIDER`, `BASE_URL`, and `API_KEY` in `.env` without changing tracked
files. `MODEL_PROVIDER` is Harbor's reporting label and is not sent to the API.
For DeepSeek V4 Flash, use the official OpenAI-compatible endpoint (there is no
`/api/v1` path):

```dotenv
MODEL_PROVIDER=deepseek
MODEL=deepseek-v4-flash
BASE_URL=https://api.deepseek.com
```

The launcher rejects the common `https://api.deepseek.com/api/v1` typo before
starting Harbor. DeepSeek also accepts `https://api.deepseek.com/v1`.
Free OpenRouter models can be temporarily rate-limited even with a valid key.
The adapter now waits and retries two additional times after Self-Collaboration
exhausts its initial three requests; if all nine requests are throttled, the job
log reports the rate limit and affected model explicitly. In that case, retry
later or select a model with available capacity.

Harbor writes each timestamped job beneath `jobs/`. Each trial contains:

- `artifacts/workspace/`: generated workspace;
- `agent/`: Self-Collaboration console log and structured session history;
- `verifier/test-output.txt`: NL2RepoBench pytest output;
- `verifier/reward.txt`: the fractional score (`passed / 192`) consumed by Harbor.

The Harbor results view records the provider, model, and NL2RepoBench dataset
label for each run. It also aggregates uncached input, cached input, and output
tokens across every Self-Collaboration model call. Cost is recorded when the
OpenAI-compatible API includes a `cost` value in its usage response; otherwise
Harbor leaves Cost USD empty rather than estimating it from a potentially stale
pricing table. These fields apply to new runs and do not retrofit existing job
directories.

The official `nl2repobench/math-verify` task runs two containers: `main`, a
generic Python/Node image where Self-Collaboration generates the project under
`/workspace`, and a `tester` sidecar built from NL2RepoBench's original
`math-verify:1.0` evaluator image, which holds the hidden benchmark tests.
Self-Collaboration never sees the reference tests. Once it finishes, Harbor's
verifier hook signals the sidecar over the shared workspace volume; the
sidecar strips any test files Self-Collaboration generated, copies the
remaining code on top of its own reference tests, installs the package, runs
pytest, and reports `passed / 192` as the reward. This mirrors NL2RepoBench's
own upstream evaluation flow rather than reimplementing it.
