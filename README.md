# Scientific Codegen Evaluation MVP

This repository runs one Self-Collaboration generation attempt against
NL2RepoBench's `math-verify` task using Harbor as the orchestrator. Harbor keeps
the generated `/app` workspace under the trial's `artifacts/` directory and
retains Self-Collaboration logs, evaluator output, and the fractional benchmark
score under the trial logs.

## Prerequisites

- Python 3.12 or newer
- Docker with Linux containers enabled
- Git submodules initialized
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

- `artifacts/app/`: generated workspace;
- `agent/`: Self-Collaboration console log and structured session history;
- `verifier/pytest-output.log`: NL2RepoBench pytest output;
- `verifier/evaluator-output.json`: pass/fail counts and success rate;
- `verifier/reward.txt`: the score consumed by Harbor.

The Harbor results view records the provider, model, and NL2RepoBench dataset
label for each run. It also aggregates uncached input, cached input, and output
tokens across every Self-Collaboration model call. Cost is recorded when the
OpenAI-compatible API includes a `cost` value in its usage response; otherwise
Harbor leaves Cost USD empty rather than estimating it from a potentially stale
pricing table. These fields apply to new runs and do not retrofit existing job
directories.

The container is based on NL2RepoBench's original `math-verify:1.0` image, but
its benchmark workspace is root-only. Self-Collaboration runs as the
unprivileged `agent` user in `/app`, so it cannot inspect hidden benchmark
tests; Harbor runs the verifier as root. The verifier preserves NL2RepoBench's
existing install/test commands and computes the original `passed / 192` score.

The upstream NL2RepoBench post-processor removes generated packaging and test
files before overlaying the workspace onto its evaluator image. The Harbor
verifier mirrors that behavior directly rather than invoking the benchmark's
OpenHands generation path or starting nested Docker containers.
