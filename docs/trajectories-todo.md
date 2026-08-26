# Trajectories: the remaining arms

Self-Collaboration is done — see `src/evaluation_platform/self_collaboration_trajectory.py`
and its tests. This is what the next three need, and nothing else.

## The contract

Two things make an arm's run readable by Harbor's results view and by every
downstream trajectory tool:

1. `SUPPORTS_ATIF: bool = True` on the agent class.
2. A valid ATIF document at `self.logs_dir / "trajectory.json"`.

`self.logs_dir` is `<trial>/agent/`, the host-side copy of `/logs/agent/`. Build
it in `populate_context_post_run`, which runs after the trial has synced its
logs back, so every file the runner wrote is readable there.

```python
from harbor.models.trajectories import (
    Agent, FinalMetrics, Observation, ObservationResult, Step, Trajectory,
)
```

Validate in the test suite, not by eye:

```bash
.venv/Scripts/python.exe -m harbor.utils.trajectory_validator <trial>/agent/trajectory.json
```

## What each arm has to work with

| Arm | Already on disk | Granularity available | Effort |
| --- | --- | --- | --- |
| Terminus | `agent/trajectory.json` | turn-level | **done** — inherited from `Terminus2` |
| Self-Collaboration | `session-history.json` | phase-level | **done** |
| CodeS | `codes/<phase>.jsonl` — prompt *and* answer per request, for `repo_sketch`, `file_sketch`, `function_body` | **turn-level**, no conversion loss | ~half a day |
| Single-Shot | `single-shot/prompt.md`, `single-shot/response.md` | turn-level (two steps) | ~an hour |
| CodeTeam | `codeteam/` role artifacts only | phase-level as-is | ~a day |

CodeS is the best next one to do: its `.jsonl` phase records are prompt/answer
pairs already, so its trajectory is a faithful turn-level document rather than a
summary. Do Single-Shot alongside it — it is two steps and mostly free.

CodeTeam is the one that needs thought. Its roles are the reason to want a
trajectory at all, but only role artifacts survive to disk. Either accept
phase-level, or capture at the seam (below).

## The seam, if you want per-step metrics

Every arm calls `record_response_usage` from exactly one place, and that place
already holds the request and the response:

- CodeS — `run_codes.py:274`
- CodeTeam — `run_code_team.py:227` (the `_ClientProxy` on `chat.completions.create`)
- Single-Shot — `run_single_shot.py:346`

Recording steps there is a runner change, so it applies to new runs only. Doing
it there rather than four times over is the same argument `model_usage.py`
already makes about counting tokens in one place.

Without it, `model-usage.json` gives totals only: put them in `final_metrics`
and leave per-step `metrics` absent. Do not spread a total across steps or
default steps to zero — both read as a measurement that was not made.

## Schema traps that actually cost time

- **`step_id` must run sequentially from 1.** Assign with `enumerate` after the
  step list is complete; do not number as you go.
- **`model_name`, `reasoning_effort`, `reasoning_content`, `tool_calls` and
  `metrics` are agent-only.** Any of them on a `user` or `system` step is a
  validation error.
- **`observation` is *not* agent-only.** That is how a deterministic phase — a
  test run, a build — gets recorded: a `system` step with an observation.
- **`llm_call_count: 0`** marks a step that called no model. On an `agent` step
  it additionally forbids `metrics` and `reasoning_content`.
- **`observation.results[].source_call_id`** must match a `tool_call_id` in the
  *same* step, or be null. Null is correct for anything not from a tool call.
- **Every model is `extra: "forbid"`.** Custom data goes in the `extra` dict,
  nowhere else.
- **Multi-agent arms**: ATIF-v1.7 `subagent_trajectories` embeds whole
  trajectories. Each embedded one needs a unique non-null `trajectory_id`
  (`session_id` may be shared).

## Two rules that are not the schema's

- **Never put agent kwargs in `agent.extra`.** Harbor's own Terminus does, and
  the result is a live OpenRouter key sitting in
  `jobs/2026-08-26__16-33-26/.../agent/trajectory.json`. `jobs/` is gitignored so
  nothing is committed, but trajectories are the file this platform would share.
  Copy `resolved-setup.json` instead — it names the endpoint and the credential's
  variable, never its value. That key should be rotated.
- **`populate_context_post_run` must never raise.** The run is already over; a
  broken converter must leave it unrecorded, not fail it twice. Missing file →
  `logger.debug`. Malformed file → `logger.warning`. Both return `None`.

## Housekeeping

- The instruction is not in any runner's output. Stash it on `self` during
  `run()` or the trajectory has no opening user step.
- When doing arm #2, lift `_final_metrics`, `_read_json` and the
  write-and-validate wrapper out of `self_collaboration_trajectory.py` into a
  shared module. Do not copy them a third time.

## Known follow-up on Self-Collaboration

Its steps are phases, not model turns: the tool collapses each Analyst's and
Coder's internal tool-calling turns into one result string before writing
`session-history.json`, so the detail is gone before the host sees it. The
trajectory says so in its own `notes`.

Turn-level would mean dumping `BaseAgent._messages` (`analyst._messages`,
`coder._messages` in upstream `core/agent.py`) from the runner — OpenAI-format
message lists including tool calls, which map straight onto ATIF `tool_calls`
and `observation`. That is a container-side change and applies to new runs only,
which is why it was left out of the first pass.
