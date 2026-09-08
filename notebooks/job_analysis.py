"""Read a Harbor job directory into one row per trial.

Every arm of the NL2RepoBench comparison writes the same kind of job directory —
one sub-directory per trial, holding a `result.json`, a verifier report, and
whatever the agent itself logged. What differs is only the last part, so this
module keeps the shared reading and the zero-cause classifier in one place and
lets each arm contribute its own columns on top.

That split is what makes a cross-arm comparison mean anything. A cause like
`ModuleNotFoundError: a file it never wrote` is assigned by the same code
whichever arm produced it, so counting those across arms compares the arms
rather than three slightly different parsers.

Reading a job
-------------

    import job_analysis as ja

    df = ja.load_job("2026-09-04__22-49-39")          # one arm
    both = ja.load_jobs({"single-shot": "2026-09-04__22-49-39",
                         "terminus-2": "2026-09-05__02-26-02"})

The arm is detected from `agent_info.name` in the job's own results, so nothing
has to be passed for it. Results are cached per (job, solved_at).

The common columns
------------------

`COMMON_COLUMNS` is the contract every arm satisfies, and is what a comparison
should be built on. Anything outside it is arm-specific and named in that arm's
`ARMS` entry.

Two of the common columns deserve a note.

`scored` is False when the verifier never returned a reward — a host or harness
fault, not a finding about the code. Those rows carry `reward = NaN`, never 0,
so they cannot quietly drag a mean down. Aggregate over `df[df.scored]` and say
the n.

`hit_budget` is the arm-neutral form of "the method was stopped rather than
deciding it was finished": the output-token ceiling for a single reply, the turn
ceiling for an agent loop, the round ceiling for a role loop. `budget_name` and
`budget_value` say which budget, read from each trial's own configuration rather
than assumed here.
"""

from __future__ import annotations

import collections
import datetime as dt
import functools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "jobs"
TASK_METADATA = REPO / "benchmarks" / "nl2repobench" / "task_metadata.csv"

LEVELS = ["Easy", "Medium", "Hard"]

#: How far a trial got before it stopped, earliest failure first. The harness
#: fault is last because it is not a point on that pipeline at all.
STAGES = ["never configured", "never imported", "ran, all failed", "no verdict"]

#: What a trial produced, for the budget view. The first two earned something.
OUTCOMES = ["solved", "partial credit", *STAGES, "lost to host fault"]
EARNED = ("solved", "partial credit")

COMMON_COLUMNS = (
    "arm", "trial", "task", "difficulty", "n_hidden_tests", "spec_words",
    "reward", "scored", "solved",
    "cost_usd", "in_tok", "cache_tok", "out_tok", "total_tok", "workspace_files",
    "t_total", "t_agent", "t_verify",
    "budget_name", "budget_value", "hit_budget", "ending",
    "exception", "outcome", "fault_stage", "fault_cause",
    "pt_ran", "pt_passed", "pt_failed", "pt_error", "pt_skipped",
)


# --------------------------------------------------------------------------- #
# reading primitives
# --------------------------------------------------------------------------- #

_SUMMARY = re.compile(r"^=+ (.*?) in [\d.]+s[^=]*=+$", re.M)
_HEADING = re.compile(r"^=+ ([a-zA-Z ]+?) =+$", re.M)
# pytest prefixes the raised exception of a collection error or a test failure
# with `E   `. This is the line that says what actually went wrong.
_RAISED = re.compile(r"^E\s+([A-Za-z_][\w.]*(?:Error|Exception|Warning|Failed))\b:?[ ]?(.*)$", re.M)
# When pytest dies before it can report anything — a plugin that fails to load,
# say — there is no `E   ` line, only a plain interpreter traceback.
_BARE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception)): (.*)$", re.M)


def read_text(path, tail_bytes: int | None = None) -> str:
    """File contents, optionally only the last `tail_bytes`. Missing file -> ''."""
    try:
        with open(path, "rb") as handle:
            if tail_bytes is not None:
                handle.seek(max(0, os.path.getsize(path) - tail_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def seconds(block) -> float:
    """Wall-clock length of one phase, from the pair of timestamps bounding it."""
    if not block:
        return float("nan")
    read = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (read(block["finished_at"]) - read(block["started_at"])).total_seconds()


def _split_sections(text: str) -> dict:
    """pytest's report, split on its `==== NAME ====` banners."""
    found: dict[str, str] = {}
    marks = [(m.start(), m.end(), m.group(1).strip()) for m in _HEADING.finditer(text)]
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        found[name] = found.get(name, "") + text[end:stop]
    return found


def pytest_outcome(tests: str) -> dict:
    """pytest's own summary line, as counts. `pt_ran` is False if it never got one."""
    counts = {f"pt_{kind}": 0 for kind in ("passed", "failed", "error", "skipped")}
    summaries = _SUMMARY.findall(tests)
    counts["pt_ran"] = bool(summaries)
    if summaries:
        for count, kind in re.findall(
                r"(\d+) (passed|failed|errors?|skipped)", summaries[-1]):
            counts[f"pt_{kind.rstrip('s')}"] = int(count)
    return counts


# --------------------------------------------------------------------------- #
# why a trial scored a flat zero
# --------------------------------------------------------------------------- #

def _nested_root(workspace: str):
    """The single sub-directory a whole project was built inside, if there is one.

    A workspace holding exactly one directory and no files at all is a project
    that was created one level too deep. Nothing in it is importable from where
    the tests run, however correct the code inside it may be.
    """
    try:
        entries = os.listdir(workspace)
    except OSError:
        return None
    dirs = [e for e in entries if os.path.isdir(os.path.join(workspace, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(workspace, e))]
    return os.path.join(workspace, dirs[0]) if len(dirs) == 1 and not files else None


@functools.lru_cache(maxsize=None)
def _written_modules(workspace: str) -> tuple:
    """What the agent wrote, as `(top-level, dotted, bare, dotted-one-level-down)`.

    This is what separates a missing module the agent was supposed to write from
    a third-party library it assumed would be installed. Both raise the identical
    `ModuleNotFoundError`, and they are opposite mistakes.

    Dotted paths are resolved against the two roots a Python project is imported
    from, the workspace itself and `src/`, so `src/tablib/utils.py` is recorded
    as `tablib.utils`. The fourth set repeats that against a nested project root
    (see `_nested_root`), which is what catches a package that exists and is
    correct but sits one directory too deep to import. Bare names are kept as a
    last resort only: a leaf like `utils` matches some vendored file in half
    these repositories, so it is never consulted while a more specific answer is
    available.
    """
    roots = [(base, True) for base in (workspace, os.path.join(workspace, "src"))]
    nested = _nested_root(workspace)
    if nested:
        roots += [(base, False) for base in (nested, os.path.join(nested, "src"))]

    tops, dotted, bare, deep = set(), set(), set(), set()
    for root, dirs, files in os.walk(workspace):
        relative = os.path.relpath(root, workspace)
        depth = 0 if relative == "." else len(relative.split(os.sep))
        if depth > 5:
            dirs[:] = []
            continue
        for entry in list(files) + list(dirs):
            name = entry[:-3] if entry.endswith(".py") else entry
            bare.add(name)
            for base, at_root in roots:
                inside = os.path.relpath(os.path.join(root, entry), base)
                if inside.startswith(".."):
                    continue
                parts = inside.split(os.sep)
                parts[-1] = name
                (dotted if at_root else deep).add(".".join(parts))
                if at_root and len(parts) == 1:
                    tops.add(name)
    return frozenset(tops), frozenset(dotted), frozenset(bare), frozenset(deep)


@functools.lru_cache(maxsize=None)
def _workspace_files(workspace: str) -> int:
    """How many files the run left behind, to five directory levels.

    A run that wrote nothing at all is worth telling apart from one that wrote
    the wrong thing, and the reward alone will not do it — one task in this
    benchmark scores 0.98 against an empty workspace, because its tests import
    the library already installed in the image rather than the generated code.
    """
    total = 0
    for root, dirs, files in os.walk(workspace):
        relative = os.path.relpath(root, workspace)
        depth = 0 if relative == "." else len(relative.split(os.sep))
        if depth > 5:
            dirs[:] = []
            continue
        total += len(files)
    return total


def _refine(kind: str, message: str, workspace: str) -> str:
    """Split the two overloaded import exceptions into their real causes.

    `ModuleNotFoundError` and `ImportError` each cover several distinct mistakes,
    and the exception name alone loses the distinction. The message carries it:
    the name that could not be found, and the path it was looked for in.
    """
    kind = kind.rsplit(".", 1)[-1]   # json.decoder.JSONDecodeError -> JSONDecodeError

    if kind == "ModuleNotFoundError":
        found = re.search(r"No module named '([\w.]+)'", message)
        if not found:
            return kind
        missing = found.group(1)
        tops, dotted, bare, deep = _written_modules(workspace)
        # Most specific answer first. The file sitting at exactly that dotted
        # path means it exists and is simply not importable — a packaging or
        # layout mistake. Resolving only one level down means the whole project
        # was built inside a sub-directory, which is the same code and a
        # different mistake. A written package missing the module it imports
        # means the agent named a file it never produced.
        if missing in dotted:
            return "ModuleNotFoundError: own code, wrong layout"
        if missing in deep:
            return "ModuleNotFoundError: project built one level too deep"
        if missing.split(".")[0] in tops:
            return "ModuleNotFoundError: a file it never wrote"
        if missing.split(".")[-1] in bare:
            return "ModuleNotFoundError: own code, wrong layout"
        return "ModuleNotFoundError: uninstalled dependency"

    if kind == "ImportError":
        if "circular import" in message:
            return "ImportError: circular import"
        # A path into the interpreter's own libraries means the name is missing
        # from something the agent did not write — it wrote against an API that
        # the installed version does not have.
        if "site-packages" in message or "/usr/local/lib" in message:
            return "ImportError: name gone from an installed module"
        return "ImportError: name missing from its own module"

    return kind


def classify_zero(trial: Path, text: str, stdout: str) -> tuple[str, str]:
    """Why one trial scored a flat 0.00, as (stage, cause).

    `stage` is how far the pipeline got before it stopped; `cause` is the
    technical reason it stopped there. Precedence runs earliest-first, because a
    project pytest could not configure never reached the question of whether its
    code imports, and code that never imported never reached the question of
    whether it behaves.

    Where a report holds several exceptions, the cause is the one raised most
    often — the error blocking the most of the suite rather than whichever
    pytest happened to print first.
    """
    if not text or "Tester timed out" in stdout:
        return "no verdict", "tester timed out"
    if "pytest: command not found" in text:
        return "no verdict", "pytest missing from the image"
    if re.search(r"^ERROR: \S*pyproject\.toml:", text, re.M):
        return "never configured", "pyproject.toml is not valid TOML"
    if "[pytest] section in setup.cfg" in text:
        return "never configured", "setup.cfg uses a removed pytest section"
    if "unrecognized arguments" in text:
        return "never configured", "pytest config needs an uninstalled plugin"

    section = _split_sections(text)
    workspace = str(trial / "artifacts" / "workspace")
    refine = lambda raised: [_refine(kind, msg, workspace) for kind, msg in raised]
    collecting = refine(_RAISED.findall(section.get("ERRORS", "")))
    running = refine(_RAISED.findall(section.get("FAILURES", "")))
    fallback = not (collecting or running)
    if fallback:
        # A conftest that fails to import is reported outside any banner, and a
        # plugin that fails to load is a bare traceback with no `E   ` at all.
        collecting = refine(_RAISED.findall(text) or _BARE.findall(text))

    if collecting or running:
        cause = collections.Counter(collecting + running).most_common(1)[0][0]
        if fallback:
            stage = "never imported" if "/workspace/" in text else "never configured"
        else:
            # Where this cause struck hardest decides the stage, so cause and
            # stage always describe the same failure rather than two of them.
            stage = ("never imported" if collecting.count(cause) >= running.count(cause)
                     else "ran, all failed")
        return stage, cause

    if "no tests ran" in text or "collected 0 items" in text:
        return "never configured", "no tests were collected"
    if "test session starts" in text and not _SUMMARY.search(text):
        # The report stops mid-progress-line with no summary: pytest was killed
        # while the suite was still running, and the 0.00 is the tester's
        # fallback rather than a measurement of the code. Some of these reports
        # show tests passing before the cut.
        return "no verdict", "test run killed mid-suite"
    return "never imported", "unclassified"


# --------------------------------------------------------------------------- #
# the arms
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Arm:
    """One code-generation method, and how to read what it uniquely records.

    `read_trial` returns the columns only this arm has. `finalize` runs once over
    the assembled frame and must set `hit_budget` and `ending` — the two places
    where "the method was stopped rather than finishing" is arm-specific.
    """

    name: str
    agent_names: tuple[str, ...]
    budget_name: str
    read_trial: Callable[[Path, dict], dict]
    finalize: Callable[[pd.DataFrame], pd.DataFrame]
    extra_columns: tuple[str, ...] = field(default=())


# --- single-shot: one request, one reply, files parsed out of it -------------

_WROTE = re.compile(r"Wrote (\d+) file")
_PROMPT_CHARS = re.compile(r": (\d+) characters in")


def _read_single_shot(trial: Path, result: dict) -> dict:
    log = read_text(trial / "agent" / "single-shot.log")
    prompt_chars = _PROMPT_CHARS.search(log)
    written = _WROTE.search(log)
    return {
        "prompt_chars": int(prompt_chars.group(1)) if prompt_chars else 0,
        "n_written": int(written.group(1)) if written else 0,
        # How often the reply's file blocks were malformed enough to notice.
        "w_unclosed": log.count("the code block was never closed"),
        "w_empty": log.count("the file block was empty"),
        "w_dupe": log.count("named more than once"),
        "w_nofence": log.count("no fenced code block"),
        # The runner's own verdict on a truncated reply that said nothing new.
        "looped": "repeated itself" in log,
    }


def _finalize_single_shot(frame: pd.DataFrame) -> pd.DataFrame:
    frame["hit_budget"] = frame.out_tok >= frame.budget_value
    frame["ending"] = frame.hit_budget.map(
        {False: "reply finished", True: "cut at the output ceiling"})
    return frame


# --- terminus-2: an agent loop with a shell and a turn budget ----------------

# The trajectory is read with this rather than json.load, because the same file
# carries the API key the run was launched with and nothing here wants it.
_TOOL_CALL = re.compile(r'"function_name":\s*"(\w+)"')


def _read_terminus(trial: Path, result: dict) -> dict:
    loop = (result.get("agent_result") or {}).get("metadata") or {}
    calls = _TOOL_CALL.findall(read_text(trial / "agent" / "trajectory.json"))
    times = loop.get("api_request_times_msec")
    return {
        "episodes": loop.get("n_episodes"),
        "summarizations": loop.get("summarization_count"),
        "median_request_ms": pd.Series(times).median() if times else float("nan"),
        "final_call": calls[-1] if calls else "",
    }


def _finalize_terminus(frame: pd.DataFrame) -> pd.DataFrame:
    # Reaching the budget is only a cut-off if the agent was still working: a
    # trial that called `mark_task_complete` on its very last allowed turn is a
    # finished run, not a truncated one.
    frame["hit_budget"] = (
        (frame.episodes >= frame.budget_value) & (frame.final_call == "bash_command"))
    frame["ending"] = frame.hit_budget.map(
        {False: "declared complete", True: "cut at turn ceiling"})
    frame.loc[~frame.scored, ["hit_budget", "ending"]] = [False, "lost to host fault"]
    return frame


# --- self-collaboration: Analyst, then Coder/Tester rounds -------------------

_STEP = re.compile(r"^\s*\[Step (\d+)/(\d+)\]\[(.*)\]\s*$", re.M)
_ROUND = re.compile(r"^\s*=== Round (\d+)/(\d+): (\w+)", re.M)
_TOOL = re.compile(r"'(\w+)'")


def _read_self_collaboration(trial: Path, result: dict) -> dict:
    """What each role did, counted off the log's own banners and step lines.

    The Coder's block is split out separately because its step budget is per
    round: a round that reached its last allowed step was stopped by the budget,
    and knowing how often that happens is the difference between "the loop ran
    out of rounds" and "every round ran out of steps".
    """
    log = read_text(trial / "agent" / "self-collaboration.log")
    coder_blocks = re.split(r"=== Round \d+/\d+: Coder \(Patcher\) ===", log)[1:]
    analyst_block = re.split(r"=== Round \d+/\d+:", log)[0]

    def steps(block):
        return [(int(a), int(b)) for a, b, _ in _STEP.findall(block)]

    tools = collections.Counter()
    for _, _, listed in _STEP.findall(log):
        tools.update(_TOOL.findall(listed))

    # `result.json` carries no reasoning-token count; the agent's usage file does.
    try:
        spend = json.loads(
            (trial / "agent" / "model-usage.json").read_text(encoding="utf-8"))
    except OSError:
        spend = {}

    return {
        "reasoning_tok": spend.get("reasoning_tokens"),
        "rounds": max((int(n) for n, _, _ in _ROUND.findall(log)), default=0),
        "coder_rounds": len(coder_blocks),
        "tester_rounds": len(re.findall(r"=== Round \d+/\d+: Tester", log)),
        "self_test_passed": "Tests PASSED!" in log,
        "self_test_failures": log.count("Tests FAILED"),
        "analyst_steps": max((a for a, _ in steps(analyst_block)), default=0),
        "coder_steps": sum(max((a for a, _ in steps(b)), default=0) for b in coder_blocks),
        # Coder rounds that used every step they were allowed.
        "coder_rounds_at_cap": sum(
            1 for b in coder_blocks if any(a == cap for a, cap in steps(b))),
        "n_bash": tools["bash"],
        "n_read": tools["read_file"],
        "n_edit": tools["edit_file"],
    }


def _finalize_self_collaboration(frame: pd.DataFrame) -> pd.DataFrame:
    # The Tester runs between Coder rounds and never after the final one, so a
    # trial that never satisfied it used every round it had.
    frame["hit_budget"] = ~frame.self_test_passed
    frame["ending"] = "used every round, never self-verified"
    passed = frame.self_test_passed
    frame.loc[passed, "ending"] = (
        "passed its own tests at round " + frame.loc[passed, "rounds"].astype(str))
    return frame



# --- CodeS: sketch the repo, sketch each file, then fill every function ------

_CODES_FILE = re.compile(
    r"function body prompts: file (\d+) of (\d+), \S+, (\d+) functions so far")
_CODES_BUDGET = re.compile(
    r"BudgetExhausted: Stopped after (\d+)\s?(?:tokens|s), at the configured (\w+) of")
# The filled sketches that came back as unparseable Python. Anchored at the start
# of a line so the interpreter's own `SyntaxWarning: invalid escape sequence`
# chatter, of which there are thousands, is not counted as a failure.
_CODES_UNPARSEABLE = re.compile(r"^(?:invalid syntax|unterminated|expected )", re.M)


def _read_codes(trial: Path, result: dict) -> dict:
    """The pipeline's own progress log: what it planned, and where it stopped.

    Only `codes.log` is read. The three `agent/codes/*.jsonl` files beside it are
    the full prompt-and-response record of every stage and run to gigabytes per
    job; the log carries the counts without any of that.
    """
    log = read_text(trial / "agent" / "codes.log")
    files = _CODES_FILE.findall(log)
    stop = _CODES_BUDGET.search(log)
    completed = "Done. Repo at:" in log
    return {
        "completed": completed,
        "requests": log.count("HTTP Request: POST"),
        # How many files the repo sketch planned, and how far the filler got.
        "planned_files": int(files[0][1]) if files else 0,
        "files_reached": int(files[-1][0]) if files else 0,
        # Function bodies queued for writing — the size of the decomposition.
        "function_prompts": int(files[-1][2]) if files else 0,
        "stopped_at": int(stop.group(1)) if stop else float("nan"),
        "stop_limit": stop.group(2) if stop else ("" if completed else "killed"),
        "unparseable": len(_CODES_UNPARSEABLE.findall(log)),
    }


def _finalize_codes(frame: pd.DataFrame) -> pd.DataFrame:
    # The pipeline either reaches its own "Done" or it is stopped, by one of its
    # two configured budgets or by the harness around it.
    frame["hit_budget"] = ~frame.completed
    frame["ending"] = "finished the whole pipeline"
    for limit, label in (("max_token_budget", "stopped at the token budget"),
                         ("max_wall_clock_seconds", "stopped at the wall clock"),
                         ("killed", "killed before it finished")):
        frame.loc[frame.stop_limit == limit, "ending"] = label
    # Some runs print the pipeline's own "Done" having produced nothing: the repo
    # sketch came back with no file tree, so there was never anything to fill.
    # That is not a finished pipeline and must not be counted as one.
    frame.loc[frame.completed & (frame.workspace_files == 0), "ending"] = "wrote nothing at all"
    return frame


ARMS = (
    Arm(
        name="single-shot",
        agent_names=("single-shot",),
        budget_name="max_tokens",
        read_trial=_read_single_shot,
        finalize=_finalize_single_shot,
        extra_columns=("prompt_chars", "n_written", "w_unclosed", "w_empty",
                       "w_dupe", "w_nofence", "looped"),
    ),
    Arm(
        name="terminus-2",
        agent_names=("terminus-2-baseline",),
        budget_name="max_turns",
        read_trial=_read_terminus,
        finalize=_finalize_terminus,
        extra_columns=("episodes", "summarizations", "median_request_ms", "final_call"),
    ),
    Arm(
        name="self-collaboration",
        agent_names=("self-collaboration",),
        budget_name="max_rounds",
        read_trial=_read_self_collaboration,
        finalize=_finalize_self_collaboration,
        extra_columns=("reasoning_tok", "rounds", "coder_rounds", "tester_rounds",
                       "self_test_passed",
                       "self_test_failures", "analyst_steps", "coder_steps",
                       "coder_rounds_at_cap", "n_bash", "n_read", "n_edit"),
    ),
    Arm(
        name="codes",
        agent_names=("codes",),
        budget_name="max_token_budget",
        read_trial=_read_codes,
        finalize=_finalize_codes,
        extra_columns=("completed", "requests", "planned_files", "files_reached",
                       "function_prompts", "stopped_at", "stop_limit", "unparseable"),
    ),
)

_BY_AGENT = {agent: arm for arm in ARMS for agent in arm.agent_names}


def arm_for(agent_name: str) -> Arm:
    """The arm an `agent_info.name` belongs to."""
    try:
        return _BY_AGENT[agent_name]
    except KeyError:
        raise KeyError(
            f"no arm registered for agent {agent_name!r}; "
            f"known: {sorted(_BY_AGENT)}") from None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def job_path(job) -> Path:
    """A job directory, given either its name under `jobs/` or a path."""
    path = Path(job)
    return path if path.is_absolute() or path.exists() else JOBS / str(job)


@functools.lru_cache(maxsize=None)
def load_job(job, *, solved_at: float = 0.9, metadata_csv=None) -> pd.DataFrame:
    """One row per trial of one job, with `COMMON_COLUMNS` plus that arm's own.

    The arm is detected from the job's own `agent_info.name`. `solved_at` is the
    reward at or above which a task counts as solved rather than partially
    credited — an analysis choice, so it is a parameter here rather than a
    constant. Every budget is read from each trial's configuration instead.
    """
    root = job_path(job)
    rows, arm, budgets = [], None, set()

    for trial in sorted(p for p in root.iterdir() if p.is_dir()):
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        if arm is None:
            arm = arm_for(result["agent_info"]["name"])

        verdict = result.get("verifier_result")
        reward = verdict["rewards"]["reward"] if verdict else float("nan")
        usage = result.get("agent_result") or {}
        kwargs = result["config"]["agent"]["kwargs"]
        budgets.add(kwargs.get(arm.budget_name))

        # A zero has to be explained, so its report is read whole; for the rest
        # the summary line at the end is all that is wanted, and one of these
        # files is 6 MB of tracebacks.
        tests = read_text(trial / "verifier" / "test-output.txt",
                          tail_bytes=None if reward == 0 else 32_768)
        stdout = read_text(trial / "verifier" / "test-stdout.txt")

        row = {
            "arm": arm.name,
            "trial": trial.name,
            "task": result["task_id"]["name"],
            "reward": reward,
            "scored": verdict is not None,
            "solved": bool(reward >= solved_at),   # NaN >= x is False, as wanted
            "cost_usd": usage.get("cost_usd"),
            "in_tok": usage.get("n_input_tokens"),
            "cache_tok": usage.get("n_cache_tokens"),
            "out_tok": usage.get("n_output_tokens"),
            "total_tok": (usage.get("n_input_tokens") or 0)
                         + (usage.get("n_output_tokens") or 0) or None,
            "t_total": seconds({"started_at": result["started_at"],
                                "finished_at": result["finished_at"]}),
            "t_agent": seconds(result.get("agent_execution")),
            "t_verify": seconds(result.get("verifier")),
            "workspace_files": _workspace_files(str(trial / "artifacts" / "workspace")),
            "budget_name": arm.budget_name,
            "budget_value": kwargs.get(arm.budget_name),
            "exception": (result["exception_info"] or {}).get("exception_type"),
            # Harness faults: neither the model's doing nor the verifier's finding.
            "no_pytest": "pytest: command not found" in tests,
            "tester_timeout": "Tester timed out" in stdout,
        }
        row.update(pytest_outcome(tests))
        row.update(arm.read_trial(trial, result))
        row["fault_stage"], row["fault_cause"] = (
            classify_zero(trial, tests, stdout) if reward == 0 else ("", ""))
        rows.append(row)

    frame = arm.finalize(pd.DataFrame(rows))
    frame = _join_metadata(frame, metadata_csv)
    frame["outcome"] = frame.apply(_outcome, axis=1)
    frame.attrs["arm"] = arm
    frame.attrs["budget"] = budgets.pop() if len(budgets) == 1 else sorted(budgets)
    frame.attrs["solved_at"] = solved_at
    return frame


def load_jobs(jobs: dict, **kwargs) -> pd.DataFrame:
    """Several arms' jobs in one frame, keyed by the label you give each.

    The label overrides the detected arm name, so a comparison can run two jobs
    of the same arm side by side — one per configuration, say.
    """
    frames = []
    for label, job in jobs.items():
        frame = load_job(job, **kwargs).copy()
        frame["arm"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _join_metadata(frame: pd.DataFrame, metadata_csv=None) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv or TASK_METADATA,
                       usecols=["task", "difficulty", "n_hidden_tests", "spec_words"])
    frame = frame.merge(meta, on="task", how="left")
    frame["difficulty"] = pd.Categorical(frame["difficulty"], LEVELS, ordered=True)
    return frame


def _outcome(row) -> str:
    """What the trial produced, for the budget view.

    A zero is reported by the stage it failed at rather than as one bucket, so
    this cut and the cause counts tell the same story at two resolutions.
    """
    if not row.scored:
        return "lost to host fault"
    if row.solved:
        return "solved"
    if row.reward > 0:
        return "partial credit"
    return row.fault_stage


# --------------------------------------------------------------------------- #
# small summaries the notebooks share
# --------------------------------------------------------------------------- #

def headline(frame: pd.DataFrame) -> str:
    """The four numbers every run notebook opens with."""
    scored = frame[frame.scored]
    lost = len(frame) - len(scored)
    lines = [
        f"trials          : {len(frame)}"
        + (f"  ({len(scored)} scored, {lost} lost to a host fault)" if lost else ""),
        f"mean reward     : {scored.reward.mean():.3f}   median"
        f" {scored.reward.median():.3f}   (n={len(scored)})",
        f"total tokens    : {fmt_tokens(frame.total_tok.sum())}"
        f"  ({frame.cache_tok.sum() / frame.in_tok.sum():.0%} of input served from cache)",
        f"total wall clock: {frame.t_total.sum() / 3600:.1f} h summed over trials",
    ]
    return "\n".join(lines)


def zero_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """How much of each difficulty band scored nothing."""
    table = (frame[frame.scored].assign(zero=lambda f: f.reward == 0)
             .groupby("difficulty", observed=True).zero.agg(["sum", "size", "mean"]))
    table.columns = ["trials at zero", "scored trials", "share of the level"]
    return table


def token_quartile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Trials in four equal groups by tokens, with what each returned and used."""
    labels = ("Q1\nleanest", "Q2", "Q3", "Q4\nheaviest")
    binned = frame.assign(quartile=pd.qcut(frame.total_tok, 4, labels=labels))
    group = binned.groupby("quartile", observed=True).agg(
        mean_reward=("reward", "mean"), tokens=("total_tok", "sum"),
        n=("reward", "size"))
    group["share"] = group.tokens / frame.total_tok.sum()
    return group


def by_difficulty(frame: pd.DataFrame, **extra) -> pd.DataFrame:
    """Median tokens and mean reward per level, plus any arm-specific column.

        ja.by_difficulty(df, median_turns=("episodes", "median"))
    """
    return frame.groupby("difficulty", observed=True).agg(
        trials=("reward", "size"), median_tok=("total_tok", "median"),
        mean_reward=("reward", "mean"), **extra)


def cause_rows(frame: pd.DataFrame) -> list[tuple[str, int, str]]:
    """`(cause, count, stage)` for every zero, in pipeline order then by count."""
    zeros = frame[frame.scored & (frame.reward == 0)]
    rows = []
    for stage in STAGES:
        counts = zeros[zeros.fault_stage == stage].fault_cause.value_counts()
        rows += [(cause, int(n), stage) for cause, n in counts.items()]
    return rows


# --------------------------------------------------------------------------- #
# comparing arms
# --------------------------------------------------------------------------- #

#: Causes a single execution of the generated code would have surfaced at once:
#: the file does not parse, a name is not defined, or the package cannot import
#: itself. None of these needs a test suite or a specification to find — running
#: the code once is enough — so they separate methods that execute what they
#: write from methods that only emit it.
EXECUTION_CATCHABLE = ("SyntaxError", "IndentationError", "NameError",
                       "ImportError: circular import")


def arm_order(frame: pd.DataFrame) -> list:
    """The arms in the order they were loaded, which is the order to plot them."""
    return list(dict.fromkeys(frame.arm))


def paired(frame: pd.DataFrame, values: str = "reward") -> pd.DataFrame:
    """Tasks as rows, arms as columns, restricted to tasks every arm scored.

    Every arm ran the same benchmark, so the honest comparison is per task and
    not between two marginal distributions. Dropping the tasks some arm never
    scored costs a few rows and removes the possibility that one arm's mean is
    flattered by a task another arm lost to a host fault.

    Which tasks those are is always decided by the reward, whatever column is
    being asked for. A trial lost to a host fault has no reward but does have a
    `solved` of False, so pivoting a boolean column on its own would silently
    compare a different — and larger — set of tasks.
    """
    arms = arm_order(frame)
    rewards = (frame.pivot_table(index="task", columns="arm", values="reward",
                                 observed=True).reindex(columns=arms))
    keep = rewards.dropna().index
    if values == "reward":
        return rewards.loc[keep]
    wide = frame.pivot_table(index="task", columns="arm", values=values,
                             observed=True)
    return wide.reindex(columns=arms).loc[keep]


def compare(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: what it scored, what it consumed, and what that came to.

    Consumption is counted in tokens rather than in money. A bill depends on
    which model was billed and at what price, and on how much of the input was
    served from cache at a discount; a token count does not, so it is the
    measure that still compares two arms run against different models or at
    different times. `cost_usd` is still on the frame for anyone who wants it.

    Two denominators are given because they disagree. `total_per_solved` counts
    everything the arm read and wrote, and so charges an agent loop for
    re-reading its own context every turn. `out_per_solved` counts only what the
    model generated. An arm can lead on one and trail on the other.

    `usage_reported` says how many trials recorded usage at all — one arm has
    trials that recorded none, so its totals are a floor rather than a total.
    """
    rows = {}
    for arm in arm_order(frame):
        part = frame[frame.arm == arm]
        scored = part[part.scored]
        solved = int(part.solved.sum())
        total, out = part.total_tok.sum(), part.out_tok.sum()
        per = lambda n: n / solved if solved else float("nan")
        rows[arm] = {
            "trials": len(part),
            "scored": len(scored),
            "mean_reward": scored.reward.mean(),
            "median_reward": scored.reward.median(),
            "solved": solved,
            "zeros": int((scored.reward == 0).sum()),
            "in_tok": part.in_tok.sum(),
            "out_tok": out,
            "total_tok": total,
            "cache_share": part.cache_tok.sum() / part.in_tok.sum(),
            "total_per_solved": per(total),
            "out_per_solved": per(out),
            "usage_reported": int(part.total_tok.notna().sum()),
            "median_minutes": part.t_total.median() / 60,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fmt_tokens(n) -> str:
    """A token count as a short string: `408.7M`, `50M`, `12.3k`, `940`.

    A whole number keeps no decimal, so an axis of round ticks reads `50M` and
    not `50.0M` while a single value still keeps its precision.
    """
    if pd.isna(n):
        return "n/a"
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            text = f"{n / scale:.1f}"
            return (text[:-2] if text.endswith(".0") else text) + suffix
    return f"{n:.0f}"


def head_to_head(frame: pd.DataFrame) -> pd.DataFrame:
    """Tasks on which the row arm scored strictly above the column arm."""
    wide = paired(frame)
    arms = list(wide.columns)
    return pd.DataFrame(
        [[pd.NA if a == b else int((wide[a] > wide[b]).sum()) for b in arms]
         for a in arms], index=arms, columns=arms, dtype="Int64")


def oracle(frame: pd.DataFrame) -> pd.DataFrame:
    """What picking the best arm per task would score, and what each arm adds.

    The first row is that upper bound. Each remaining row removes one arm and
    recomputes it, so `reward_lost` is the arm's marginal contribution to the
    set — what would be given up by not having run it at all. An arm that never
    beats the others on any task contributes exactly nothing here, however well
    it does in isolation.
    """
    wide = paired(frame)
    solved = paired(frame, values="solved").astype(bool)
    sets = {"all arms": list(wide.columns)}
    sets.update({f"without {arm}": [c for c in wide.columns if c != arm]
                 for arm in wide.columns})
    table = pd.DataFrame({
        "mean_reward": {k: wide[cols].max(axis=1).mean() for k, cols in sets.items()},
        "solved": {k: int(solved[cols].any(axis=1).sum()) for k, cols in sets.items()},
    })
    table["reward_lost"] = table.mean_reward["all arms"] - table.mean_reward
    table["solved_lost"] = table.solved["all arms"] - table.solved
    return table


def solved_alone(frame: pd.DataFrame) -> pd.DataFrame:
    """Per arm, tasks it solved that no other arm did, against tasks it shared."""
    solved = paired(frame, values="solved").astype(bool)
    others = {arm: solved[[c for c in solved.columns if c != arm]].any(axis=1)
              for arm in solved.columns}
    return pd.DataFrame({
        "only this arm": {a: int((solved[a] & ~others[a]).sum()) for a in solved},
        "also solved elsewhere": {a: int((solved[a] & others[a]).sum()) for a in solved},
    }).loc[list(solved.columns)]


def outcome_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """Trials per outcome, arms as columns, best outcome first."""
    table = frame.pivot_table(index="outcome", columns="arm", values="reward",
                              aggfunc="size", observed=True)
    return (table.reindex(index=OUTCOMES).reindex(columns=arm_order(frame))
            .fillna(0).astype(int))


def cause_by_arm(frame: pd.DataFrame, causes=None) -> pd.DataFrame:
    """Zero counts as causes by arms, for `causes` or for every cause seen."""
    zeros = frame[frame.scored & (frame.reward == 0)]
    table = zeros.pivot_table(index="fault_cause", columns="arm", values="reward",
                              aggfunc="size", observed=True)
    table = table.reindex(columns=arm_order(frame))
    if causes is not None:
        table = table.reindex(index=list(causes))
    return table.fillna(0).astype(int)
