"""Single-Shot, run inside the task container.

This is the baseline arm, and the only one in this platform that is not a
published tool: one request carries the task's specification and the
instruction that says how to lay a repository out in a reply, and the files
that reply names are written into the workspace. Nothing plans, nothing tests,
and nothing reads the result back.

Two decisions make it a control rather than a fourth solution.

The prompt is fixed here rather than exposed as a hyperparameter. Its only
content beyond the task's own specification is the mechanical contract for
naming files, which exists because the benchmark grades files on disk while a
model emits text. A baseline whose wording is a knob would be tuned, and a
tuned baseline is no longer the thing the other solutions are measured
against. Its digest is recorded with every run, so a result can be checked
against the prompt that produced it.

The reply is parsed, never repaired. A response that stopped at the token
ceiling is recorded as truncated, and a section that could not be read as a
file is recorded as a warning, because a baseline that ran out of output
tokens and a baseline that did not know the answer are different findings that
one reward figure cannot tell apart.

Two benchmarks ask two different questions, and the arm answers whichever one
it was pointed at. NL2RepoBench mounts an empty workspace and supplies a
specification, so the request carries the specification alone and the reply is
a whole project. SWE-Bench-Pro ships the repository inside the task's own image
and supplies a description of a change wanted in it, so the repository is part
of the task's input rather than something a method goes and finds: the request
carries it, and the reply is the files that change.

What goes into that second prompt is assembled here rather than chosen, which
is the line between apparatus and method. The runner reads the paths the task's
own text names and lists what the repository tracks; it does not search, rank
or embed, because deciding *which* code the model should see is retrieval, and
a control that retrieves is no longer the thing the solutions are measured
against. What was assembled, and what the budget cut, is recorded per run.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

# The runner is uploaded next to its own dependencies, so its directory carries
# the shared usage accounting under a flat name. On the host, where the tests
# import this module from the package, that same directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_categories import FailureTally  # noqa: E402
from model_usage import UsageTotals, record_response_usage  # noqa: E402
from model_routing import (  # noqa: E402
    ServedProviders,
    routing_from_env,
    with_routing,
)

# Where the benchmark grades, and so the only place this runner writes. It is
# a module global rather than a constant because the two benchmarks disagree
# about it — NL2RepoBench mounts `/workspace`, SWE-Bench-Pro ships the
# repository at `/app` — and `main` rebinds it from the run's hyperparameters
# before anything reads it.
WORKSPACE = Path("/workspace")
RESPONSE_DIR = Path("/logs/agent/single-shot")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")
# What a throttled attempt waits before the next one. A refused request — a
# rejected credential, an unknown model — is refused just as quickly the second
# time and is not worth waiting for.
RATE_LIMIT_WAITS = (15, 30)
# Path roots the writer will not accept from the model, whatever it says.
# `.git` is the workspace's own bookkeeping rather than part of the project.
FORBIDDEN_ROOTS = {".git"}

# The whole of this baseline's scaffolding. It says how to lay files out in a
# reply and nothing about how to solve anything: no role, no plan, no
# suggestion to test, and no worked example beyond the two lines needed to
# show the shape.
FORMAT_INSTRUCTION = """\
Write out the complete project, one file at a time. Introduce each file with a
line naming its path, then give that file's entire contents in a fenced code
block:

### FILE: src/example/core.py
```python
def add(left: int, right: int) -> int:
    return left + right
```

- Paths are relative to the project root: `pyproject.toml` belongs at
  `pyproject.toml`, not inside a directory named after the project.
- Where the specification shows a directory tree, its top-level directory is
  the project root itself and not a directory to create.
- No absolute paths, and no `..`.
- Include every file the project needs in order to be installed and used,
  packaging and configuration among them.
- Write each file out in full. Do not abbreviate a body to a comment, and do
  not leave a placeholder to be filled in later.
- If a file's own contents contain a fenced code block, fence that file with
  more backticks than appear anywhere inside it.
- Only the file blocks are read. Any other text is ignored.
"""

# The same scaffolding for a benchmark that hands over a repository instead of
# an empty directory. It says how to lay files out and which files to lay out,
# and still nothing about how to solve anything. The two sentences the first
# contract does not need are the two a repair benchmark makes load-bearing: a
# file arrives as a whole or not at all, so an abbreviated body deletes the code
# it stands in for, and a file nobody names is left alone.
#
# It opens by saying what the reply is, which is a correction rather than an
# instruction. SWE-Bench-Pro's task text was written for an agent holding a
# shell — it asks in so many words for a script to be run "using the bash tool",
# for its output to be checked, and for the answer to be as long as it needs to
# be. For an arm with no tools that is a false premise, and the first run
# against this benchmark showed what it costs: the model wrote out 2 679 bash
# invocations, waited for output that no one was going to send, and repeated two
# lines until it reached the token ceiling — 131 100 tokens, thirteen minutes,
# and not one file. Saying what the interface is says nothing about how to solve
# anything, which is what keeps this a control; leaving it unsaid measures the
# benchmark's phrasing rather than the model.
EDIT_FORMAT_INSTRUCTION = """\
You are answering in one reply and you have no tools. Nothing you write is
executed, no command's output comes back to you, and there is no second turn:
what is above is everything you will be shown of this repository. Where the
task suggests searching the code, running a script, or checking its output, it
is describing a setting this reply is not in. Do not write commands out as
though they were going to run, and do not wait for their results.

The repository is already on disk. Change it by writing out each file you
change, one at a time: a line naming its path, then that file's entire new
contents in a fenced code block:

### FILE: src/example/core.py
```python
def add(left: int, right: int) -> int:
    return left + right
```

- Paths are relative to the repository root.
- No absolute paths, and no `..`.
- Give each file's entire new contents, from its first line to its last. What
  you write replaces the file, so a fragment, a diff, an ellipsis, or a comment
  standing in for code that was already there deletes the rest of that file.
- Name only the files you change or add. A file you do not name is left exactly
  as it is.
- Do not write test files. The tests this work is judged by are already in
  place.
- If a file's own contents contain a fenced code block, fence that file with
  more backticks than appear anywhere inside it.
- Only the file blocks are read. Any other text is ignored.
"""

# Which question the benchmark is asking, and the contract each one is answered
# under. `evaluation_platform.experiment_config.TASK_SHAPES` is the same list on
# the configuration side, and a shape absent here would be rejected there.
REPOSITORY_SHAPE = "repository"
EDIT_SHAPE = "edit"
FORMAT_INSTRUCTIONS = {
    REPOSITORY_SHAPE: FORMAT_INSTRUCTION,
    EDIT_SHAPE: EDIT_FORMAT_INSTRUCTION,
}

# The suffixes a path has to carry to be read as a source path out of a task's
# own text. A list rather than "anything with a dot" because a PR description is
# prose: it names versions, hosts and command flags, and every one of those that
# resolved to a file would spend the context budget on something the task did
# not point at.
SOURCE_SUFFIXES = (
    "bash", "c", "cc", "cfg", "cjs", "conf", "cpp", "cs", "css", "go", "h",
    "hpp", "html", "ini", "java", "js", "json", "jsx", "kt", "less", "md",
    "mjs", "php", "proto", "py", "rb", "rs", "scss", "sh", "sql", "svelte",
    "toml", "tmpl", "ts", "tsx", "txt", "vue", "yaml", "yml",
)
_INSTRUCTION_PATH = re.compile(
    r"(?<![\w/.-])[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:"
    + "|".join(SOURCE_SUFFIXES)
    + r")(?![A-Za-z0-9_])"
)
# How the budget is split between the two halves of a repository context. The
# files the task named are the half it pointed at, so they are served first; the
# listing is what is left, and it degrades to a shorter listing rather than
# displacing a file. A fixed split rather than a tuned one: it is the shape of
# the apparatus, not a setting to search over.
NAMED_FILE_SHARE = 0.75
# The fence the context puts around a file it quotes. Long enough that a file
# containing ordinary Markdown does not close it early.
_CONTEXT_FENCE = "`" * 6
# How long a `git` call may take before the context is assembled without it.
_GIT_TIMEOUT_SECONDS = 120

_FILE_HEADER = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?FILE:[ \t]*(?P<path>[^\n]+?)[ \t]*$",
    re.MULTILINE,
)
_FENCE_OPEN = re.compile(r"^(?P<fence>`{3,}|~{3,})[^\n]*$", re.MULTILINE)


def main() -> None:
    global WORKSPACE

    hyperparameters = _read_hyperparameters()
    WORKSPACE = Path(hyperparameters["workspace"])
    shape = hyperparameters["task_shape"]
    format_instruction = FORMAT_INSTRUCTIONS[shape]
    instruction = _read_instruction()
    context, context_record = (
        build_repository_context(
            WORKSPACE, instruction, hyperparameters["context_chars"]
        )
        if shape == EDIT_SHAPE
        else ("", None)
    )
    prompt = build_prompt(instruction, format_instruction, context)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    (RESPONSE_DIR / "prompt.md").write_text(prompt, encoding="utf-8")

    totals = UsageTotals()
    routing = routing_from_env()
    observed = ServedProviders()
    failures = FailureTally()
    try:
        answer, finish_reason = _request(
            _client(hyperparameters["request_timeout_seconds"]),
            prompt,
            hyperparameters,
            totals,
            os.environ["MODEL"],
            routing=routing,
            observed=observed,
            failures=failures,
        )
    finally:
        USAGE_PATH.write_text(
            json.dumps(totals.to_dict(), indent=2), encoding="utf-8"
        )
    (RESPONSE_DIR / "response.md").write_text(answer, encoding="utf-8")

    shape_of_reply = reply_shape(answer)
    files, warnings = parse_files(answer)
    files, stripped_root = _strip_workspace_root(files)
    written = _write_files(files)
    _record_resolved_setup(
        hyperparameters=hyperparameters,
        format_instruction=format_instruction,
        finish_reason=finish_reason,
        written=written,
        warnings=warnings,
        requests=totals.responses,
        stripped_root=stripped_root,
        routing=routing,
        providers=observed.names(),
        failures=failures,
        context=context_record,
        change=record_change(WORKSPACE) if shape == EDIT_SHAPE else None,
        reply=shape_of_reply,
    )

    print(f"Wrote {len(written)} file(s) to {WORKSPACE}.")
    for warning in warnings:
        print(f"Warning: {warning}")
    if shape_of_reply["degenerate"]:
        print(
            f"The reply repeated itself: {shape_of_reply['lines']} lines, of "
            f"which {shape_of_reply['distinct_lines']} were distinct. Whatever "
            "it spent, it stopped saying anything new early on, so read this "
            "as a run that looped rather than as one that needed more room.",
            file=sys.stderr,
        )
    elif finish_reason == "length":
        print(
            "The reply stopped at the token ceiling, so the project is "
            "truncated. Raise the 'max_tokens' hyperparameter to give the "
            "baseline room for a repository of this size.",
            file=sys.stderr,
        )
    if not written:
        print(
            "No file was written: the reply named none in the expected form. "
            "It is kept verbatim at /logs/agent/single-shot/response.md.",
            file=sys.stderr,
        )


def build_prompt(
        instruction: str,
        format_instruction: str = FORMAT_INSTRUCTION,
        context: str = "",
) -> str:
    """The one message the baseline sends: the task, then how to answer it.

    Where a benchmark supplies a repository as well as a task, it goes between
    the two, in the order the trial has them: this is the work, this is what it
    is being done to, this is how to write the answer down.
    """
    parts = [instruction.strip()]
    if context.strip():
        parts.append(context.strip())
    parts.append(format_instruction.strip())
    return "\n\n---\n\n".join(parts) + "\n"


def reply_shape(answer: str) -> dict[str, Any]:
    """How much of the reply was new, for telling two ceilings apart.

    A reply that reached the token cap is recorded as truncated, and until now
    that was the whole of what the record said about it. Two very different
    things arrive under that one word. One is a model that was writing a large
    project and ran out of room, where the ceiling is the finding and a higher
    one would help. The other is a model that fell into a loop and spent the
    budget repeating itself, where the ceiling is incidental and raising it buys
    more of the same.

    The first run against SWE-Bench-Pro was the second kind: 10 709 lines of
    which 57 were distinct. Counting distinct lines separates them at a glance
    and costs nothing, which is worth more here than a cleverer measure — this
    is a description of what came back, not a judgement about it, and nothing in
    the run behaves differently because of it.
    """
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    distinct = len(set(lines))
    return {
        "chars": len(answer),
        "lines": len(lines),
        "distinct_lines": distinct,
        # The share of the reply that said something it had not already said.
        # Near 1.0 for prose or code, near 0 for a loop.
        "novelty": round(distinct / len(lines), 4) if lines else None,
        "degenerate": _is_degenerate(len(lines), distinct),
    }


# What counts as a reply that stopped saying anything new. Deliberately blunt:
# a long reply with almost no distinct lines in it. Short replies are exempt
# because a handful of repeated lines is ordinary in code — imports, closing
# braces, blank-separated one-liners — and only becomes a finding at length.
DEGENERATE_MIN_LINES = 200
DEGENERATE_NOVELTY = 0.05


def _is_degenerate(lines: int, distinct: int) -> bool:
    return lines >= DEGENERATE_MIN_LINES and distinct <= lines * DEGENERATE_NOVELTY


def build_repository_context(
        workspace: Path,
        instruction: str,
        budget: int,
) -> tuple[str, dict[str, Any]]:
    """Put the repository the task is about in front of the one request.

    A repair benchmark's input is a repository and a description of a change
    wanted in it. The solutions reach the first half through tools; this arm has
    one request, so the half that is on disk has to travel in the prompt or not
    at all, and an arm that answered without it would be measuring what a model
    can invent rather than what it can fix.

    What travels is assembled, not chosen. Two things go in, in this order: the
    full contents of every source path the task's own text names, and a listing
    of what the repository tracks. Neither is a judgement about relevance — the
    first is transcription of the instruction, the second is `git ls-files` — so
    the arm stays a control. Both halves are cut to fit the budget from the
    bottom, the named files first served and the listing shortened to whatever
    is left, and every cut is recorded, so a poor result can be read against
    what the model was actually shown.
    """
    named = _paths_named_by(instruction, workspace)
    quoted, quoted_record, spent = _quote_files(
        workspace, named, int(budget * NAMED_FILE_SHARE)
    )
    paths, listed_by_git = _tracked_paths(workspace)
    listing, listing_record = _listing(paths, max(0, budget - spent))

    sections = [
        f"The repository this task is about is on disk at {workspace}. What "
        "follows was read from it and is not part of the task's own text.",
    ]
    if quoted:
        sections.append("## The files the task names\n\n" + "\n\n".join(quoted))
    if listing:
        sections.append("## What the repository contains\n\n" + listing)

    context = "\n\n".join(sections)
    return context, {
        "workspace": str(workspace),
        "budget_chars": budget,
        "chars": len(context),
        "named_paths": named,
        "quoted_files": quoted_record,
        "listing": listing_record
        | {"source": "git ls-files" if listed_by_git else "filesystem walk"},
    }


def _paths_named_by(instruction: str, workspace: Path) -> list[str]:
    """The source paths a task's own text points at, as the repository has them.

    A description that says `lib/ansible/module_utils/facts/network/linux.py`
    has named a file, and a benchmark whose instructions carry a "New interfaces
    introduced" section names one about half the time. Anything the text mentions
    that the repository does not have is dropped rather than guessed at, and a
    match is kept only where it resolves to a regular file inside the workspace:
    the instruction is text supplied to this process, and a path that climbs out
    of the workspace is not a file this run may read.
    """
    seen: dict[str, None] = {}
    for match in _INSTRUCTION_PATH.finditer(instruction):
        candidate = match.group(0).strip("./")
        if not candidate or candidate in seen:
            continue
        if _readable(workspace, candidate) is not None:
            seen[candidate] = None
    return list(seen)


def _readable(workspace: Path, path: str) -> Path | None:
    """The file `path` names inside `workspace`, or None if it names none."""
    candidate = PurePosixPath(path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (workspace / candidate).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        return None
    return resolved if resolved.is_file() else None


def _quote_files(
        workspace: Path,
        paths: list[str],
        budget: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Quote each named file whole, and drop the ones that do not fit.

    A file cut in half is worse than a file left out: nothing distinguishes code
    that is absent from code that ends there, and a replacement written against
    a truncated original deletes the tail. So a file that does not fit is omitted
    and recorded as omitted, and the ones that do fit arrive whole.
    """
    quoted: list[str] = []
    record: list[dict[str, Any]] = []
    spent = 0
    for path in paths:
        resolved = _readable(workspace, path)
        if resolved is None:
            continue
        try:
            contents = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            record.append({"path": path, "included": False, "reason": str(error)})
            continue
        fence = _fence_for(contents)
        section = f"### {path}\n{fence}\n{contents.rstrip()}\n{fence}"
        if spent + len(section) > budget:
            record.append(
                {
                    "path": path,
                    "included": False,
                    "chars": len(contents),
                    "reason": "did not fit the context budget",
                }
            )
            continue
        quoted.append(section)
        record.append({"path": path, "included": True, "chars": len(contents)})
        spent += len(section)
    return quoted, record, spent


def _fence_for(contents: str) -> str:
    """A fence longer than any run of backticks the quoted file contains."""
    longest = max((len(run) for run in re.findall(r"`+", contents)), default=0)
    return "`" * max(len(_CONTEXT_FENCE), longest + 1)


def _tracked_paths(workspace: Path) -> tuple[list[str], bool]:
    """Every path the repository tracks, and whether git was what said so.

    `git ls-files` is the listing that matches what the benchmark grades: it
    leaves out build output, virtual environments and caches, which in a
    repository of this size are most of the files and none of the code. Where
    there is no git — a workspace that is a plain directory — the filesystem
    answers instead, and the record says which one did.
    """
    listed = _git(workspace, ["ls-files", "-z"])
    if listed is not None:
        return _split_nul(listed), True

    paths = []
    for path in workspace.rglob("*"):
        if path.is_file() and not any(part in FORBIDDEN_ROOTS for part in path.parts):
            paths.append(str(path.relative_to(workspace)).replace("\\", "/"))
    return sorted(paths), False


def _split_nul(listed: str) -> list[str]:
    """Sorted paths out of a `-z` listing.

    `-z` rather than lines because git's default is to quote any path it
    considers unusual — anything non-ASCII among them — into a C string with
    octal escapes. A repository the size of ansible has a handful, and read as
    lines they arrive as mangled paths that do not exist.
    """
    return sorted(path for path in listed.split("\0") if path.strip())


def _listing(paths: list[str], budget: int) -> tuple[str, dict[str, Any]]:
    """The most detailed complete listing of the repository the budget holds.

    Cutting an alphabetical path list at the budget would be the worst of both:
    a section headed "what the repository contains" that in fact contains the
    first directory of it, with everything after `bin/` silently absent. So the
    listing loses detail rather than coverage. Every path if they fit; otherwise
    directories at the deepest level that fits, each with the number of files
    under it, so a repository of fifteen thousand files still arrives whole at
    the shape a reader needs to find their way around it.
    """
    if not paths or budget <= 0:
        return "", {"form": "none", "lines": 0, "paths_total": len(paths)}

    full = "\n".join(paths)
    if len(full) <= budget:
        return full, {
            "form": "paths",
            "lines": len(paths),
            "paths_total": len(paths),
        }

    deepest = max(path.count("/") for path in paths)
    for depth in range(deepest, 0, -1):
        text, lines = _directory_listing(paths, depth)
        if len(text) <= budget:
            return text, {
                "form": f"directories to depth {depth}",
                "lines": lines,
                "paths_total": len(paths),
            }

    # A repository whose top level alone does not fit. Nothing here is a
    # complete listing any more, so it is cut and said to be cut.
    text, lines = _directory_listing(paths, 1)
    kept = text[:budget].rsplit("\n", 1)[0]
    return kept + "\n\n(cut at the context budget.)", {
        "form": "directories to depth 1, cut",
        "lines": kept.count("\n") + 1,
        "paths_total": len(paths),
    }


def _directory_listing(paths: list[str], depth: int) -> tuple[str, int]:
    """The repository rolled up to `depth`, with a file count per directory.

    A path shallower than `depth` is named outright rather than counted: it is
    already as specific as this listing gets.
    """
    counts: dict[str, int] = {}
    for path in paths:
        parts = path.split("/")
        key = "/".join(parts[:depth]) + "/" if len(parts) > depth else path
        counts[key] = counts.get(key, 0) + 1

    lines = [
        f"{key} ({counts[key]} file{'' if counts[key] == 1 else 's'})"
        if key.endswith("/")
        else key
        for key in sorted(counts)
    ]
    return "\n".join(lines), len(lines)


def record_change(workspace: Path) -> dict[str, Any]:
    """What the run did to the repository, kept beside the reply that did it.

    A generation benchmark archives the workspace, because the workspace *is*
    the answer. Here the workspace is somebody's repository with a few files
    changed in it, and archiving gigabytes to record a few hundred lines would be
    recording the wrong thing. The diff is the answer, so the diff is what is
    kept.

    Read-only, deliberately. Staging the changes so that new files show up in
    `git diff` would leave the repository in a state the verifier did not put it
    in, so added files are listed by name instead and the index is not touched.
    """
    diff = _git(workspace, ["diff"])
    added = _git(workspace, ["ls-files", "--others", "--exclude-standard", "-z"])
    if diff is None and added is None:
        return {"available": False}

    if diff:
        (RESPONSE_DIR / "patch.diff").write_text(diff, encoding="utf-8")
    return {
        "available": True,
        "patch_chars": len(diff or ""),
        "files_modified": (diff or "").count("diff --git "),
        "files_added": _split_nul(added or ""),
    }


def _git(workspace: Path, arguments: list[str]) -> str | None:
    """Run one read-only git command in the workspace, or None if it cannot.

    Never fatal. A missing git, a directory that is not a repository, or a
    command that takes too long costs the run a listing or a diff; it does not
    cost the run the one request it exists to make.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"git {' '.join(arguments)} was unavailable: {error}", file=sys.stderr)
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_files(answer: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Read the reply as a sequence of named files.

    Tolerant on purpose, in one direction only. A reply that fences its files
    unevenly, or names one without fencing it at all, is still read for the
    code it contains, because discarding a section the model did write would
    understate the baseline. Nothing is inferred that the reply did not say: a
    section with no file name is not a file, and no contents are invented for
    one that named a file and then wrote nothing.
    """
    headers = list(_FILE_HEADER.finditer(answer))
    files: list[tuple[str, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for index, header in enumerate(headers):
        path = _strip_decoration(header.group("path"))
        end = headers[index + 1].start() if index + 1 < len(headers) else len(answer)
        body, warning = _extract_body(answer[header.end():end])
        if warning is not None:
            warnings.append(f"{path or 'unnamed file'}: {warning}")
        if not path:
            warnings.append("a file block named no path and was skipped")
            continue
        if not body.strip():
            warnings.append(f"{path}: the file block was empty and was skipped")
            continue
        if path in seen:
            warnings.append(f"{path}: named more than once; the last one is kept")
        seen.add(path)
        files.append((path, body))

    return files, warnings


def _extract_body(section: str) -> tuple[str, str | None]:
    """The contents of the first fenced block in one file's section."""
    opening = _FENCE_OPEN.search(section)
    if opening is None:
        return (
            section.strip("\n"),
            "no fenced code block; read the section as written",
        )

    fence = opening.group("fence")
    # CommonMark closes a fence with at least as many of the same character as
    # opened it, which is what lets a Markdown file containing three backticks
    # be delivered inside four.
    closing = re.compile(
        rf"^{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$", re.MULTILINE
    )
    start = min(opening.end() + 1, len(section))
    match = closing.search(section, start)
    if match is None:
        return section[start:], "the code block was never closed"
    return section[start:match.start()], None


def _strip_decoration(path: str) -> str:
    return path.strip().strip("`*\"'").strip()


def _strip_workspace_root(
        files: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], str | None]:
    """Drop a leading `workspace/` the reply copied out of the specification.

    This is a correction for an artefact of the apparatus, not a helping hand.
    A benchmark that shows the project as a directory tree shows it rooted at
    the directory the task mounts it in, and a model writing paths as *text* can
    reproduce that root as though it were part of the project. The solutions
    never face this: they write through tools into the working directory, so the
    root never materializes. Left uncorrected it would put the package one level
    below where the tester looks and score a correct library zero — a difference
    between the arms that is about emitting filenames rather than about writing
    code.

    Narrow on purpose. It applies only when *every* file sits under the
    workspace directory's own name, which is the container's root repeated and
    never a directory a project contains. A reply that wraps itself in anything
    else — the project's own name, say — is left exactly as it was and recorded
    by `_common_root`, because that is the model's choice rather than the
    benchmark's phrasing.
    """
    root = WORKSPACE.name
    if not files:
        return files, None

    parts = [PurePosixPath(path.replace("\\", "/")).parts for path, _ in files]
    if not all(len(part) > 1 and part[0] == root for part in parts):
        return files, None

    return [
        (str(PurePosixPath(*part[1:])), content)
        for part, (_, content) in zip(parts, files)
    ], root


def _write_files(files: list[tuple[str, str]]) -> list[str]:
    """Put the named files in the workspace, and refuse the ones that are not.

    The workspace is what the benchmark grades, so a path outside it is not a
    file this run may write. The reply is untrusted text from a model, and one
    that answers with an absolute path is answering a different question.
    """
    written: list[str] = []
    for path, content in files:
        try:
            target = _target(path)
        except ValueError as error:
            print(f"Refused {path!r}: {error}", file=sys.stderr)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip("\n") + "\n", encoding="utf-8")
        written.append(path)
    return written


def _target(path: str) -> Path:
    candidate = PurePosixPath(path.replace("\\", "/"))
    if candidate.is_absolute():
        raise ValueError("an absolute path is outside the workspace")
    parts = tuple(part for part in candidate.parts if part != ".")
    if not parts:
        raise ValueError("the path is empty")
    if ".." in parts:
        raise ValueError("the path climbs out of the workspace")
    if parts[0] in FORBIDDEN_ROOTS:
        raise ValueError(f"{parts[0]} is not part of the generated project")
    resolved = (WORKSPACE / PurePosixPath(*parts)).resolve()
    if not resolved.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("the path resolves outside the workspace")
    return resolved


def _request(
        client: Any,
        prompt: str,
        hyperparameters: dict[str, Any],
        totals: UsageTotals,
        model: str,
        routing: dict[str, Any] | None = None,
        observed: ServedProviders | None = None,
        failures: FailureTally | None = None,
        sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str | None]:
    """Send the one request, retrying only what a retry can fix.

    The attempts re-send the same prompt after a provider failure; they are not
    a second look at the answer, and a run that used all of them still made one
    request that returned. How many were needed is recorded either way.

    Every attempt is classified, whether it returned or raised, so that a run
    that ended at the output ceiling and one the provider refused for a prompt
    it could not fit are two findings in the record rather than one bad reward.
    """
    failures = failures if failures is not None else FailureTally()
    attempts = hyperparameters["request_attempts"]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": hyperparameters["max_tokens"],
        "temperature": hyperparameters["temperature"],
        "top_p": hyperparameters["top_p"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if hyperparameters.get("reasoning_effort"):
        body["reasoning_effort"] = hyperparameters["reasoning_effort"]
    # The experiment's routing travels with every request, merged into
    # whatever the configuration asked the provider for.
    extra_body = with_routing(hyperparameters.get("request_extra"), routing)
    if extra_body:
        body["extra_body"] = extra_body

    # A reply of tens of thousands of tokens takes minutes to arrive, and
    # nothing else is written until it does. Without this line the log is empty
    # for the whole of it, which is indistinguishable from a run that is stuck.
    print(
        f"Sending one request to {model}: {len(prompt)} characters in, at most "
        f"{hyperparameters['max_tokens']} tokens out. Waiting up to "
        f"{hyperparameters['request_timeout_seconds']}s for the reply."
    )

    failure: Exception | None = None
    for attempt in range(attempts):
        if failure is not None:
            print(
                f"Attempt {attempt} failed with "
                f"{type(failure).__name__}: {failure}",
                file=sys.stderr,
            )
        if failure is not None and _is_rate_limit(failure):
            sleep(RATE_LIMIT_WAITS[min(attempt - 1, len(RATE_LIMIT_WAITS) - 1)])
        try:
            response = client.chat.completions.create(**body)
            record_response_usage(response, totals)
            if observed is not None:
                observed.record(response)
            choice = response.choices[0]
            failures.record_finish_reason(getattr(choice, "finish_reason", None))
            content = choice.message.content
            if not content:
                # A reply that is empty — filtered, or reasoning with nothing
                # after it — is a failed attempt rather than a project of no
                # files.
                raise ValueError("the model returned an empty completion")
            return content, getattr(choice, "finish_reason", None)
        except Exception as error:  # noqa: BLE001 - reported below.
            failures.record_error(error)
            failure = error

    raise RuntimeError(
        f"Model {model!r} refused all {attempts} attempts at the single "
        f"request this baseline makes. The last was "
        f"{type(failure).__name__}: {failure}"
    ) from failure


def _is_rate_limit(error: Exception) -> bool:
    """Whether a failed request was throttled rather than rejected."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429
    return "rate limit" in str(error).lower() or "429" in str(error)


def _client(timeout_seconds: int) -> Any:
    """The OpenAI-compatible client, pointed at the configured endpoint.

    The credential is read indirectly, through the name the experiment
    configuration chose for it, so a configuration decides what the variable is
    called rather than the runner.

    `max_retries=0` leaves the retrying to `_request`, which is the half that
    is configured and recorded. The client retries twice on its own by default,
    and multiplied by the attempts above that is nine requests and up to an
    hour and a half against a task allowed one hour — a trial that dies at its
    timeout having recorded nothing.
    """
    from openai import OpenAI

    api_key_env = os.environ.get("API_KEY_ENV", "OPENAI_API_KEY")
    return OpenAI(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get(api_key_env),
        timeout=float(timeout_seconds),
        max_retries=0,
    )


def _read_instruction() -> str:
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(encoding="utf-8")


def _read_hyperparameters() -> dict[str, Any]:
    return json.loads(
        Path(os.environ["SINGLE_SHOT_HYPERPARAMETERS_PATH"]).read_text(
            encoding="utf-8"
        )
    )


def _record_resolved_setup(
        hyperparameters: dict[str, Any],
        finish_reason: str | None,
        written: list[str],
        warnings: list[str],
        requests: int,
        format_instruction: str = FORMAT_INSTRUCTION,
        stripped_root: str | None = None,
        routing: dict[str, Any] | None = None,
        providers: list[str] | None = None,
        failures: FailureTally | None = None,
        context: dict[str, Any] | None = None,
        change: dict[str, Any] | None = None,
        reply: dict[str, Any] | None = None,
) -> None:
    """Record what actually ran, next to the run's other logs.

    Beside the setup the other solutions record, this states the two things a
    reward figure cannot say about a single reply: whether it was cut off at
    the token ceiling, and whether the project arrived where the benchmark
    looks for it. Both are read from what happened rather than assumed.

    On a repair benchmark it states a third: what the request was actually shown
    of the repository. A prompt whose budget cut the one file the task named is
    a different finding from a model that read it and got it wrong, and only the
    record can tell them apart.
    """
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "solution": "Single-Shot",
                # This baseline has no upstream revision to pin: its identity
                # is the prompt it sent and the code that sent it.
                "runner_sha256": _digest(Path(__file__).read_bytes()),
                "prompt_sha256": _digest(format_instruction.encode("utf-8")),
                "model": os.environ.get("MODEL"),
                "base_url": os.environ.get("BASE_URL"),
                # What the run asked for, and which server actually answered.
                # A model name does not name a server, and an aggregator picks
                # by price unless told otherwise, so both are recorded.
                "routing": routing,
                "providers_served": providers or [],
                "hyperparameters": hyperparameters,
                # The sampling this run actually used, read back out of the
                # values the request was built from rather than off the file
                # that supplied them, which may have moved on since.
                "generation": _generation(hyperparameters),
                # Why the run ended where it did, in categories that stay apart
                # from each other, from a timeout, and from a failing test.
                "failures": (failures or FailureTally()).to_dict(),
                "requests": requests,
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "files_written": len(written),
                # Set when every path arrived under the workspace directory's
                # own name, copied from the tree the specification draws, and
                # was written one level up. See `_strip_workspace_root`.
                "stripped_root": stripped_root,
                # A reply that puts the whole project inside one directory
                # scores zero for a reason that has nothing to do with the code
                # in it. Recording it makes that visible in the run rather than
                # leaving it to be guessed at from the reward.
                "common_top_level_directory": _common_root(written),
                "parse_warnings": warnings,
                # What the one request was shown of the repository, and what
                # the budget cut. Absent on a benchmark that supplies no
                # repository, where the specification is the whole of the input.
                # What came back, in the one measure that separates a reply
                # that ran out of room from a reply that ran out of things to
                # say. See `reply_shape`.
                "reply": reply,
                "repository_context": context,
                # What the run left behind in that repository, which on a repair
                # benchmark is the answer itself rather than the workspace.
                "change": change,
                "reasoning_requested": bool(
                    hyperparameters.get("reasoning_effort")
                    or hyperparameters.get("request_extra")
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _generation(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """The sampling values this run sent, and where they came from.

    Written from the resolved hyperparameters the request was built out of, so
    the record is of what was used rather than of what was configured. The two
    agree unless something between the configuration and the request changed
    them, which is the case worth being able to see.
    """
    return {
        "temperature": hyperparameters["temperature"],
        "top_p": hyperparameters["top_p"],
        "max_tokens": hyperparameters["max_tokens"],
        "source": "configs/generation.yaml",
        "applied_via": "request body, one request",
    }


def _common_root(written: list[str]) -> str | None:
    """The one directory every generated file sits under, if there is one."""
    if not written:
        return None
    roots = {PurePosixPath(path).parts[0] for path in written}
    if len(roots) != 1:
        return None
    if not all(len(PurePosixPath(path).parts) > 1 for path in written):
        return None
    return roots.pop()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
