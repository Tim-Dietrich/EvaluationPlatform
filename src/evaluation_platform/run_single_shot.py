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
"""

import hashlib
import json
import os
import re
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

_FILE_HEADER = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?FILE:[ \t]*(?P<path>[^\n]+?)[ \t]*$",
    re.MULTILINE,
)
_FENCE_OPEN = re.compile(r"^(?P<fence>`{3,}|~{3,})[^\n]*$", re.MULTILINE)


def main() -> None:
    hyperparameters = _read_hyperparameters()
    instruction = _read_instruction()
    prompt = build_prompt(instruction)
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

    files, warnings = parse_files(answer)
    files, stripped_root = _strip_workspace_root(files)
    written = _write_files(files)
    _record_resolved_setup(
        hyperparameters=hyperparameters,
        finish_reason=finish_reason,
        written=written,
        warnings=warnings,
        requests=totals.responses,
        stripped_root=stripped_root,
        routing=routing,
        providers=observed.names(),
        failures=failures,
    )

    print(f"Wrote {len(written)} file(s) to {WORKSPACE}.")
    for warning in warnings:
        print(f"Warning: {warning}")
    if finish_reason == "length":
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


def build_prompt(instruction: str) -> str:
    """The one message the baseline sends: the task, then how to answer it."""
    return f"{instruction.strip()}\n\n---\n\n{FORMAT_INSTRUCTION.strip()}\n"


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
        stripped_root: str | None = None,
        routing: dict[str, Any] | None = None,
        providers: list[str] | None = None,
        failures: FailureTally | None = None,
) -> None:
    """Record what actually ran, next to the run's other logs.

    Beside the setup the other solutions record, this states the two things a
    reward figure cannot say about a single reply: whether it was cut off at
    the token ceiling, and whether the project arrived where the benchmark
    looks for it. Both are read from what happened rather than assumed.
    """
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "solution": "Single-Shot",
                # This baseline has no upstream revision to pin: its identity
                # is the prompt it sent and the code that sent it.
                "runner_sha256": _digest(Path(__file__).read_bytes()),
                "prompt_sha256": _digest(FORMAT_INSTRUCTION.encode("utf-8")),
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
