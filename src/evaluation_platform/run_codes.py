"""Run CodeS inside a Harbor task container.

CodeS decomposes "write this repository" into three layers of sketch:
RepoSketcher proposes the file tree, FileSketcher writes each Python file as
signatures with empty bodies, and SketchFiller implements one function per
request from that file's sketch and the sketches it imports. A separate stage
then assembles the answers into a repository on disk.

The tool ships two drivers for that pipeline. `from_scratch_inference.py` runs
it against the fine-tuned CodeS model through a local `transformers` pipeline,
which needs a GPU and weights and reaches no API at all;
`baselines/from_scratch_gpt35_eval.py` runs the same three phases against an
OpenAI-compatible endpoint. This platform supplies one credential and one base
URL, so the second is the driver integrated here, and what is evaluated is
CodeS's multi-layer sketch driven by the experiment's model rather than the
fine-tuned model of the paper.

That driver is a script rather than a library: its prompts, its arguments and
its loop are all statements at module scope, so importing it runs it. This
runner therefore repeats its loop and takes everything else from the tool —
the templates are read out of the driver's own source, the phase inputs come
from CodeS's `utils`, and the repository is assembled by CodeS's
`transfer_output_to_repo`. What is added around it is described where it
happens: the generated repository goes to `/workspace` and the phase records
to `/logs/agent/`; the requests carry the sampling, reasoning and usage
accounting the driver has no notion of; the run is bounded and assembles
whatever it has even when it stops early; one file's failure is contained to
that file; and a phase's requests may be in flight together.
"""

import ast
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

# The runner is uploaded next to its own dependencies, so its directory carries
# the shared usage accounting under a flat name. On the host, where the tests
# import this module from the package, that same directory is the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_usage import UsageTotals, record_response_usage  # noqa: E402

CODES_ROOT = Path("/installed-agent/codes")
EVALUATION_SCRIPTS = CODES_ROOT / "validation" / "evaluation_scripts"
# The tool's own OpenAI-compatible driver. Its prompt templates are read from
# this file rather than copied into this repository, so the prompts a run used
# are the pinned revision's and cannot drift away from it unnoticed.
DRIVER_PATH = EVALUATION_SCRIPTS / "baselines" / "from_scratch_gpt35_eval.py"
WORKSPACE = Path("/workspace")
ARTIFACTS_DIR = Path("/logs/agent/codes")
USAGE_PATH = Path("/logs/agent/model-usage.json")
RESOLVED_SETUP_PATH = Path("/logs/agent/resolved-setup.json")

# How long the wrapper waits before retrying a rate-limited request, in
# seconds, once CodeS's own attempts have been exhausted.
RATE_LIMIT_WAITS = (15, 30)

# The three phases, under the names the tool gives them in its templates and in
# the files it writes. Keeping the names means its own assembly stage reads the
# records this runner writes without being told anything.
REPO_SKETCH = "repo_sketch.json"
FILE_SKETCH = "file_sketch.json"
FUNCTION_BODY = "function_body.json"
PHASES = (REPO_SKETCH, FILE_SKETCH, FUNCTION_BODY)


class BudgetExhausted(RuntimeError):
    """A run reached the wall-clock or token budget its configuration set."""


def main() -> None:
    hyperparameters = _read_hyperparameters()
    readme = _read_instruction()
    _require_deprecated_ast_aliases()
    tool = _import_tool()

    totals = UsageTotals()
    budget = _Budget(hyperparameters, totals)
    request = _requester(
        _client(),
        hyperparameters,
        totals,
        budget,
        model=os.environ["MODEL"],
    )
    records: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}

    try:
        _run_phases(
            tool,
            records,
            readme=readme,
            request=request,
            workers=hyperparameters["concurrent_requests"],
        )
    finally:
        # Everything below reports on the run rather than continuing it, so it
        # happens whether the pipeline finished, exhausted its budget, or
        # failed outright. A partial pipeline still assembles: file sketches
        # without their function bodies are a repository of correct interfaces
        # and empty bodies, which the verifier can grade, where an unwritten
        # workspace is scored as nothing at all.
        _write_usage(totals)
        _write_records(records)
        outcome = _guarded(lambda: _write_repository(tool, records))
        _guarded(lambda: _record_resolved_setup(hyperparameters, records, outcome))
    print(f"Done. Repo at: {WORKSPACE}")


def _run_phases(
        tool: Any,
        records: dict[str, list[dict[str, Any]]],
        readme: str,
        request: Callable[[dict[str, Any]], str],
        workers: int,
) -> None:
    """The three phases, in the order and the shape the tool's driver runs them.

    Each phase turns the previous phase's answers into the next phase's
    prompts through CodeS's own `utils`, so the decomposition — which files the
    repository sketch names, which of them get a file sketch, which functions
    get a body, and what context each prompt carries — is the tool's and not
    this runner's. What this runner decides is only that the requests within
    one phase may be in flight together.
    """
    repo_sketch = {
        "readme": readme,
        "instruction": tool.templates[REPO_SKETCH].format_map({"readme": readme}),
    }
    records[REPO_SKETCH].append(repo_sketch)
    _answer(tool, [repo_sketch], request, workers=1)

    records[FILE_SKETCH].extend(
        tool.utils.generate_file_sketch_input_openai(
            repo_sketch, readme, tool.templates[FILE_SKETCH]
        )
    )
    _answer(tool, records[FILE_SKETCH], request, workers)

    sketches = {each["file_path"]: each for each in records[FILE_SKETCH]}
    for each in sketches.values():
        if not each["file_path"].endswith(".py"):
            continue
        # A file sketch the model returned malformed can defeat the tool's
        # parsing of it, and one file is not the run. The failure is contained
        # to the file it came from, counted, and reported in the resolved
        # setup; the alternative loses every other file to it.
        prompts = _guarded(
            lambda each=each: tool.utils.generate_function_body_input_openai(
                each, readme, sketches, "", tool.templates[FUNCTION_BODY]
            )
        )
        if prompts is None:
            each["function_prompts_failed"] = True
            continue
        records[FUNCTION_BODY].extend(prompts)
    _answer(tool, records[FUNCTION_BODY], request, workers)


def _answer(
        tool: Any,
        prompts: list[dict[str, Any]],
        request: Callable[[dict[str, Any]], str],
        workers: int,
) -> None:
    """Fill in `generated` and `parsed` for one phase's prompts.

    The tool's driver makes these requests one at a time. Nothing in a phase
    depends on another request in the same phase — the file sketches all read
    the one repository sketch, and the function bodies all read the completed
    file sketches — so running them together changes how long the phase takes
    and not what any prompt says. It is off unless a configuration asks for it,
    and the records stay in the order the tool would have produced them.
    """

    def answer(prompt: dict[str, Any]) -> None:
        prompt["generated"] = request(prompt)
        prompt["parsed"] = tool.utils.parse_reponse(prompt["generated"])

    if workers <= 1 or len(prompts) <= 1:
        for prompt in prompts:
            answer(prompt)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # `list` rather than a bare `map`, so an exception in any request is
        # raised here rather than discarded with the unread iterator.
        list(executor.map(answer, prompts))


def _requester(
        client: Any,
        hyperparameters: dict[str, Any],
        totals: UsageTotals,
        budget: "_Budget",
        model: str,
        sleep: Callable[[float], None] = time.sleep,
) -> Callable[[dict[str, Any]], str]:
    """One request, made the way the tool makes it, with four things added.

    The tool's driver gives each request `request_attempts` tries, the first at
    `temperature` and the rest at `retry_temperature`, and calls `exit(1)` when
    they are all refused. Both figures are hardcoded there, which puts the
    sampling of a run outside its recorded setup; they are hyperparameters here
    and the request is otherwise the one the tool builds.

    Added around it: the token limit and `top_p` the tool never sends, the
    reasoning fields it has no notion of, a wait between attempts when the
    provider is throttling rather than refusing, and the usage accounting every
    solution in this platform shares. The wait is inside the tool's own attempt
    count rather than beside it, so a throttled request costs time and not a
    multiple of the requests a configuration asked for.
    """
    attempts = hyperparameters["request_attempts"]
    lock = threading.Lock()

    def send(instruction: str, temperature: float) -> Any:
        body: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": instruction}],
        }
        if hyperparameters.get("max_tokens") is not None:
            body["max_tokens"] = hyperparameters["max_tokens"]
        if hyperparameters.get("top_p") is not None:
            body["top_p"] = hyperparameters["top_p"]
        if hyperparameters.get("reasoning_effort"):
            body["reasoning_effort"] = hyperparameters["reasoning_effort"]
        if hyperparameters.get("request_extra"):
            body["extra_body"] = dict(hyperparameters["request_extra"])
        response = client.chat.completions.create(**body)
        with lock:
            record_response_usage(response, totals)
        content = response.choices[0].message.content
        if not content:
            # A reply that is empty — filtered, truncated before any text, or
            # reasoning with nothing after it — is a failed attempt rather than
            # a file with no contents.
            raise ValueError("the model returned an empty completion")
        return content

    def request(prompt: dict[str, Any]) -> str:
        budget.check()
        failure: Exception | None = None
        for attempt in range(attempts):
            if failure is not None and _is_rate_limit(failure):
                sleep(RATE_LIMIT_WAITS[min(attempt - 1, len(RATE_LIMIT_WAITS) - 1)])
            try:
                return send(
                    prompt["instruction"],
                    hyperparameters["temperature"]
                    if attempt == 0
                    else hyperparameters["retry_temperature"],
                )
            except Exception as error:  # noqa: BLE001 - reported below.
                failure = error
        raise RuntimeError(
            f"Model {model!r} refused all {attempts} attempts at the "
            f"{prompt.get('file_path', 'repository sketch')} request. The last "
            f"was {type(failure).__name__}: {failure}"
        ) from failure

    return request


def _is_rate_limit(error: Exception) -> bool:
    """Whether a failed request was throttled rather than rejected.

    Only a throttled request is worth waiting for: a rejected credential or an
    unknown model will be refused just as quickly the second time. The
    provider's status code decides it where the client exposes one, and the
    message otherwise.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status == 429
    return "rate limit" in str(error).lower() or "429" in str(error)


class _Budget:
    """What bounds a pipeline whose length the model chooses.

    CodeS makes one request for the repository sketch, one for each Python
    file the sketch names, and one for every function in every one of those
    file sketches. None of those counts is set by the configuration: a
    repository sketch naming thirty modules costs an order of magnitude more
    than one naming three, and the tool has no bound of its own. Left unset the
    only limit is Harbor's agent timeout, which is reached with nothing
    recorded about how far the run got.
    """

    def __init__(self, hyperparameters: dict[str, Any], totals: UsageTotals) -> None:
        self.seconds = hyperparameters.get("max_wall_clock_seconds")
        self.tokens = hyperparameters.get("max_token_budget")
        self.totals = totals
        self.started = time.monotonic()

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if self.seconds is not None and elapsed >= self.seconds:
            raise BudgetExhausted(
                f"Stopped after {elapsed:.0f}s, at the configured "
                f"max_wall_clock_seconds of {self.seconds}."
            )
        spent = self.totals.input_tokens + self.totals.output_tokens
        if self.tokens is not None and spent >= self.tokens:
            raise BudgetExhausted(
                f"Stopped after {spent} tokens, at the configured "
                f"max_token_budget of {self.tokens}."
            )


def _write_repository(
        tool: Any,
        records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the phases into the repository, in the directory Harbor grades.

    CodeS's own assembly stage parses each file sketch, replaces each function
    body with the one SketchFiller wrote for it, and formats the result. That
    logic is the tool's and is used unchanged. What is changed is where the
    result lands and what a bad file costs.

    The tool writes into a directory named after the repository, one level
    below the output directory it is given, which suits a batch run keeping a
    hundred generated repositories side by side. Harbor grades `/workspace`
    itself: the task's verifier copies what it finds there on top of the
    benchmark's reference tests, so a project generated one level down would be
    scored as an empty repository rather than reported as misplaced.

    The tool also assembles every file in one unguarded loop, so a single file
    whose sketch does not parse ends the assembly and leaves the repository
    partly written or empty. Each file is assembled on its own here, and one
    that cannot be filled in is written as the sketch FileSketcher produced —
    correct signatures with empty bodies — rather than not written at all.
    """
    sketches = tool.transfer.get_files(_phase_path(FILE_SKETCH))
    bodies = tool.transfer.get_functions(_phase_path(FUNCTION_BODY))
    written, unfilled, rejected = 0, 0, []
    for relative_path, sketch in sketches.items():
        target = _target_path(relative_path)
        if target is None:
            rejected.append(relative_path)
            continue
        content = sketch
        if relative_path in bodies:
            filled = _guarded(
                lambda sketch=sketch, name=relative_path: (
                    tool.transfer.fill_in_function(sketch, bodies[name])
                )
            )
            if filled is None:
                unfilled += 1
            else:
                content = filled
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written += 1
    return {
        "files_written": written,
        "files_left_as_sketches": unfilled,
        "file_paths_rejected": rejected,
    }


def _target_path(relative_path: str) -> Path | None:
    """Where one generated file goes, or nothing if it does not go in the tree.

    The paths come from a file tree the model wrote, so they are not trusted to
    be relative. A path that resolves outside the workspace is dropped and
    named in the resolved setup rather than written where it points.
    """
    target = (WORKSPACE / relative_path).resolve()
    if target == WORKSPACE.resolve() or WORKSPACE.resolve() not in target.parents:
        return None
    return target


def _import_tool() -> Any:
    """CodeS's own modules, and the prompts out of its own driver.

    `utils` and `transfer_output_to_repo` sit beside the driver and are
    imported from there; `extract_sketch` and `prompt_construction_utils` sit
    at the root of the checkout and are reached by `utils` itself, which puts
    the root on the path when it loads.

    `transfer_output_to_repo` parses command-line arguments at import, so it is
    given an argument list before being imported. Every option it declares has
    a default and this runner uses none of them: it calls the module's
    functions and decides the destinations itself.
    """
    sys.path.insert(0, str(EVALUATION_SCRIPTS))
    sys.path.insert(0, str(CODES_ROOT))
    argv, sys.argv = sys.argv, [str(DRIVER_PATH)]
    try:
        tool = _Tool(
            utils=importlib.import_module("utils"),
            transfer=importlib.import_module("transfer_output_to_repo"),
            templates=_read_templates(DRIVER_PATH.read_text(encoding="utf-8")),
        )
    finally:
        sys.argv = argv
    return tool


class _Tool:
    """The parts of CodeS this runner drives."""

    def __init__(self, utils: Any, transfer: Any, templates: dict[str, str]) -> None:
        self.utils = utils
        self.transfer = transfer
        self.templates = templates


def _read_templates(source: str) -> dict[str, str]:
    """The driver's prompt templates, read from its source without running it.

    The driver is a script: importing it parses arguments, opens files and
    starts making requests. Its prompts are nevertheless a plain literal
    assignment at module scope, so they can be read out of its syntax tree.
    That is deliberately not the same as copying them into this repository,
    where they would be a second copy free to disagree with the revision the
    configuration pins.
    """
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
                isinstance(target, ast.Name) and target.id == "TEMPLATE_DICT"
                for target in node.targets
        ):
            continue
        templates = ast.literal_eval(node.value)
        missing = [phase for phase in PHASES if phase not in templates]
        if missing:
            raise RuntimeError(
                f"{DRIVER_PATH} defines prompts for "
                f"{', '.join(sorted(templates))} but this pipeline needs "
                f"{', '.join(missing)} as well."
            )
        return templates
    raise RuntimeError(
        f"{DRIVER_PATH} no longer assigns TEMPLATE_DICT at module level, so "
        "the prompts of the pinned revision cannot be read from it."
    )


def _client() -> Any:
    """The OpenAI-compatible client, pointed at the configured endpoint.

    The credential is read indirectly, through the name the experiment
    configuration chose for it, so a configuration decides what the variable is
    called rather than the tool.
    """
    from openai import OpenAI

    api_key_env = os.environ.get("API_KEY_ENV", "OPENAI_API_KEY")
    return OpenAI(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get(api_key_env),
    )


def _require_deprecated_ast_aliases() -> None:
    """Fail now, with the reason, rather than three phases in.

    CodeS builds and inspects syntax trees with `ast.Str`, which Python
    deprecated in 3.8 and removed in 3.14. Where it is gone, the tool raises an
    `AttributeError` from inside the prompt construction of its third phase —
    after the whole repository sketch and every file sketch have been paid for.
    """
    if not hasattr(ast, "Str"):
        raise RuntimeError(
            f"CodeS needs ast.Str, which Python {sys.version_info.major}."
            f"{sys.version_info.minor} has removed. Run the task image on "
            "Python 3.13 or older, or pin a revision of CodeS that has been "
            "ported."
        )


def _guarded(action: Callable[[], Any]) -> Any:
    """Run `action`, reporting a failure to the log and returning nothing.

    Used where one file, or one record of what happened, is worth less than
    the rest of the run. Every use is somewhere the alternative is losing work
    that is already paid for.
    """
    try:
        return action()
    except Exception as error:  # noqa: BLE001 - contained deliberately.
        print(f"Contained failure: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def _phase_path(phase: str) -> str:
    """Where one phase's records are written.

    The tool's own suffix is kept, because its assembly stage reads
    `<phase>.json.jsonl` and is given these files to read.
    """
    return str(ARTIFACTS_DIR / f"{phase}.jsonl")


def _write_records(records: dict[str, list[dict[str, Any]]]) -> None:
    """The prompts and answers of every phase, beside the run's other logs.

    These are the tool's own record of what it did: the repository sketch it
    proposed, the sketch of every file, and every function it filled in. The
    tool writes them beneath the directory it generates into, which under the
    arrangement above would be inside the repository being graded, where the
    verifier would copy them onto the benchmark's tests.

    Only answered requests are written, because the tool appends each answer as
    it arrives and its assembly stage reads these same files back. A run that
    stopped part-way through a phase leaves prompts with no answer, and one of
    those reaching the assembly stage fails it — losing every file of a
    repository that was otherwise ready to be graded, over the requests that
    were never made.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        Path(_phase_path(phase)).write_text(
            "".join(
                json.dumps(record) + "\n"
                for record in records[phase]
                if "parsed" in record
            ),
            encoding="utf-8",
        )


def _write_usage(totals: UsageTotals) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(totals.to_dict(), indent=2), encoding="utf-8")


def _read_instruction() -> str:
    """The task's specification, which is what CodeS reads as the README.

    The tool takes a repository's `README.md` as the whole of its input and
    every phase's prompt quotes it. Harbor's instruction is that document.
    """
    return Path(os.environ["HARBOR_TASK_INSTRUCTION_PATH"]).read_text(encoding="utf-8")


def _read_hyperparameters() -> dict[str, Any]:
    return json.loads(
        Path(os.environ["CODES_HYPERPARAMETERS_PATH"]).read_text(encoding="utf-8")
    )


def _record_resolved_setup(
        hyperparameters: dict[str, Any],
        records: dict[str, list[dict[str, Any]]],
        outcome: dict[str, Any] | None,
) -> None:
    """Record what actually ran, next to the run's other logs.

    The experiment configuration states the intended setup; this states the
    observed one. For CodeS the observation that the configuration cannot
    predict is the shape of the pipeline itself: how many files the repository
    sketch named, how many functions the file sketches declared, and how many
    of each survived to the repository. A run that stops early is visible here
    as phases that were reached and phases that were not.
    """
    RESOLVED_SETUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED_SETUP_PATH.write_text(
        json.dumps(
            {
                "codes_commit": _installed_commit(),
                "model": os.environ.get("MODEL"),
                "base_url": os.environ.get("BASE_URL"),
                "hyperparameters": hyperparameters,
                "files_sketched": len(records[FILE_SKETCH]),
                "functions_requested": len(records[FUNCTION_BODY]),
                "requests_answered": sum(
                    1
                    for phase in PHASES
                    for record in records[phase]
                    if "generated" in record
                ),
                "file_sketches_that_yielded_no_prompts": sum(
                    1
                    for record in records[FILE_SKETCH]
                    if record.get("function_prompts_failed")
                ),
                **(outcome or {"files_written": None}),
                "concurrent_requests": hyperparameters["concurrent_requests"],
                "reasoning_requested": bool(
                    hyperparameters.get("reasoning_effort")
                    or hyperparameters.get("request_extra")
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _installed_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CODES_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
