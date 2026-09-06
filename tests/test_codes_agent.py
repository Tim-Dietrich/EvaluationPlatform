import asyncio
import json
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.codes_agent import (
    CODES_COMMIT,
    CODES_ROOT,
    CODES_SPARSE_PATH,
    HYPERPARAMETERS_PATH,
    RUNTIME_PACKAGES,
    TASK_INSTRUCTION_PATH,
    TASK_WORKSPACE,
    CodeSAgent,
)
from evaluation_platform.experiment_config import (
    CODE_S,
    ConfigurationError,
    resolve_generation_kwargs,
)
from evaluation_platform.failure_categories import FailureTally
from evaluation_platform.model_routing import ServedProviders
from evaluation_platform.model_usage import UsageTotals
from evaluation_platform import run_codes
from evaluation_platform.run_codes import (
    FILE_SKETCH,
    FUNCTION_BODY,
    PHASES,
    REPO_SKETCH,
    BudgetExhausted,
    _Budget,
    _client,
    _is_rate_limit,
    _memoise_file_sketching,
    _read_templates,
    _requester,
    _run_phases,
    _write_records,
    _write_repository,
)


class RecordingEnvironment:
    default_user = "root"

    def __init__(self):
        self.commands = []
        self.uploads = []
        self.upload_contents = []

    async def exec(self, **kwargs):
        self.commands.append(kwargs)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target):
        self.uploads.append((Path(source), target))
        self.upload_contents.append(Path(source).read_text(encoding="utf-8"))

    async def download_file(self, source, target):
        Path(target).write_text(
            '{"input_tokens": 0, "cached_input_tokens": 0, '
            '"output_tokens": 0, "cost_usd": null}',
            encoding="utf-8",
        )


def install(agent: CodeSAgent) -> RecordingEnvironment:
    environment = RecordingEnvironment()
    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))
    return environment


def uploaded(environment: RecordingEnvironment) -> dict[str, str]:
    return dict(
        zip(
            [target for _, target in environment.uploads],
            environment.upload_contents,
        )
    )


def test_install_pins_the_revision_and_uploads_the_runner(tmp_path):
    environment = install(CodeSAgent(logs_dir=tmp_path))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert CODES_COMMIT in commands
    assert f"https://github.com/NL2Code/CodeS.git {CODES_ROOT}" in commands
    for package in RUNTIME_PACKAGES:
        assert package in commands
    targets = [target for _, target in environment.uploads]
    # The runner imports the shared usage accounting and the shared provider
    # routing by their flat names, so both have to arrive beside it.
    assert targets == [
        "/installed-agent/run_codes.py",
        "/installed-agent/model_usage.py",
        "/installed-agent/model_routing.py",
        "/installed-agent/failure_categories.py",
    ]


def test_install_fetches_only_the_part_of_the_checkout_a_run_reads(tmp_path):
    """The repository carries two hundred sample repositories no run opens.

    Every trial clones it, so what is fetched is a per-trial cost paid
    hundreds of times over a benchmark run. The revision is still pinned: a
    narrower working tree, not a different commit.
    """
    environment = install(CodeSAgent(logs_dir=tmp_path))

    clone = next(
        command for command in environment.commands if "git clone" in command["command"]
    )
    assert "--filter=blob:none" in clone["command"]
    assert f"sparse-checkout set {CODES_SPARSE_PATH}" in clone["command"]
    assert f"checkout --quiet {CODES_COMMIT}" in clone["command"]


def test_install_leaves_out_what_only_the_fine_tuned_model_needs(tmp_path):
    """The tool's `requirements.txt` and its other driver are not this run's.

    `torch` and `transformers` serve the inference driver for the fine-tuned
    CodeS model, which needs a GPU and reaches no API; `tree-sitter`, `isort`
    and `autopep8` serve the corpus preparation that builds its training data.
    Installing them would put a scientific stack in every trial.
    """
    environment = install(CodeSAgent(logs_dir=tmp_path))

    installed = next(
        command["command"]
        for command in environment.commands
        if "pip install" in command["command"]
    )
    for package in ("torch", "transformers", "tree-sitter", "isort", "autopep8"):
        assert package not in installed


def test_run_uploads_instruction_instead_of_putting_it_on_docker_command_line(tmp_path):
    environment = RecordingEnvironment()
    agent = CodeSAgent(
        logs_dir=tmp_path,
        extra_env={"API_KEY": "secret", "MODEL": "test/model"},
    )
    instruction = "Build the library\n" * 10_000

    asyncio.run(
        agent.run(
            instruction,
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    invocation = environment.commands[-1]
    assert "Build the library" not in invocation["command"]
    assert "secret" not in invocation["command"]
    assert invocation["env"] == {
        "HARBOR_TASK_INSTRUCTION_PATH": TASK_INSTRUCTION_PATH,
        "CODES_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
    }
    assert uploaded(environment)[TASK_INSTRUCTION_PATH] == instruction
    assert invocation["cwd"] == TASK_WORKSPACE


def test_run_uploads_the_complete_setup_including_filled_in_defaults(tmp_path):
    environment = RecordingEnvironment()
    agent = CodeSAgent(logs_dir=tmp_path, concurrent_requests=8)

    asyncio.run(
        agent.run(
            "Build the library",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    hyperparameters = json.loads(uploaded(environment)[HYPERPARAMETERS_PATH])
    assert hyperparameters["concurrent_requests"] == 8
    assert hyperparameters["request_attempts"] == 5
    assert hyperparameters["retry_temperature"] == 0.1
    assert hyperparameters["request_timeout_seconds"] == 600
    # The sampling this pipeline's every request will carry, from the one file
    # that sets it for every arm, beside the solution's own settings.
    shared = resolve_generation_kwargs({})
    assert {name: hyperparameters[name] for name in shared} == shared


@pytest.mark.parametrize(
    "hyperparameter",
    [
        {"concurrent_requests": 0},
        {"request_attempts": "many"},
        {"retry_temperature": "cold"},
        # Sampling is not this solution's to state; it is set for every arm at
        # once in `configs/generation.yaml`.
        {"temperature": 0.3},
        # CodeTeam's, which CodeS has no stage for.
        {"architects": 4},
        # Self-Collaboration's, likewise.
        {"test_command": "pytest -q"},
    ],
)
def test_agent_rejects_hyperparameters_the_runner_cannot_use(tmp_path, hyperparameter):
    with pytest.raises(ConfigurationError):
        CodeSAgent(logs_dir=tmp_path, **hyperparameter)


def test_usage_recorded_in_the_container_reaches_harbor(tmp_path):
    (tmp_path / "model-usage.json").write_text(
        json.dumps(
            {
                "input_tokens": 900,
                "cached_input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 20,
                "cost_usd": 0.25,
            }
        ),
        encoding="utf-8",
    )
    agent = CodeSAgent(logs_dir=tmp_path)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 900
    assert context.n_cache_tokens == 100
    assert context.n_output_tokens == 50
    assert context.cost_usd == 0.25


# The tool's own driver and phase helpers, reduced to what the runner uses.
# Standing in for them rather than importing the checkout keeps this under test
# on a machine that has only cloned this repository.


DRIVER_SOURCE = '''
import os
from openai import OpenAI

TEMPLATE_DICT = {
    "repo_sketch.json": """Design a repository.

## Repository README
```md
{readme}
```""",
    "file_sketch.json": """Sketch one file.

## Target File Path
{file_path}""",
    "function_body.json": """Fill in one function.

## Target Function
{function_signature}""",
}

parser = argparse.ArgumentParser()
for repo in os.listdir("."):
    print("this driver makes requests at import time")
'''


class FakeUtils:
    """CodeS's phase helpers, in the shape the runner calls them."""

    @staticmethod
    def parse_reponse(response):
        return response.strip()

    @staticmethod
    def generate_file_sketch_input_openai(each, readme, template):
        return [
            {
                "readme": readme,
                "repo_sketch": each["parsed"],
                "file_path": path,
                "instruction": template.format_map({"file_path": path}),
            }
            for path in each["parsed"].split()
            if path.endswith(".py")
        ]

    @staticmethod
    def generate_function_body_input_openai(each, readme, insts, repo_path, template):
        if "unparseable" in each["parsed"]:
            raise SyntaxError("the file sketch does not parse")
        return [
            {
                "current_file_path": each["file_path"],
                "function_header": f"def {name}():",
                "instruction": template.format_map({"function_signature": name}),
            }
            for name in each["parsed"].split()
            if not name.endswith(".py")
        ]


class FakeTransfer:
    """CodeS's assembly stage, in the shape the runner calls it."""

    @staticmethod
    def get_files(path):
        return {
            json.loads(line)["file_path"]: json.loads(line)["parsed"]
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        }

    @staticmethod
    def get_functions(path):
        grouped: dict[str, dict[str, list[str]]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            name = record["function_header"].split("def ")[-1].split("(")[0]
            grouped.setdefault(record["current_file_path"], {}).setdefault(
                name, []
            ).append(record["parsed"])
        return grouped

    @staticmethod
    def fill_in_function(sketch, function_map):
        if "unfillable" in sketch:
            raise SyntaxError("the sketch cannot be parsed for assembly")
        return sketch + "".join(
            f"\n# filled {name}" for name in sorted(function_map)
        )


def fake_tool(templates: dict[str, str] | None = None) -> Any:
    return SimpleNamespace(
        utils=FakeUtils,
        transfer=FakeTransfer,
        templates=templates or _read_templates(DRIVER_SOURCE),
    )


def test_the_prompts_are_the_pinned_revisions_rather_than_a_copy_of_them():
    """The templates are read from the driver's source, not duplicated here.

    A copy in this repository would be free to disagree with the revision a
    configuration pins, and nothing would say which of the two a run used.
    """
    templates = _read_templates(DRIVER_SOURCE)

    assert set(templates) == set(PHASES)
    assert "{readme}" in templates[REPO_SKETCH]
    assert "{file_path}" in templates[FILE_SKETCH]


def test_a_driver_that_no_longer_carries_the_prompts_is_reported():
    with pytest.raises(RuntimeError, match="TEMPLATE_DICT"):
        _read_templates("PROMPTS = {'repo_sketch.json': 'x'}\n")


def test_a_driver_missing_a_phase_this_pipeline_runs_is_reported():
    with pytest.raises(RuntimeError, match="file_sketch.json"):
        _read_templates("TEMPLATE_DICT = {'repo_sketch.json': 'x'}\n")


def answered(record: dict[str, Any]) -> str:
    """A canned answer keyed on which phase the prompt belongs to."""
    if "## Target Function" in record["instruction"]:
        return "body-of " + record["function_header"]
    if "## Target File Path" in record["instruction"]:
        return f"{record['file_path'].removesuffix('.py')}_one"
    return "alpha.py beta.py notes.md"


def unbounded() -> _Budget:
    """A budget that stops nothing, for the tests that are not about budgets."""
    return _Budget(CODE_S.resolve({}), UsageTotals())


def run_phases(
        tool: Any = None,
        workers: int = 1,
        budget: _Budget | None = None,
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    _run_phases(
        tool or fake_tool(),
        records,
        readme="# library",
        request=answered,
        workers=workers,
        budget=budget or unbounded(),
    )
    return records


def test_each_phase_asks_only_about_what_the_previous_one_produced():
    records = run_phases()

    # The repository sketch named three paths; only the Python ones are
    # sketched, and each sketch's declared functions are what get bodies.
    assert [record["file_path"] for record in records[FILE_SKETCH]] == [
        "alpha.py",
        "beta.py",
    ]
    assert [record["current_file_path"] for record in records[FUNCTION_BODY]] == [
        "alpha.py",
        "beta.py",
    ]
    assert records[REPO_SKETCH][0]["parsed"] == "alpha.py beta.py notes.md"


def test_running_a_phase_concurrently_produces_the_same_records():
    assert run_phases(workers=4) == run_phases(workers=1)


def test_one_file_the_tool_cannot_parse_does_not_cost_the_others():
    """CodeS's own prompt construction raises on a sketch it cannot parse.

    Left uncontained that ends the phase, so one bad response would discard
    every other file's function bodies as well as its own.
    """

    def request(record):
        if record.get("file_path") == "alpha.py":
            return "unparseable ((("
        return answered(record)

    records: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    _run_phases(
        fake_tool(),
        records,
        readme="#",
        request=request,
        workers=1,
        budget=unbounded(),
    )

    assert [record["current_file_path"] for record in records[FUNCTION_BODY]] == [
        "beta.py"
    ]
    assert records[FILE_SKETCH][0]["function_prompts_failed"] is True


def test_a_budget_stops_prompt_construction_and_not_only_requests():
    """The third phase spends without sending, so a budget read at each request
    is not read for the whole of the stretch a run is likeliest to overrun in.

    That stretch is CodeS's own construction of the function body prompts,
    whose length is set by how many files the model named and how many
    functions it declared in them. Unbounded it reaches the agent timeout,
    which records nothing; bounded it stops, and the file sketches already paid
    for still assemble.
    """
    totals = UsageTotals()

    def request(record):
        totals.output_tokens += 20
        return answered(record)

    records: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    with pytest.raises(BudgetExhausted, match="max_token_budget"):
        _run_phases(
            fake_tool(),
            records,
            readme="#",
            request=request,
            workers=1,
            budget=_Budget(CODE_S.resolve({"max_token_budget": 25}), totals),
        )

    assert [record["file_path"] for record in records[FILE_SKETCH]] == [
        "alpha.py",
        "beta.py",
    ]
    assert records[FUNCTION_BODY] == []


def sketching_module() -> Any:
    """CodeS's sketch machinery, cut down to the shape the memo has to survive.

    The tool reaches `replace_function_body` through its own module's globals
    rather than through a reference it captured at import, which is what lets
    the runner install a memo on the module without modifying the checkout.
    Reproducing that needs a real module namespace, so the double is defined
    inside one rather than assembled from functions defined here.
    """
    module = ModuleType("extract_sketch_double")
    exec(
        dedent(
            '''
            built = []


            def replace_function_body(
                    source_code, unimplemented_function_name="", index=0
            ):
                """Parse, render and format one file. Expensive in the tool."""
                built.append((source_code, unimplemented_function_name))
                return f"[{source_code}:{unimplemented_function_name or 'whole'}]"


            def relevant_final_prompt(current, imported, function_name):
                """One function's context: what it imports, then its own file."""
                sketches = [replace_function_body(each) for each in imported]
                sketches.append(replace_function_body(current, function_name))
                return "".join(sketches)
            '''
        ),
        module.__dict__,
    )
    return module


def test_an_imported_files_sketch_is_built_once_not_once_per_function():
    """What a design of any size spends its wall clock on before it asks.

    CodeS gives every function body prompt the sketches of the files its own
    file imports, and rebuilds each of them for every function, though the call
    passes the file's contents and nothing else and so returns the same string
    every time. Each rebuild parses and formats a whole file. Multiplied by the
    functions of every file it is the hours this phase can take.
    """
    module = sketching_module()
    _memoise_file_sketching(module)

    for name in ("read", "write", "close"):
        module.relevant_final_prompt("main.py", ["io.py", "util.py"], name)

    assert [file for file, function in module.built if not function] == [
        "io.py",
        "util.py",
    ]
    # The sketch of the file the function is in names that function, so it
    # differs for every prompt and is built for every prompt.
    assert [function for file, function in module.built if function] == [
        "read",
        "write",
        "close",
    ]


def test_the_memo_answers_with_what_the_tool_would_have_returned():
    plain, memoised = sketching_module(), sketching_module()
    _memoise_file_sketching(memoised)

    names = ("read", "write")
    assert [
        memoised.relevant_final_prompt("main.py", ["io.py"], name) for name in names
    ] == [plain.relevant_final_prompt("main.py", ["io.py"], name) for name in names]


def test_the_memo_is_installed_once_however_often_the_tool_is_imported():
    module = sketching_module()

    _memoise_file_sketching(module)
    installed = module.replace_function_body
    _memoise_file_sketching(module)

    assert module.replace_function_body is installed


def build_repository(tmp_path, monkeypatch, records) -> tuple[Path, dict[str, Any]]:
    monkeypatch.setattr(run_codes, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(run_codes, "ARTIFACTS_DIR", tmp_path / "logs" / "codes")
    _write_records(records)
    return tmp_path / "workspace", _write_repository(fake_tool(), records)


def test_the_generated_repository_is_the_workspace_harbor_grades(tmp_path, monkeypatch):
    """The verifier copies what it finds at the workspace onto the reference
    tests, so a project one directory down is scored as an empty repository."""
    workspace, outcome = build_repository(tmp_path, monkeypatch, run_phases())

    assert outcome["files_written"] == 2
    assert sorted(path.name for path in workspace.iterdir()) == [
        "alpha.py",
        "beta.py",
    ]
    assert "# filled alpha_one" in (workspace / "alpha.py").read_text(encoding="utf-8")


def test_the_tools_own_record_stays_out_of_the_repository_being_graded(
        tmp_path, monkeypatch
):
    workspace, _ = build_repository(tmp_path, monkeypatch, run_phases())

    artifacts = tmp_path / "logs" / "codes"
    assert sorted(path.name for path in artifacts.iterdir()) == [
        f"{phase}.jsonl" for phase in sorted(PHASES)
    ]
    assert workspace not in artifacts.parents


def test_a_file_that_cannot_be_assembled_is_written_as_its_sketch(
        tmp_path, monkeypatch
):
    """The tool assembles every file in one unguarded loop.

    One file whose sketch defeats the assembly would end it, leaving the
    repository partly written. Kept to the file it came from, the cost is that
    file's function bodies rather than every file after it.
    """
    records = run_phases()
    records[FILE_SKETCH][0]["parsed"] = "unfillable_one"

    workspace, outcome = build_repository(tmp_path, monkeypatch, records)

    assert outcome["files_left_as_sketches"] == 1
    assert outcome["files_written"] == 2
    assert (workspace / "alpha.py").read_text(encoding="utf-8") == "unfillable_one"


def test_a_generated_path_outside_the_workspace_is_dropped(tmp_path, monkeypatch):
    """The file tree is written by the model, so its paths are not trusted."""
    records = run_phases()
    records[FILE_SKETCH][0]["file_path"] = "../escaped.py"

    workspace, outcome = build_repository(tmp_path, monkeypatch, records)

    assert outcome["file_paths_rejected"] == ["../escaped.py"]
    assert not (tmp_path / "escaped.py").exists()
    assert [path.name for path in workspace.iterdir()] == ["beta.py"]


def test_a_run_that_stopped_part_way_still_assembles_what_it_paid_for(
        tmp_path, monkeypatch
):
    """A budget stop is the normal way a CodeS run ends on a large design.

    The prompts it never answered must not reach the assembly stage, which
    reads the same files: one unanswered record there loses every file of a
    repository that was otherwise ready to be graded.
    """
    records = run_phases()
    records[FUNCTION_BODY][1].pop("parsed")
    records[FUNCTION_BODY][1].pop("generated")

    workspace, outcome = build_repository(tmp_path, monkeypatch, records)

    assert outcome["files_written"] == 2
    assert "# filled" in (workspace / "alpha.py").read_text(encoding="utf-8")
    assert (workspace / "beta.py").read_text(encoding="utf-8") == "beta_one"


class RecordingCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=RecordingCompletions(responses))


def response(
        content="ok", prompt=10, completion=5, cached=2, reasoning=1, cost=0.5,
        finish_reason="stop",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details={"cached_tokens": cached},
            completion_tokens_details={"reasoning_tokens": reasoning},
            cost=cost,
        ),
    )


class RateLimited(Exception):
    status_code = 429


def requester(responses, totals=None, waits=None, failures=None, **hyperparameters):
    client = FakeClient(responses)
    # The same split the agent makes: sampling comes from the shared file (or
    # from all three stated together), the rest is the solution's own.
    generation = resolve_generation_kwargs(hyperparameters)
    resolved = CODE_S.resolve(hyperparameters) | generation
    request = _requester(
        client,
        resolved,
        totals if totals is not None else UsageTotals(),
        _Budget(resolved, totals if totals is not None else UsageTotals()),
        model="test/model",
        failures=failures,
        sleep=(waits if waits is not None else []).append,
    )
    return client, request


def test_the_request_the_tool_builds_carries_what_the_tool_omits():
    client, request = requester(
        [response()],
        max_tokens=4096,
        top_p=0.9,
        temperature=0.0,
        reasoning_effort="high",
        request_extra={"reasoning": {"enabled": False}},
    )

    request({"instruction": "write the file"})

    call = client.chat.completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": "write the file"}]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 4096
    assert call["top_p"] == 0.9
    assert call["reasoning_effort"] == "high"
    assert call["extra_body"] == {"reasoning": {"enabled": False}}


def test_the_two_the_tool_omits_are_sent_whether_or_not_they_were_stated():
    """The tool's own request carries model, temperature and messages only.

    All three sampling parameters are sent here regardless, because they are
    not this solution's to leave out: every arm sends the same three, from the
    same file, or the comparison is between token budgets as much as between
    methods. Reproducing the tool's request exactly would mean letting the
    provider choose two of them, differently from every other arm.
    """
    client, request = requester([response()])

    request({"instruction": "write the file"})

    call = client.chat.completions.calls[0]
    assert sorted(call) == ["max_tokens", "messages", "model", "temperature", "top_p"]
    assert {name: call[name] for name in resolve_generation_kwargs({})} == (
        resolve_generation_kwargs({})
    )


def test_a_retried_request_uses_the_temperature_the_tool_retries_at():
    client, request = requester(
        [ValueError("upstream hiccup"), response()],
        temperature=0.0,
        top_p=0.95,
        max_tokens=4096,
        retry_temperature=0.25,
    )

    request({"instruction": "write the file"})

    assert [call["temperature"] for call in client.chat.completions.calls] == [
        0.0,
        0.25,
    ]


def test_a_run_records_what_every_response_reported_spending():
    totals = UsageTotals()
    _, request = requester(
        [response(), response(prompt=20, completion=7)], totals=totals
    )

    request({"instruction": "one"})
    request({"instruction": "two"})

    assert totals.input_tokens == 30
    assert totals.output_tokens == 12
    assert totals.cached_input_tokens == 4
    assert totals.reasoning_tokens == 2
    assert totals.cost_usd == 1.0
    assert totals.responses == 2


def test_a_throttled_request_is_waited_out_rather_than_losing_the_trial():
    waits: list[float] = []
    _, request = requester([RateLimited("429 rate limit"), response()], waits=waits)

    request({"instruction": "write the file"})

    assert waits == [run_codes.RATE_LIMIT_WAITS[0]]


def test_a_rejected_request_is_retried_without_waiting_for_a_rate_limit():
    """The tool retries any failure; only a throttled one is worth waiting for."""
    waits: list[float] = []
    _, request = requester([ValueError("upstream hiccup"), response()], waits=waits)

    request({"instruction": "write the file"})

    assert waits == []


def test_an_empty_completion_is_a_failed_attempt_rather_than_an_empty_file():
    client, request = requester([response(content=""), response(content="written")])

    assert request({"instruction": "write the file"}) == "written"
    assert len(client.chat.completions.calls) == 2


def test_a_request_that_is_refused_throughout_names_the_model_and_the_file():
    _, request = requester([RateLimited("429")] * 5, request_attempts=5)

    with pytest.raises(RuntimeError) as error:
        request({"instruction": "write it", "file_path": "alpha.py"})

    assert "test/model" in str(error.value)
    assert "alpha.py" in str(error.value)


def test_the_tools_own_attempt_count_bounds_the_requests_a_prompt_costs():
    """Waiting out a rate limit must not multiply the attempts a run asked for."""
    client, request = requester([RateLimited("429")] * 3, request_attempts=3)

    with pytest.raises(RuntimeError):
        request({"instruction": "write it"})

    assert len(client.chat.completions.calls) == 3


def test_a_sketch_cut_off_at_the_ceiling_is_counted_rather_than_scored():
    """CodeS's failure mode at the ceiling is a parse failure, not a test failure.

    FileSketcher returns a whole file of signatures per response and
    SketchFiller returns a whole function. A response cut off mid-definition
    does not fail a test — the file it belonged to is written as an unfilled
    sketch — so the reward says nothing about what happened.
    """
    failures = FailureTally()
    _, request = requester([response(finish_reason="length")], failures=failures)

    request({"instruction": "write the file"})

    assert failures.to_dict()["counts"]["output_limit_exhausted"] == 1
    assert failures.to_dict()["counts"]["context_exhausted"] == 0


def test_a_prompt_the_context_window_cannot_hold_is_counted_apart():
    """The phase most able to overflow: a function body prompt carries the
    file's own sketch and the sketches it imports."""

    class BadRequest(Exception):
        status_code = 400

    failures = FailureTally()
    refusal = BadRequest("maximum context length is 65536 tokens")
    _, request = requester(
        [refusal, refusal], request_attempts=2, failures=failures
    )

    with pytest.raises(RuntimeError):
        request({"instruction": "write the file"})

    counts = failures.to_dict()["counts"]
    assert counts["context_exhausted"] == 2
    assert counts["output_limit_exhausted"] == 0
    assert counts["request_timeout"] == 0


def test_the_run_records_the_sampling_every_phase_used(tmp_path, monkeypatch):
    """Every request in this pipeline is built from the same three values.

    `retry_temperature` is recorded beside them because it is the temperature an
    attempt after the first actually used, which the shared value alone does not
    say. The rest of the pipeline's shape is recorded elsewhere in the same file.
    """
    monkeypatch.setattr(
        run_codes, "RESOLVED_SETUP_PATH", tmp_path / "resolved-setup.json"
    )
    monkeypatch.setattr(run_codes, "CODES_ROOT", tmp_path)
    hyperparameters = resolve_generation_kwargs({}) | CODE_S.resolve({})

    run_codes._record_resolved_setup(
        hyperparameters,
        {phase: [] for phase in run_codes.PHASES},
        None,
        None,
        ServedProviders(),
    )

    recorded = json.loads(
        (tmp_path / "resolved-setup.json").read_text(encoding="utf-8")
    )
    shared = resolve_generation_kwargs({})
    assert {name: recorded["generation"][name] for name in shared} == shared
    assert recorded["generation"]["source"] == "configs/generation.yaml"
    assert recorded["generation"]["retry_temperature"] == 0.1
    assert recorded["failures"]["counts"] == {
        "output_limit_exhausted": 0,
        "context_exhausted": 0,
        "request_timeout": 0,
    }


@pytest.mark.parametrize(
    "hyperparameters, expected",
    [
        # The shape every configuration in this repository uses to turn
        # reasoning off. Read as a truthy dict it recorded the opposite of what
        # it asked for, beside a `reasoning_tokens` of zero that said so.
        ({"request_extra": {"reasoning": {"enabled": False}}}, False),
        ({"request_extra": {"reasoning": {"enabled": True}}}, True),
        ({"request_extra": {"reasoning": {"effort": "high"}}}, True),
        # Reasoning paid for and hidden from the reply is still reasoning.
        ({"request_extra": {"reasoning": {"exclude": True}}}, True),
        # A request extra that is about something else entirely.
        ({"request_extra": {"provider": {"order": ["baidu/fp8"]}}}, False),
        ({"reasoning_effort": "high"}, True),
        ({"reasoning_effort": "none"}, False),
        ({}, False),
    ],
)
def test_the_record_says_whether_reasoning_was_asked_for_not_mentioned(
        hyperparameters, expected
):
    assert run_codes._reasoning_requested(hyperparameters) is expected


def test_the_record_says_which_file_cost_what_and_whether_its_sketch_survived():
    """A run's shape is decided by two replies nobody configured.

    `files_sketched` and `functions_requested` say a run fanned out; this says
    which file did it, and that the sketch which declared them was a reply cut
    off at the output ceiling rather than a design.
    """
    records = {
        REPO_SKETCH: [],
        FILE_SKETCH: [
            {
                "file_path": "small.py",
                "parsed": "def a():\n    pass\n",
                "finish_reason": "stop",
            },
            {
                "file_path": "runaway.py",
                # Cut off mid-definition: the ceiling, not a design.
                "parsed": "def one():\n    pass\n\n\ndef two_",
                "finish_reason": "length",
            },
            # Asked for, never answered — the budget stopped the phase first.
            {"file_path": "unanswered.py"},
        ],
        FUNCTION_BODY: [
            {"current_file_path": "small.py", "generated": "..."},
            {"current_file_path": "runaway.py", "generated": "..."},
            {"current_file_path": "runaway.py"},
            {"current_file_path": "runaway.py"},
        ],
    }

    files = {each["file_path"]: each for each in run_codes._per_file(records)}

    # Widest first, so the file that set the size of the run reads first.
    assert [each["file_path"] for each in run_codes._per_file(records)][0] == "runaway.py"
    assert files["runaway.py"]["functions_requested"] == 3
    assert files["runaway.py"]["functions_answered"] == 1
    assert files["runaway.py"]["sketch_parses"] is False
    assert files["runaway.py"]["sketch_finish_reason"] == "length"
    assert files["small.py"]["functions_requested"] == 1
    assert files["small.py"]["sketch_parses"] is True
    assert files["unanswered.py"]["sketch_answered"] is False
    assert files["unanswered.py"]["sketch_parses"] is None
    assert files["unanswered.py"]["sketch_chars"] is None


def test_a_response_records_the_reason_it_ended_on_the_prompt_it_answered():
    """The tally counts how many replies hit the ceiling; this says which."""
    client, request = requester([response(finish_reason="length")])
    prompt = {"instruction": "sketch a file", "file_path": "wide.py"}

    request(prompt)

    assert prompt["finish_reason"] == "length"


def test_a_rate_limit_is_recognised_by_status_before_message():
    assert _is_rate_limit(RateLimited("throttled"))
    assert _is_rate_limit(RuntimeError("HTTP 429 Too Many Requests"))
    assert not _is_rate_limit(SimpleNamespace(status_code=401, args=()))
    assert not _is_rate_limit(RuntimeError("invalid api key"))


def test_a_budget_stops_a_pipeline_whose_length_the_model_chose():
    totals = UsageTotals()
    resolved = CODE_S.resolve({"max_token_budget": 25})
    budget = _Budget(resolved, totals)

    budget.check()
    totals.input_tokens, totals.output_tokens = 20, 10
    with pytest.raises(BudgetExhausted, match="max_token_budget"):
        budget.check()


def test_a_pipeline_without_a_budget_is_bounded_only_by_harbors_timeout():
    budget = _Budget(CODE_S.resolve({}), UsageTotals())

    budget.check()

    assert budget.seconds is None
    assert budget.tokens is None


def test_the_client_leaves_the_retrying_to_the_recorded_attempts(monkeypatch):
    """Fifteen requests where the configuration asked for five.

    The OpenAI client waits ten minutes on every read and retries twice on its
    own by default. Beneath the attempts here that is two and a half hours
    spent on one file sketch, against a task allowed one — and spent inside a
    single call, where neither budget can see it.
    """
    built: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("API_KEY_ENV", "API_KEY")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://openrouter.ai/api/v1")

    _client(CODE_S.resolve({})["request_timeout_seconds"])

    assert built["max_retries"] == 0
    assert built["timeout"] == 600.0


class SpendingThenFailing:
    """A provider that charges for an attempt and then fails it."""

    def __init__(self, totals: UsageTotals, per_attempt: int):
        self.calls = 0
        self._totals = totals
        self._per_attempt = per_attempt

    def create(self, **_kwargs):
        self.calls += 1
        self._totals.output_tokens += self._per_attempt
        raise RuntimeError("the provider stopped answering")


def test_a_budget_already_spent_stops_the_attempts_a_prompt_has_left():
    """A prompt costs `request_attempts * request_timeout_seconds` at worst.

    Read once on the way into a prompt, a budget cannot end a run until every
    one of those attempts has been paid for. That is how a run reaches
    Harbor's timeout with a budget it spent long before, and a killed
    container writes no repository, no records and no usage at all.
    """
    totals = UsageTotals()
    resolved = CODE_S.resolve(
        {"request_attempts": 5, "max_token_budget": 25}
    ) | resolve_generation_kwargs({})
    completions = SpendingThenFailing(totals, per_attempt=20)
    request = _requester(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        resolved,
        totals,
        _Budget(resolved, totals),
        model="test/model",
        sleep=[].append,
    )

    with pytest.raises(BudgetExhausted, match="max_token_budget"):
        request({"instruction": "sketch a file"})

    # Two attempts made, three abandoned: the third was refused by the budget
    # rather than sent and waited out.
    assert completions.calls == 2


def test_an_unreadable_tool_repository_fails_instead_of_waiting_for_a_password(
        tmp_path,
):
    """A private or misspelled repository is a fast failure, not a stalled trial."""
    environment = install(CodeSAgent(logs_dir=tmp_path))

    clone = next(
        command for command in environment.commands if "git clone" in command["command"]
    )
    assert clone["env"]["GIT_TERMINAL_PROMPT"] == "0"
