import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    SINGLE_SHOT,
    ConfigurationError,
    resolve_generation_kwargs,
)
from evaluation_platform.model_routing import ServedProviders
from evaluation_platform.failure_categories import FailureTally
from evaluation_platform.model_usage import UsageTotals
from evaluation_platform import run_single_shot
from evaluation_platform.run_single_shot import (
    FORMAT_INSTRUCTION,
    _client,
    _common_root,
    _record_resolved_setup,
    _request,
    _strip_workspace_root,
    _write_files,
    build_prompt,
    parse_files,
)
from evaluation_platform.single_shot_agent import (
    HYPERPARAMETERS_PATH,
    OPENAI_PACKAGE,
    TASK_INSTRUCTION_PATH,
    SingleShotAgent,
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


def uploaded(environment) -> dict[str, str]:
    return dict(
        zip(
            [target for _, target in environment.uploads],
            environment.upload_contents,
        )
    )


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


class RateLimited(Exception):
    status_code = 429


def response(content="### FILE: a.py\n```python\nx = 1\n```", finish_reason="stop"):
    return SimpleNamespace(
        provider="Baidu",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details={"cached_tokens": 2},
            completion_tokens_details={"reasoning_tokens": 1},
            cost=0.5,
        ),
    )


def requester(
        responses, totals=None, waits=None, routing=None, observed=None,
        failures=None, **hyperparameters,
):
    client = FakeClient(responses)
    # The same split the agent makes: sampling comes from the shared file (or
    # from all three stated together), the rest is the solution's own.
    generation = resolve_generation_kwargs(hyperparameters)
    resolved = SINGLE_SHOT.resolve(hyperparameters) | generation

    def send():
        return _request(
            client,
            "the specification",
            resolved,
            totals if totals is not None else UsageTotals(),
            model="test/model",
            routing=routing,
            observed=observed,
            failures=failures,
            sleep=(waits if waits is not None else []).append,
        )

    return client, send


def test_the_sampling_a_run_used_is_written_into_its_own_record(tmp_path, monkeypatch):
    """What was actually sent, recorded next to the result it produced.

    Read back out of the values the request was built from rather than off the
    file that supplied them, so a run months old can be checked against its own
    sampling rather than against whatever that file says today.
    """
    monkeypatch.setattr(
        run_single_shot, "RESOLVED_SETUP_PATH", tmp_path / "resolved-setup.json"
    )
    hyperparameters = resolve_generation_kwargs({}) | SINGLE_SHOT.resolve({})

    _record_resolved_setup(
        hyperparameters=hyperparameters,
        finish_reason="stop",
        written=["pyproject.toml"],
        warnings=[],
        requests=1,
    )

    recorded = json.loads(
        (tmp_path / "resolved-setup.json").read_text(encoding="utf-8")
    )
    shared = resolve_generation_kwargs({})
    assert {name: recorded["generation"][name] for name in shared} == shared
    assert recorded["generation"]["source"] == "configs/generation.yaml"


def test_a_reply_cut_off_at_the_ceiling_is_counted_as_its_own_failure(
        tmp_path, monkeypatch
):
    """The finding a reward figure cannot carry.

    A baseline that ran out of output tokens and one that did not know the
    answer both score badly. `truncated` says which for this one reply; the
    category says it in the same words every other arm uses.
    """
    monkeypatch.setattr(
        run_single_shot, "RESOLVED_SETUP_PATH", tmp_path / "resolved-setup.json"
    )
    failures = FailureTally()
    _, send = requester([response(finish_reason="length")], failures=failures)

    send()
    _record_resolved_setup(
        hyperparameters=resolve_generation_kwargs({}) | SINGLE_SHOT.resolve({}),
        finish_reason="length",
        written=[],
        warnings=[],
        requests=1,
        failures=failures,
    )

    recorded = json.loads(
        (tmp_path / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["truncated"] is True
    assert recorded["failures"]["counts"]["output_limit_exhausted"] == 1
    # Kept apart from the two it is most often confused with.
    assert recorded["failures"]["counts"]["context_exhausted"] == 0
    assert recorded["failures"]["counts"]["request_timeout"] == 0


def test_a_prompt_the_context_window_cannot_hold_is_counted_apart(tmp_path):
    """Prompt plus ceiling overflowed the window, so nothing was generated.

    A different finding from a truncated reply, and from a provider that was
    slow: the remedy is a smaller prompt rather than a larger ceiling.
    """
    failures = FailureTally()
    refusal = BadRequest("This model's maximum context length is 65536 tokens")
    _, send = requester([refusal, refusal, refusal], failures=failures)

    with pytest.raises(RuntimeError):
        send()

    assert failures.to_dict()["counts"]["context_exhausted"] == 3
    assert failures.to_dict()["counts"]["output_limit_exhausted"] == 0


class BadRequest(Exception):
    """An OpenAI-style 400, carrying the provider's payload the way one does."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 400


def test_install_uploads_the_runner_and_pins_the_client(tmp_path):
    """The baseline has no repository to clone: the platform is the subject."""
    environment = RecordingEnvironment()
    agent = SingleShotAgent(logs_dir=tmp_path)

    asyncio.run(agent.install(cast(BaseEnvironment, cast(object, environment))))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert OPENAI_PACKAGE in commands
    assert "git clone" not in commands
    assert set(uploaded(environment)) == {
        "/installed-agent/run_single_shot.py",
        "/installed-agent/model_usage.py",
        "/installed-agent/model_routing.py",
        "/installed-agent/failure_categories.py",
    }


def test_run_uploads_instruction_instead_of_putting_it_on_the_command_line(tmp_path):
    environment = RecordingEnvironment()
    agent = SingleShotAgent(
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
    assert uploaded(environment)[TASK_INSTRUCTION_PATH] == instruction


def test_the_configured_hyperparameters_reach_the_container(tmp_path):
    environment = RecordingEnvironment()
    agent = SingleShotAgent(
        logs_dir=tmp_path, max_tokens=16384, temperature=0.3, top_p=0.9
    )

    asyncio.run(
        agent.run(
            "write it",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    hyperparameters = json.loads(uploaded(environment)[HYPERPARAMETERS_PATH])
    # The sampling the run resolved to, whichever of the two places it came
    # from, arrives beside the solution's own settings under the names the
    # runner reads.
    assert hyperparameters["max_tokens"] == 16384
    assert hyperparameters["temperature"] == 0.3
    assert hyperparameters["top_p"] == 0.9
    # Defaults are filled in, so what the container receives is the whole
    # setup rather than only the part that was written down.
    assert hyperparameters["request_attempts"] == 3


def test_sampling_left_unstated_comes_from_the_one_file_that_states_it(tmp_path):
    """An agent built without sampling reads the file every arm reads.

    The three values are not this baseline's to default. There is one place
    they are written down, and an agent that was handed none of them goes
    there rather than inventing figures of its own.
    """
    environment = RecordingEnvironment()
    agent = SingleShotAgent(logs_dir=tmp_path)

    asyncio.run(
        agent.run(
            "write it",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    hyperparameters = json.loads(uploaded(environment)[HYPERPARAMETERS_PATH])
    shared = resolve_generation_kwargs({})
    assert {name: hyperparameters[name] for name in shared} == shared


def test_some_but_not_all_of_the_sampling_parameters_is_refused(tmp_path):
    """They travel together, or a run samples at a value nobody chose."""
    with pytest.raises(ConfigurationError):
        SingleShotAgent(logs_dir=tmp_path, temperature=0.3)


@pytest.mark.parametrize(
    "setting",
    [
        {"repository": "https://example.invalid/tool.git"},
        {"commit": "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"},
        # Self-Collaboration's, which this baseline has no phase for.
        {"analyst_steps": 10},
        {"max_tokens": 0, "temperature": 0.0, "top_p": 0.95},
    ],
)
def test_the_baseline_rejects_settings_that_would_decide_nothing(tmp_path, setting):
    with pytest.raises(ConfigurationError):
        SingleShotAgent(logs_dir=tmp_path, **setting)


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
    agent = SingleShotAgent(logs_dir=tmp_path)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 900
    assert context.n_cache_tokens == 100
    assert context.n_output_tokens == 50
    assert context.cost_usd == 0.25


def test_the_prompt_is_the_task_plus_the_file_contract_and_nothing_else():
    """What separates a baseline from a fourth solution.

    Everything the model is told beyond the benchmark's own specification is
    the mechanical contract for naming files, which exists because the
    benchmark grades files on disk and a model emits text.
    """
    prompt = build_prompt("Implement a math verification library.")

    assert prompt.startswith("Implement a math verification library.")
    assert FORMAT_INSTRUCTION.strip() in prompt
    assert len(prompt) == len(
        "Implement a math verification library."
    ) + len("\n\n---\n\n") + len(FORMAT_INSTRUCTION.strip()) + 1


def test_a_file_containing_a_code_fence_survives_being_delivered_in_one():
    """A README with a fenced example is the ordinary case, not an exotic one.

    CommonMark closes a fence with at least as many of the same character as
    opened it, and the format instruction asks for exactly that.
    """
    files, warnings = parse_files(
        "### FILE: README.md\n"
        "````markdown\n"
        "Example:\n\n```python\nprint(1)\n```\n"
        "````\n"
    )

    assert files == [("README.md", "Example:\n\n```python\nprint(1)\n```\n")]
    assert warnings == []


def test_a_named_file_that_was_never_fenced_is_still_read():
    """Tolerant in one direction: never invent, but never discard either.

    Throwing away a section the model did write would understate the baseline,
    which is the one error this comparison cannot afford to make.
    """
    files, warnings = parse_files("## FILE: setup.py\nfrom setuptools import setup\n")

    # The blank lines around a section belong to the reply's layout rather
    # than to the file; the writer settles the trailing newline as it writes.
    assert files == [("setup.py", "from setuptools import setup")]
    assert any("no fenced code block" in warning for warning in warnings)


def test_a_reply_cut_off_mid_file_keeps_what_arrived_and_says_so():
    files, warnings = parse_files("### FILE: core.py\n```python\ndef add(a, b):\n")

    assert files == [("core.py", "def add(a, b):\n")]
    assert any("never closed" in warning for warning in warnings)


def test_a_section_that_named_no_file_is_not_a_file():
    files, warnings = parse_files("```python\nx = 1\n```\n")

    assert files == []
    assert warnings == []


def test_a_path_written_as_markdown_is_read_as_a_path():
    files, _ = parse_files("### FILE: `src/x.py`\n```python\nx = 1\n```\n")

    assert files == [("src/x.py", "x = 1\n")]


def test_a_file_named_twice_keeps_the_later_one_and_records_it():
    files, warnings = parse_files(
        "### FILE: x.py\n```python\nfirst = 1\n```\n"
        "### FILE: x.py\n```python\nsecond = 2\n```\n"
    )

    assert [content for _, content in files] == ["first = 1\n", "second = 2\n"]
    assert any("more than once" in warning for warning in warnings)


def test_an_empty_file_block_is_recorded_rather_than_written():
    files, warnings = parse_files("### FILE: x.py\n```python\n```\n")

    assert files == []
    assert any("empty" in warning for warning in warnings)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.py", "src/../../outside.py", ".git/config"],
)
def test_a_path_outside_the_project_is_refused(tmp_path, monkeypatch, path):
    """The reply is untrusted text, and the workspace is what is graded."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(run_single_shot, "WORKSPACE", workspace)

    written = _write_files([(path, "payload\n"), ("kept.py", "x = 1\n")])

    assert written == ["kept.py"]
    assert not (tmp_path / "outside.py").exists()


def test_files_are_written_where_the_benchmark_grades_them(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(run_single_shot, "WORKSPACE", workspace)

    written = _write_files(
        [("pyproject.toml", "[project]\n"), ("src/x/core.py", "def add():\n    ...")]
    )

    assert written == ["pyproject.toml", "src/x/core.py"]
    assert (workspace / "src" / "x" / "core.py").read_text(
        encoding="utf-8"
    ) == "def add():\n    ...\n"


def test_a_workspace_root_copied_out_of_the_specification_is_dropped():
    """A correction for the apparatus, not a helping hand.

    NL2RepoBench draws the project as a tree rooted at `workspace/`, the
    directory the task mounts it in. A model writing paths as text can
    reproduce that root as though it were part of the project, which would put
    the package one level below where the tester looks and score a correct
    library zero. The solutions never face this: they write through tools into
    the working directory, so the root never materializes.
    """
    files, stripped = _strip_workspace_root(
        [("workspace/pyproject.toml", "a"), ("workspace/src/x.py", "b")]
    )

    assert files == [("pyproject.toml", "a"), ("src/x.py", "b")]
    assert stripped == "workspace"


@pytest.mark.parametrize(
    "files",
    [
        # The project's own name is the model's choice, not the benchmark's
        # phrasing, and is left exactly as it was.
        [("math_verify/x.py", "a")],
        # Only when every file shares it.
        [("workspace/x.py", "a"), ("pyproject.toml", "b")],
        # A file called `workspace` is a file, not a root.
        [("workspace", "a")],
        [],
    ],
)
def test_nothing_else_is_moved(files):
    assert _strip_workspace_root(files) == (files, None)


def test_a_project_wrapped_in_a_directory_is_recorded_rather_than_rewritten():
    """A reply that nests everything scores zero for a reason that is not code.

    Moving the files would be the runner deciding what the model meant. The
    observation is recorded instead, so a zero of this kind is legible in the
    run rather than guessed at from the reward.
    """
    assert _common_root(["math_verify/core.py", "math_verify/pyproject.toml"]) == (
        "math_verify"
    )
    assert _common_root(["pyproject.toml", "src/x/core.py"]) is None
    assert _common_root([]) is None


def test_the_request_carries_the_configured_sampling_and_the_one_message():
    client, send = requester(
        [response()],
        max_tokens=16384,
        temperature=0.0,
        top_p=0.9,
        reasoning_effort="high",
        request_extra={"reasoning": {"enabled": False}},
    )

    send()

    call = client.chat.completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": "the specification"}]
    assert call["max_tokens"] == 16384
    assert call["temperature"] == 0.0
    assert call["top_p"] == 0.9
    assert call["reasoning_effort"] == "high"
    assert call["extra_body"] == {"reasoning": {"enabled": False}}


def test_the_experiments_routing_travels_with_the_request():
    """The one request this arm makes goes to the server the experiment pinned.

    Merged into whatever the configuration already asked the provider for, so
    a routing directive and a reasoning setting do not displace each other.
    """
    routing = {"order": ["baidu/fp8"], "allow_fallbacks": False}
    client, send = requester(
        [response()],
        routing=routing,
        request_extra={"reasoning": {"enabled": False}},
    )

    send()

    assert client.chat.completions.calls[0]["extra_body"] == {
        "reasoning": {"enabled": False},
        "provider": routing,
    }


def test_which_server_answered_is_read_back_off_the_reply():
    """Pinning states an intent; the recorded provider proves it held."""
    observed = ServedProviders()
    _, send = requester([response()], observed=observed)

    send()

    assert observed.names() == ["Baidu"]


def test_a_throttled_request_waits_and_is_sent_again():
    """One request means one chance, so a provider hiccup would cost the task."""
    waits: list[float] = []
    _, send = requester([RateLimited(), response()], waits=waits, request_attempts=3)

    answer, finish_reason = send()

    assert waits == [15]
    assert finish_reason == "stop"
    assert "FILE: a.py" in answer


def test_a_refused_request_is_not_waited_on():
    """A rejected credential is rejected just as quickly the second time."""
    waits: list[float] = []
    _, send = requester(
        [ValueError("no such model"), response()], waits=waits, request_attempts=2
    )

    send()

    assert waits == []


def test_the_client_leaves_the_retrying_to_the_recorded_attempts(monkeypatch):
    """Nine requests where the configuration asked for three.

    The OpenAI client retries twice on its own by default. Multiplied by the
    attempts here that is nine requests and up to an hour and a half against a
    task allowed one hour: a trial that dies at its timeout having recorded
    nothing about why.
    """
    built: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("API_KEY_ENV", "API_KEY")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("BASE_URL", "https://openrouter.ai/api/v1")

    _client(600)

    assert built["max_retries"] == 0
    assert built["timeout"] == 600.0
    assert built["api_key"] == "secret"


def test_the_wait_is_narrated_so_an_empty_log_is_not_a_silent_stall(capsys):
    """The reply is the first thing written, and it takes minutes to arrive.

    Without a line before the request the log stays empty for the whole of it,
    which reads exactly like a run that is stuck.
    """
    _, send = requester(
        [response()], max_tokens=32768, temperature=0.0, top_p=0.95
    )

    send()

    printed = capsys.readouterr().out
    assert "test/model" in printed
    assert "32768" in printed


def test_a_reply_that_stopped_at_the_token_ceiling_reports_it():
    _, send = requester([response(finish_reason="length")])

    _, finish_reason = send()

    assert finish_reason == "length"


def test_what_the_one_request_spent_is_accumulated():
    totals = UsageTotals()
    _, send = requester([response()], totals=totals)

    send()

    assert totals.input_tokens == 10
    assert totals.output_tokens == 5
    assert totals.cached_input_tokens == 2
    assert totals.reasoning_tokens == 1
    assert totals.responses == 1


def test_a_request_refused_every_time_names_the_model_and_the_last_failure():
    _, send = requester(
        [ValueError("upstream is down")] * 2, waits=[], request_attempts=2
    )

    with pytest.raises(RuntimeError) as failure:
        send()

    assert "test/model" in str(failure.value)
    assert "upstream is down" in str(failure.value)
