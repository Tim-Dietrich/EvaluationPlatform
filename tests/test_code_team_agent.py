import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.code_team_agent import (
    CODE_TEAM_COMMIT,
    CODE_TEAM_ROOT,
    HYPERPARAMETERS_PATH,
    RUNTIME_PACKAGES,
    TASK_INSTRUCTION_PATH,
    TASK_WORKSPACE,
    VECTOR_RAG_PACKAGES,
    CodeTeamAgent,
)
from evaluation_platform.experiment_config import (
    CODE_TEAM,
    ConfigurationError,
    resolve_generation_kwargs,
)
from evaluation_platform.model_usage import UsageTotals
from evaluation_platform import run_code_team
from evaluation_platform.failure_categories import FailureTally
from evaluation_platform.model_routing import ServedProviders
from evaluation_platform.run_code_team import (
    _build_config,
    _build_context,
    _instrumented_client,
    _is_rate_limit,
    _record_resolved_setup,
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


def install(agent: CodeTeamAgent) -> RecordingEnvironment:
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
    environment = install(CodeTeamAgent(logs_dir=tmp_path))

    commands = "\n".join(command["command"] for command in environment.commands)
    assert CODE_TEAM_COMMIT in commands
    assert f"git clone --quiet https://github.com/WhitenWhiten/CodeTeam.git {CODE_TEAM_ROOT}" in commands
    for package in RUNTIME_PACKAGES:
        assert package in commands
    targets = [target for _, target in environment.uploads]
    # The runner imports the shared usage accounting and the shared provider
    # routing by their flat names, so both have to
    # arrive beside it.
    assert targets == [
        "/installed-agent/run_code_team.py",
        "/installed-agent/model_usage.py",
        "/installed-agent/model_routing.py",
        "/installed-agent/failure_categories.py",
    ]


def test_install_leaves_out_the_retrieval_stack_a_run_will_not_use(tmp_path):
    without = install(CodeTeamAgent(logs_dir=tmp_path))
    with_rag = install(CodeTeamAgent(logs_dir=tmp_path, rag_enabled=True))

    def install_command(environment):
        return next(
            command["command"]
            for command in environment.commands
            if "pip install" in command["command"]
        )

    for package in VECTOR_RAG_PACKAGES:
        assert package not in install_command(without)
        assert package in install_command(with_rag)


def test_install_leaves_out_the_vector_stack_for_lexical_retrieval(tmp_path):
    environment = install(
        CodeTeamAgent(logs_dir=tmp_path, rag_enabled=True, rag_backend="lexical")
    )

    commands = "\n".join(command["command"] for command in environment.commands)
    assert "numpy" in commands
    for package in VECTOR_RAG_PACKAGES:
        assert package not in commands


def test_run_uploads_instruction_instead_of_putting_it_on_docker_command_line(tmp_path):
    environment = RecordingEnvironment()
    agent = CodeTeamAgent(
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
        "CODE_TEAM_HYPERPARAMETERS_PATH": HYPERPARAMETERS_PATH,
    }
    assert uploaded(environment)[TASK_INSTRUCTION_PATH] == instruction
    assert invocation["cwd"] == TASK_WORKSPACE


def test_run_uploads_the_complete_setup_including_filled_in_defaults(tmp_path):
    environment = RecordingEnvironment()
    agent = CodeTeamAgent(logs_dir=tmp_path, architects=6, max_qa_rounds=4)

    asyncio.run(
        agent.run(
            "Build the library",
            cast(BaseEnvironment, cast(object, environment)),
            AgentContext(),
        )
    )

    hyperparameters = json.loads(uploaded(environment)[HYPERPARAMETERS_PATH])
    assert hyperparameters["architects"] == 6
    assert hyperparameters["max_qa_rounds"] == 4
    assert hyperparameters["dynamic_developer_allocation"] is True
    # The sampling every role's requests will carry, from the one file that
    # sets it for every arm, beside the solution's own settings.
    shared = resolve_generation_kwargs({})
    assert {name: hyperparameters[name] for name in shared} == shared


@pytest.mark.parametrize(
    "hyperparameter",
    [
        {"architects": 0},
        {"git_coordination": "yes"},
        {"rag_backend": "elasticsearch"},
        # Self-Collaboration's, which CodeTeam has no role for.
        {"analyst_steps": 10},
        # Sampling is not this solution's to state; it is set for every arm at
        # once in `configs/generation.yaml`. Before that file, this arm was the
        # one that sampled at 0.2 where every other sampled at 0.0.
        {"temperature": 0.2},
    ],
)
def test_agent_rejects_hyperparameters_the_runner_cannot_use(tmp_path, hyperparameter):
    with pytest.raises(ConfigurationError):
        CodeTeamAgent(logs_dir=tmp_path, **hyperparameter)


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
    agent = CodeTeamAgent(logs_dir=tmp_path)
    context = AgentContext()

    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 900
    assert context.n_cache_tokens == 100
    assert context.n_output_tokens == 50
    assert context.cost_usd == 0.25


# The tool's own configuration, reduced to the shape the runner writes into.
# Using a stand-in rather than the checked-out tool keeps the mapping under
# test on a machine that has not initialised the submodule.


@dataclass
class FakeLLMConfig:
    provider: str = "mock"
    model: str = "Qwen2.5-72B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 8192
    base_url: str | None = None


@dataclass
class FakeRAGConfig:
    enabled: bool = False
    index_backend: str = "faiss_hnsw"
    top_k: int = 5


@dataclass
class FakeAllocationConfig:
    dynamic_enabled: bool = True
    fixed_agents: int = 4
    assignment_seed: int | None = None


class FakeSystemConfig:
    def __init__(self):
        self.workspace = "./workspace"
        self.artifacts_enabled = True
        self.artifacts_dir = None
        self.async_mode = True
        self.architects = 4
        self.architect_seed = None
        self.sds_retry = 1
        self.max_rounds = 2
        self.preprocess_requirements = True
        self.max_wall_clock_seconds = None
        self.max_token_budget = None
        self.llm = FakeLLMConfig()
        self.rag = FakeRAGConfig()
        self.git = SimpleNamespace(enabled=True)
        self.developer_allocation = FakeAllocationConfig()


def build_config(monkeypatch, **hyperparameters: Any):
    monkeypatch.setenv("MODEL", "test/model")
    monkeypatch.setenv("BASE_URL", "https://openrouter.ai/api/v1")
    # The same split the agent makes: sampling comes from the shared file (or
    # from all three stated together), the rest is the solution's own.
    generation = resolve_generation_kwargs(hyperparameters)
    return _build_config(
        SimpleNamespace(SystemConfig=FakeSystemConfig),
        CODE_TEAM.resolve(hyperparameters) | generation,
    )


def test_the_configuration_decides_where_the_repository_and_artifacts_land(monkeypatch):
    config = build_config(monkeypatch)

    # The verifier grades the workspace itself, and the tool's own run
    # artifacts are evidence about the run rather than part of the repository.
    assert config.workspace == str(run_code_team.WORKSPACE)
    assert config.artifacts_dir == str(run_code_team.ARTIFACTS_DIR)
    assert not config.artifacts_dir.startswith(str(run_code_team.WORKSPACE))


def test_qa_rounds_reach_the_setting_the_tool_calls_max_rounds(monkeypatch):
    config = build_config(monkeypatch, max_qa_rounds=5)

    assert config.max_rounds == 5


def test_each_ablation_is_one_key_in_the_experiment_configuration(monkeypatch):
    config = build_config(
        monkeypatch,
        rag_enabled=True,
        rag_backend="lexical",
        dynamic_developer_allocation=False,
        fixed_developer_agents=3,
        git_coordination=False,
    )

    assert config.rag.enabled is True
    assert config.rag.index_backend == "lexical"
    assert config.developer_allocation.dynamic_enabled is False
    assert config.developer_allocation.fixed_agents == 3
    assert config.git.enabled is False


def test_sampling_and_seeds_are_taken_from_the_configuration(monkeypatch):
    config = build_config(
        monkeypatch,
        temperature=0.0,
        top_p=0.9,
        max_tokens=4096,
        architect_seed=7,
        developer_assignment_seed=11,
    )

    # `temperature` has no environment override in the tool, so a run
    # configured here would otherwise sample at the tool's own default — 0.2,
    # where every other arm samples at 0.0.
    assert config.llm.temperature == 0.0
    assert config.llm.top_p == 0.9
    assert config.llm.max_tokens == 4096
    assert config.architect_seed == 7
    assert config.developer_allocation.assignment_seed == 11
    assert config.llm.model == "test/model"
    assert config.llm.base_url == "https://openrouter.ai/api/v1"


def test_the_generated_repository_is_the_workspace_harbor_grades(tmp_path, monkeypatch):
    monkeypatch.setattr(run_code_team, "WORKSPACE", tmp_path / "workspace")

    class FakeContext:
        def __init__(self, cfg, llm, rag, artifacts):
            self.cfg, self.llm, self.rag, self.artifacts = cfg, llm, rag, artifacts

        def make_repo_root(self) -> str:
            return "should-not-be-reached"

    context = _build_context(
        SimpleNamespace(Context=FakeContext),
        SimpleNamespace(RunArtifacts=lambda directory, enabled: (directory, enabled)),
        config=SimpleNamespace(artifacts_enabled=True),
        llm=object(),
        rag=None,
    )

    assert context.make_repo_root() == str(tmp_path / "workspace")
    assert (tmp_path / "workspace").is_dir()


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
        self.api_key = "secret"


def response(prompt=10, completion=5, cached=2, reasoning=1, cost=0.5):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details={"cached_tokens": cached},
            completion_tokens_details={"reasoning_tokens": reasoning},
            cost=cost,
        )
    )


class RateLimited(Exception):
    status_code = 429


def test_reasoning_settings_are_added_to_a_request_the_tool_builds():
    client = FakeClient([response()])
    totals = UsageTotals()

    proxy = _instrumented_client(
        client,
        totals,
        model="test/model",
        reasoning_effort="high",
        request_extra={"reasoning": {"enabled": False}},
    )
    proxy.chat.completions.create(model="test/model", messages=[])

    call = client.chat.completions.calls[0]
    assert call["reasoning_effort"] == "high"
    assert call["extra_body"] == {"reasoning": {"enabled": False}}
    # Anything the proxy does not intercept is the real client's.
    assert proxy.api_key == "secret"


def test_a_run_records_what_every_response_reported_spending():
    client = FakeClient([response(), response(prompt=20, completion=7)])
    totals = UsageTotals()
    proxy = _instrumented_client(
        client, totals, model="test/model", reasoning_effort=None, request_extra=None
    )

    proxy.chat.completions.create(messages=[])
    proxy.chat.completions.create(messages=[])

    assert totals.input_tokens == 30
    assert totals.output_tokens == 12
    assert totals.cached_input_tokens == 4
    assert totals.reasoning_tokens == 2
    assert totals.cost_usd == 1.0
    assert totals.responses == 2


def test_a_truncated_reply_is_counted_wherever_the_tool_reaches_the_api():
    """CodeTeam's most expensive failure, and the one a reward cannot name.

    A Developer returns one whole source file per response and the QA agent
    returns a JSON bundle; neither survives being cut off mid-way, and neither
    failure is the model writing wrong code. One call site serves every role,
    so counting there covers the Architects, the CTO, the Developers and QA.
    """
    failures = FailureTally()
    truncated = response()
    truncated.choices = [SimpleNamespace(finish_reason="length")]
    proxy = _instrumented_client(
        FakeClient([truncated]),
        UsageTotals(),
        model="test/model",
        reasoning_effort=None,
        request_extra=None,
        failures=failures,
    )

    proxy.chat.completions.create(messages=[])

    assert failures.to_dict()["counts"]["output_limit_exhausted"] == 1


def test_a_request_the_context_window_cannot_hold_is_counted_apart():
    """No generation happened at all, which is not a truncated reply."""

    class BadRequest(Exception):
        status_code = 400

    failures = FailureTally()
    proxy = _instrumented_client(
        FakeClient([BadRequest("maximum context length is 65536 tokens")]),
        UsageTotals(),
        model="test/model",
        reasoning_effort=None,
        request_extra=None,
        failures=failures,
    )

    with pytest.raises(BadRequest):
        proxy.chat.completions.create(messages=[])

    counts = failures.to_dict()["counts"]
    assert counts["context_exhausted"] == 1
    assert counts["output_limit_exhausted"] == 0
    assert counts["request_timeout"] == 0


def test_the_run_records_the_sampling_every_role_shared(tmp_path, monkeypatch):
    """Read off the tool's own config object, which its one client was built
    from, so the record is of what every role sampled at rather than of what a
    file said."""
    monkeypatch.setattr(
        run_code_team, "RESOLVED_SETUP_PATH", tmp_path / "resolved-setup.json"
    )
    monkeypatch.setattr(run_code_team, "CODE_TEAM_ROOT", tmp_path)
    config = build_config(monkeypatch, temperature=0.0, top_p=0.9, max_tokens=4096)

    _record_resolved_setup({}, config, None, ServedProviders())

    recorded = json.loads(
        (tmp_path / "resolved-setup.json").read_text(encoding="utf-8")
    )
    assert recorded["generation"]["temperature"] == 0.0
    assert recorded["generation"]["top_p"] == 0.9
    assert recorded["generation"]["max_tokens"] == 4096
    assert recorded["generation"]["source"] == "configs/generation.yaml"
    assert recorded["failures"]["counts"] == {
        "output_limit_exhausted": 0,
        "context_exhausted": 0,
        "request_timeout": 0,
    }


def test_a_throttled_request_is_waited_out_rather_than_losing_the_trial():
    client = FakeClient([RateLimited("429 rate limit"), response()])
    waits = []
    proxy = _instrumented_client(
        client,
        UsageTotals(),
        model="test/model",
        reasoning_effort=None,
        request_extra=None,
        sleep=waits.append,
    )

    proxy.chat.completions.create(messages=[])

    assert waits == [run_code_team.RATE_LIMIT_WAITS[0]]


def test_an_exhausted_rate_limit_names_the_model_it_gave_up_on():
    client = FakeClient([RateLimited("429")] * (len(run_code_team.RATE_LIMIT_WAITS) + 1))
    proxy = _instrumented_client(
        client,
        UsageTotals(),
        model="moonshotai/kimi-k2.5",
        reasoning_effort=None,
        request_extra=None,
        sleep=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="moonshotai/kimi-k2.5"):
        proxy.chat.completions.create(messages=[])


def test_a_rejected_request_fails_immediately_rather_than_being_retried():
    client = FakeClient([ValueError("invalid model")])
    proxy = _instrumented_client(
        client,
        UsageTotals(),
        model="test/model",
        reasoning_effort=None,
        request_extra=None,
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError):
        proxy.chat.completions.create(messages=[])


def test_a_rate_limit_is_recognised_by_status_before_message():
    assert _is_rate_limit(RateLimited("throttled"))
    assert _is_rate_limit(RuntimeError("HTTP 429 Too Many Requests"))
    assert not _is_rate_limit(SimpleNamespace(status_code=401, args=()))
    assert not _is_rate_limit(RuntimeError("invalid api key"))


def test_an_unreadable_tool_repository_fails_instead_of_waiting_for_a_password(tmp_path):
    """A private or misspelled repository is a fast failure, not a stalled trial."""
    environment = install(CodeTeamAgent(logs_dir=tmp_path))

    clone = next(
        command for command in environment.commands if "git clone" in command["command"]
    )
    assert clone["env"]["GIT_TERMINAL_PROMPT"] == "0"
