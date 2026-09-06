from pathlib import Path
from typing import Any

import pytest
import yaml

from harbor.models.agent.name import AgentName

from main import archive_setup, build_harbor_command
from evaluation_platform.code_team_agent import CodeTeamAgent
from evaluation_platform.codes_agent import CodeSAgent
from evaluation_platform.experiment_config import (
    AGENT_HYPERPARAMETERS,
    DEFAULT_CONFIG_PATH,
    GENERATION_CONFIG_PATH,
    GENERATION_PARAMETERS,
    ConfigurationError,
    load_experiment_config,
    load_generation_config,
)
from evaluation_platform.self_collaboration_agent import SelfCollaborationAgent
from evaluation_platform.single_shot_agent import SingleShotAgent
from evaluation_platform.terminus_agent import TerminusAgent


ROOT = Path(__file__).parents[1]
ENVIRONMENT = {"API_KEY": "test-credential"}


def write_config(directory: Path, **overrides: Any) -> Path:
    document = {
        "name": "unit-test",
        "run": {"jobs_dir": "jobs", "n_attempts": 1},
        "benchmark": {
            "dataset": "nl2repobench/nl2repobench",
            "ref": "sha256:pinned",
            "task_names": ["nl2repobench/math-verify"],
        },
        "model": {
            "provider": "openrouter",
            "name": "test/free-model",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "API_KEY",
        },
        "agent": {
            "import_path": "evaluation_platform.self_collaboration_agent:Agent",
            "hyperparameters": {"max_rounds": 2, "test_command": "pytest -q"},
        },
    }
    document.update(overrides)
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_shipped_configuration_enables_the_tester_phase():
    config = load_experiment_config(ROOT / DEFAULT_CONFIG_PATH, ENVIRONMENT)

    # The Tester runs between Coder rounds, so a single round leaves the team
    # at Analyst -> Coder no matter what the test command says.
    assert config.hyperparameters["max_rounds"] >= 2
    assert config.hyperparameters["test_command"]


def test_hyperparameters_reach_harbor_as_recorded_agent_kwargs(tmp_path):
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    harbor_config = config.to_harbor_config("2026-01-01__00-00-00")

    agent = harbor_config["agents"][0]
    assert agent["kwargs"]["max_rounds"] == 2
    assert agent["kwargs"]["test_command"] == "pytest -q"
    assert agent["model_name"] == "openrouter/test/free-model"
    assert harbor_config["job_name"] == "2026-01-01__00-00-00"
    assert harbor_config["datasets"] == [
        {
            "name": "nl2repobench/nl2repobench",
            "ref": "sha256:pinned",
            "task_names": ["nl2repobench/math-verify"],
        }
    ]


def test_a_single_local_task_remains_available_as_an_escape_hatch(tmp_path):
    document = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    del document["benchmark"]
    document["task"] = {"path": "harbor_tasks/hand-written"}
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_experiment_config(path, ENVIRONMENT)

    assert config.benchmark is None
    assert config.to_harbor_config("job")["tasks"] == [
        {"path": "harbor_tasks/hand-written"}
    ]


def test_a_configuration_must_choose_between_a_benchmark_and_a_task(tmp_path):
    document = yaml.safe_load(write_config(tmp_path).read_text(encoding="utf-8"))
    document["task"] = {"path": "harbor_tasks/hand-written"}
    path = tmp_path / "both.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "exactly one" in str(error.value)


def test_a_dataset_must_be_registry_qualified(tmp_path):
    path = write_config(tmp_path, benchmark={"dataset": "nl2repobench"})

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "org/name" in str(error.value)


def test_omitted_hyperparameters_fall_back_to_recorded_defaults(tmp_path):
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    assert config.hyperparameters["analyst_steps"] == 10
    assert config.hyperparameters["coder_steps"] == 15


def is_sampling_file(path: Path) -> bool:
    """Whether a file in `configs/` sets sampling rather than describing a run.

    Told apart by shape rather than by name, because there is more than one:
    `generation.yaml` sets the three values for the NL2RepoBench comparison and
    `generation-humaneval.yaml` for the replication beside it. A sampling file
    states those three names and nothing else; an experiment configuration
    names a benchmark, a model and an agent.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set(document) <= set(GENERATION_PARAMETERS)


def shipped_configurations() -> list[Path]:
    """Every experiment configuration in `configs/`, and no sampling file."""
    return sorted(
        path
        for path in (ROOT / "configs").glob("*.yaml")
        if not is_sampling_file(path)
    )


def test_every_arm_samples_at_the_same_values_from_the_same_file():
    """The invariant the sampling file exists to hold.

    Temperature, top_p and the output ceiling describe the model call, not the
    method wrapped around it. Before this file the arms disagreed on two of the
    three — CodeTeam sampled at 0.2 where the rest sampled at 0.0, and the
    output ceilings spanned 8192 to 32768 — and neither difference announces
    itself in a reward figure.

    The invariant is per comparison rather than per directory, because there is
    now more than one comparison here: the arms of the NL2RepoBench comparison
    read `generation.yaml`, and the HumanEval replication reads the values the
    paper it reproduces states. What must hold — and what this checks — is that
    every configuration resolves its sampling from a file, that the file is one
    of the sampling files shipped beside it, and that two configurations
    reading the same file cannot disagree about a single value.
    """
    sampling_files = {
        str(path)
        for path in (ROOT / "configs").glob("*.yaml")
        if is_sampling_file(path)
    }
    assert str(GENERATION_CONFIG_PATH) in sampling_files

    resolved = {
        path.name: load_experiment_config(path, ENVIRONMENT).generation
        for path in shipped_configurations()
    }
    assert resolved, "no experiment configurations were found"

    by_file: dict[str, dict[str, tuple]] = {}
    for name, generation in resolved.items():
        assert generation.source in sampling_files, name
        by_file.setdefault(generation.source, {})[name] = tuple(
            sorted(generation.as_kwargs().items())
        )

    for source, arms in by_file.items():
        values = load_generation_config(path=Path(source))
        expected = tuple(sorted(values.as_kwargs().items()))
        for name, sampled in arms.items():
            assert sampled == expected, f"{name} against {source}"


def test_no_configuration_sets_sampling_of_its_own():
    """The other half of the invariant: one place, and no second one.

    A configuration that states one of the three is rejected before Docker
    starts rather than quietly sampling its arm differently from the arms it is
    compared with.
    """
    for path in shipped_configurations():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        stated = set(document["agent"].get("hyperparameters", {})) & set(
            GENERATION_PARAMETERS
        )
        assert not stated, f"{path.name} sets {', '.join(sorted(stated))}"


@pytest.mark.parametrize("parameter", sorted(GENERATION_PARAMETERS))
def test_a_configuration_that_states_sampling_is_told_where_it_belongs(
        tmp_path, parameter
):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.single_shot_agent:SingleShotAgent",
            "hyperparameters": {parameter: 1},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert GENERATION_CONFIG_PATH.name in str(error.value)


def test_no_arm_carries_a_sampling_default_of_its_own():
    """A default is a second place to set a value, and would outlive the file."""
    for solution in AGENT_HYPERPARAMETERS.values():
        overlap = set(solution.types) & set(GENERATION_PARAMETERS)
        assert not overlap, f"{solution.solution} declares {', '.join(overlap)}"


def test_the_archived_setup_pins_the_sampling_a_resume_continues_on(tmp_path):
    """A job resumed next week runs at the sampling its first half ran at.

    The snapshot is a record of what ran, in the way the benchmark's resolved
    digest is, so a resume reads it rather than reading whatever the central
    file says by then. It is the one document allowed to state these values,
    and only when it is read as an archive.
    """
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-single-shot.yaml", ENVIRONMENT
    )
    snapshot = tmp_path / "experiment-config.yaml"
    archive_setup(config, tmp_path)

    resumed = load_experiment_config(snapshot, ENVIRONMENT, archived=True)

    assert resumed.generation.as_kwargs() == config.generation.as_kwargs()
    assert resumed.generation.source == "the archived setup"
    # Read as a configuration rather than as an archive, the same file is
    # refused: a snapshot is not a second place to set these.
    with pytest.raises(ConfigurationError):
        load_experiment_config(snapshot, ENVIRONMENT)


def test_the_sampling_a_run_used_reaches_the_agent_that_will_send_it(tmp_path):
    """Harbor records agent kwargs in the job's `config.json`, so the values
    are on file at the job level as well as in each trial's own record."""
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    kwargs = config.to_harbor_config("job")["agents"][0]["kwargs"]

    assert {name: kwargs[name] for name in GENERATION_PARAMETERS} == (
        config.generation.as_kwargs()
    )


def test_agent_environment_forwards_only_the_model_backend(tmp_path):
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    environment = config.to_harbor_config("job")["agents"][0]["env"]

    assert environment == {
        "API_KEY": "${API_KEY}",
        "API_KEY_ENV": "API_KEY",
        "BASE_URL": "https://openrouter.ai/api/v1",
        "MODEL": "test/free-model",
        # Empty when a configuration pins no routing, which leaves the choice
        # of server to the aggregator.
        "MODEL_ROUTING": "{}",
    }
    assert "test-credential" not in yaml.safe_dump(environment)


def test_templates_resolve_against_the_environment(tmp_path):
    path = write_config(
        tmp_path,
        model={
            "provider": "${MODEL_PROVIDER:-openrouter}",
            "name": "${MODEL:-moonshotai/kimi-k2.5}",
            "base_url": "${BASE_URL:-https://openrouter.ai/api/v1}",
            "api_key_env": "API_KEY",
        },
    )

    configured = load_experiment_config(
        path,
        {
            **ENVIRONMENT,
            "MODEL_PROVIDER": "deepseek",
            "MODEL": "deepseek-v4-flash",
            "BASE_URL": "https://api.deepseek.com",
        },
    )
    defaulted = load_experiment_config(path, ENVIRONMENT)

    assert configured.model_label == "deepseek/deepseek-v4-flash"
    assert configured.base_url == "https://api.deepseek.com"
    assert defaulted.model_label == "openrouter/moonshotai/kimi-k2.5"
    assert defaulted.base_url == "https://openrouter.ai/api/v1"


def test_reasoning_settings_are_accepted_as_hyperparameters(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.self_collaboration_agent:Agent",
            "hyperparameters": {
                "reasoning_effort": "high",
                "request_extra": {"reasoning": {"effort": "high", "exclude": True}},
            },
        },
    )

    config = load_experiment_config(path, ENVIRONMENT)

    assert config.hyperparameters["reasoning_effort"] == "high"
    assert config.hyperparameters["request_extra"] == {
        "reasoning": {"effort": "high", "exclude": True}
    }
    # Recorded with the run, so the reasoning setting is part of the setup a
    # result can be traced back to.
    snapshot = config.to_snapshot()["agent"]["hyperparameters"]
    assert snapshot["reasoning_effort"] == "high"


def test_a_request_passthrough_that_cannot_be_sent_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.self_collaboration_agent:Agent",
            "hyperparameters": {"request_extra": "reasoning=high"},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "request_extra" in str(error.value)


def test_unknown_hyperparameter_is_rejected_instead_of_silently_ignored(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.self_collaboration_agent:Agent",
            "hyperparameters": {"max_step": 20},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "max_step" in str(error.value)
    assert "coder_steps" in str(error.value)


def test_unknown_section_key_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        benchmark={"dataset": "nl2repobench/nl2repobench", "tasks": ["box"]},
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "tasks" in str(error.value)
    assert "task_names" in str(error.value)


def test_missing_credential_fails_before_docker_starts(tmp_path):
    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(write_config(tmp_path), {})

    assert "API_KEY" in str(error.value)


def test_incorrect_deepseek_api_path_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        model={
            "provider": "deepseek",
            "name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/api/v1",
            "api_key_env": "API_KEY",
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "https://api.deepseek.com" in str(error.value)


def test_official_deepseek_api_url_is_accepted(tmp_path):
    path = write_config(
        tmp_path,
        model={
            "provider": "deepseek",
            "name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "API_KEY",
        },
    )

    assert load_experiment_config(path, ENVIRONMENT).base_url == (
        "https://api.deepseek.com"
    )


def test_setup_is_archived_next_to_the_job_without_the_credential(tmp_path):
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)
    job_dir = tmp_path / "jobs" / "2026-01-01__00-00-00"

    snapshot_path = archive_setup(config, job_dir)

    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["agent"]["hyperparameters"]["max_rounds"] == 2
    assert snapshot["agent"]["hyperparameters"]["coder_steps"] == 15
    assert snapshot["model"]["api_key_env"] == "API_KEY"
    assert "test-credential" not in snapshot_path.read_text(encoding="utf-8")


def test_launcher_invokes_harbor_from_the_project_environment():
    command = build_harbor_command(
        ROOT / "jobs" / "generated.yaml",
        python_executable=r"C:\project\.venv\Scripts\python.exe",
    )

    assert command[-2:] == ["--config", str(ROOT / "jobs" / "generated.yaml")]


def test_env_example_documents_the_credential_and_model_overrides():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "API_KEY=" in env_example
    assert "PYTHONUTF8=1" in env_example


def test_readme_runs_harbor_through_the_configuration_launcher():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe main.py" in readme
    assert "--env-file" not in readme
    assert "harbor.exe view .\\jobs --jobs" in readme


def test_concurrency_and_resilience_settings_reach_harbor(tmp_path):
    """The knobs that make a benchmark-scale run finish are part of the setup.

    Concurrency, retries, and resource overrides all change what a run does,
    so they belong in the configuration and in its archived snapshot rather
    than in a command line that leaves no record.
    """
    path = write_config(
        tmp_path,
        run={
            "jobs_dir": "jobs",
            "n_attempts": 1,
            "n_concurrent_trials": 6,
            "retry": {"max_retries": 2, "min_wait_sec": 5},
            "environment": {"override_memory_mb": 3072},
        },
    )

    config = load_experiment_config(path, ENVIRONMENT)
    harbor_config = config.to_harbor_config("2026-01-01__00-00-00")

    assert harbor_config["n_concurrent_trials"] == 6
    assert harbor_config["retry"] == {"max_retries": 2, "min_wait_sec": 5}
    assert harbor_config["environment"] == {"override_memory_mb": 3072}
    assert config.to_snapshot()["run"]["n_concurrent_trials"] == 6


def test_a_misspelled_retry_setting_is_rejected_rather_than_ignored(tmp_path):
    path = write_config(
        tmp_path,
        run={"jobs_dir": "jobs", "retry": {"max_retry": 2}},
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "run.retry" in str(error.value)
    assert "max_retries" in str(error.value)


def test_agent_concurrency_above_the_trial_limit_is_rejected(tmp_path):
    """`agent.n_concurrent` is a sub-limit, so it can never be the larger.

    Raising one without the other is the natural mistake when scaling a
    configuration up, and Harbor would otherwise reject it only once the run
    had already been launched.
    """
    path = write_config(
        tmp_path,
        run={"jobs_dir": "jobs", "n_concurrent_trials": 4},
        agent={
            "import_path": "evaluation_platform.self_collaboration_agent:Agent",
            "n_concurrent": 8,
            "hyperparameters": {"max_rounds": 2},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "n_concurrent" in str(error.value)
    assert "rate limit" in str(error.value)


def test_the_benchmark_configuration_runs_its_tasks_in_parallel():
    """The whole-benchmark configuration is the one that has to scale.

    104 tasks at roughly six minutes each is about ten hours in sequence, so a
    concurrency of one here would make the configuration unusable for its
    stated purpose.
    """
    config = load_experiment_config(
        ROOT / "configs" / "nl2repobench-self-collaboration.yaml", ENVIRONMENT
    )

    assert config.run["n_concurrent_trials"] > 1
    assert config.run["retry"]["max_retries"] >= 1


@pytest.mark.parametrize(
    "solution",
    [
        "math-verify-codeteam.yaml",
        "math-verify-codes.yaml",
        "math-verify-single-shot.yaml",
        "math-verify-terminus.yaml",
    ],
)
def test_the_solutions_are_compared_on_the_same_benchmark_and_model(solution):
    """What makes the math-verify configurations a comparison.

    The benchmark and the model decide what the run is measured against
    rather than what is being measured. Where those differ, a difference in
    the result is no longer attributable to the solution. `run` is excluded
    deliberately: how many attempts to make and how many to run at once is how
    much of the experiment to do, not what the experiment is, and the files are
    expected to differ there while one of them is being iterated on.
    """
    self_collaboration = yaml.safe_load(
        (ROOT / "configs" / "math-verify-self-collaboration.yaml").read_text(
            encoding="utf-8"
        )
    )
    other = yaml.safe_load(
        (ROOT / "configs" / solution).read_text(encoding="utf-8")
    )

    assert other["benchmark"] == self_collaboration["benchmark"]
    assert other["model"] == self_collaboration["model"]


def test_the_job_records_which_agent_ran(tmp_path):
    """Harbor's results view reads a job's agent column from this field.

    Every trial beneath a job records the agent that ran it, from the agent
    class's own `name()`. The job above them records nothing unless its agent
    configuration is given a name, which is why the label is stated in the
    registry and written here.
    """
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    agent = config.to_harbor_config("2026-01-01__00-00-00")["agents"][0]

    assert agent["name"] == "self-collaboration"
    assert agent["import_path"].startswith(
        "evaluation_platform.self_collaboration_agent"
    )


def test_no_arm_is_named_what_harbor_calls_one_of_its_own():
    """The collision that would silently run a different agent.

    Harbor's factory prefers a configured agent name over an `import_path`
    whenever that name is one of its built-ins, and never looks at the
    `import_path` at all in that case. An arm labelled `terminus-2` or `codex`
    would therefore run Harbor's agent of that name rather than the one the
    configuration points at, with no error anywhere to say so.
    """
    built_in = set(AgentName.values())

    for schema in AGENT_HYPERPARAMETERS.values():
        assert schema.label not in built_in, (
            f"{schema.solution} is labelled {schema.label!r}, which is one of "
            "Harbor's own agent names"
        )


def test_each_arms_label_is_the_name_its_agent_reports():
    """The job's agent column and its trials' should say the same thing.

    The label reaches the job record and the agent class's `name()` reaches
    every trial record beneath it. They are set in two places, so they are
    checked against each other here.
    """
    agents = {
        "evaluation_platform.self_collaboration_agent": SelfCollaborationAgent,
        "evaluation_platform.code_team_agent": CodeTeamAgent,
        "evaluation_platform.codes_agent": CodeSAgent,
        "evaluation_platform.single_shot_agent": SingleShotAgent,
        "evaluation_platform.terminus_agent": TerminusAgent,
    }

    assert set(agents) == set(AGENT_HYPERPARAMETERS)
    for module, agent_class in agents.items():
        assert agent_class.name() == AGENT_HYPERPARAMETERS[module].label


def test_the_shipped_single_shot_configuration_resolves_to_a_complete_setup():
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-single-shot.yaml", ENVIRONMENT
    )

    hyperparameters = config.hyperparameters
    assert config.agent_import_path.startswith("evaluation_platform.single_shot_agent")
    # The baseline is not somebody else's tool at a revision, and saying it
    # were would put a version in the record that decided nothing.
    assert config.agent_repository is None
    assert config.agent_commit is None
    # A whole repository has to fit in one reply. Below this the arm measures
    # the token ceiling rather than the model, which is the one way a baseline
    # can be unfair without looking it. It is the shared ceiling now, so the
    # requirement is on the file that sets it rather than on this arm.
    assert config.generation.max_tokens >= 16384
    # Reasoning is pinned rather than inherited, as in the configurations
    # beside this one.
    assert "request_extra" in hyperparameters or "reasoning_effort" in hyperparameters


def test_the_shipped_terminus_configuration_resolves_to_a_complete_setup():
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-terminus.yaml", ENVIRONMENT
    )

    hyperparameters = config.hyperparameters
    assert config.agent_import_path.startswith("evaluation_platform.terminus_agent")
    # Terminus ships inside Harbor; the Harbor release is the pin, and it is
    # recorded per run rather than written in the configuration.
    assert config.agent_repository is None
    assert config.agent_commit is None
    # Terminus bounds itself at a million turns, which is no bound at all, so
    # a configuration that states none leaves the trial's timeout as the only
    # limit and the arm's cost unstated.
    assert hyperparameters["max_turns"] >= 1
    assert "request_extra" in hyperparameters or "reasoning_effort" in hyperparameters


@pytest.mark.parametrize(
    "solution",
    [
        "math-verify-codeteam.yaml",
        "math-verify-codes.yaml",
        "math-verify-terminus.yaml",
    ],
)
def test_every_arm_that_can_pin_its_reasoning_pins_it_the_same_way(solution):
    """An arm that reasons where another does not is not a comparison.

    Self-Collaboration is the documented exception and is absent here: the
    revision under evaluation sends no reasoning field at all on its
    tool-calling path, so the setting cannot be stated without patching the
    tool. Everywhere it can be stated, it is stated identically, whether it
    reaches the provider through the OpenAI client or through LiteLLM.
    """
    baseline = load_experiment_config(
        ROOT / "configs" / "math-verify-single-shot.yaml", ENVIRONMENT
    ).hyperparameters
    other = load_experiment_config(
        ROOT / "configs" / solution, ENVIRONMENT
    ).hyperparameters

    assert other.get("request_extra") == baseline.get("request_extra")
    assert other.get("reasoning_effort") == baseline.get("reasoning_effort")


def test_the_shipped_codeteam_configuration_resolves_to_a_complete_setup():
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-codeteam.yaml", ENVIRONMENT
    )

    hyperparameters = config.hyperparameters
    assert config.agent_import_path.startswith("evaluation_platform.code_team_agent")
    # The QA role tests and repairs between rounds, so a run with none of them
    # is planning and implementation with nothing verifying the result.
    assert hyperparameters["max_qa_rounds"] >= 1
    assert hyperparameters["architects"] >= 1
    # Reasoning is pinned rather than inherited. Left unset, how much the model
    # reasons is the provider's default for it at that moment: set outside the
    # experiment, absent from its record, and free to change between runs.
    # Sampling is deliberately not asserted here; the configuration says which
    # values it uses and why, and iterating on them is expected.
    assert "request_extra" in hyperparameters or "reasoning_effort" in hyperparameters
    # A budget, because the tool has no bound of its own on what a task costs.
    assert hyperparameters["max_wall_clock_seconds"] >= 1
    assert hyperparameters["max_token_budget"] >= 1


def test_the_shipped_codes_configuration_resolves_to_a_complete_setup():
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-codes.yaml", ENVIRONMENT
    )

    hyperparameters = config.hyperparameters
    assert config.agent_import_path.startswith("evaluation_platform.codes_agent")
    # Reasoning is pinned rather than inherited, as in the two configurations
    # beside this one. Left unset, how much the model reasons is the provider's
    # default for it at that moment: set outside the experiment, absent from
    # its record, and free to change between runs.
    assert "request_extra" in hyperparameters or "reasoning_effort" in hyperparameters
    # A response truncated mid-definition does not fail a test, it fails to
    # parse, and CodeS returns a whole file or a whole function per response.
    assert config.generation.max_tokens >= 8192
    # A budget, because the tool has no bound of its own on what a task costs,
    # and unlike the other two solutions its length is chosen by the model
    # rather than by a number of rounds a configuration could lower.
    assert hyperparameters["max_wall_clock_seconds"] >= 1
    assert hyperparameters["max_token_budget"] >= 1


def test_the_codes_configuration_stays_within_the_provider_limit_it_declares():
    """Two concurrency limits multiply for CodeS, where elsewhere there is one.

    `agent.n_concurrent` caps how many trials call the model at once, and
    `concurrent_requests` is a second multiplier inside each of them. Their
    product is what reaches the provider, which is worth stating where the
    configuration is read rather than discovering as a rate limit.
    """
    config = load_experiment_config(
        ROOT / "configs" / "math-verify-codes.yaml", ENVIRONMENT
    )

    assert config.agent_n_concurrent is not None
    assert config.agent_n_concurrent <= config.run["n_concurrent_trials"]
    assert config.hyperparameters["concurrent_requests"] >= 1


def test_hyperparameters_are_validated_against_the_solution_that_will_run(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.code_team_agent:CodeTeamAgent",
            # Self-Collaboration's Coder budget, which CodeTeam has no role for.
            "hyperparameters": {"coder_steps": 15},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "coder_steps" in str(error.value)
    assert "CodeTeam" in str(error.value)


def test_a_configuration_naming_an_unintegrated_solution_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.metagpt_agent:MetaGPTAgent",
            "hyperparameters": {},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "metagpt_agent" in str(error.value)
    assert "evaluation_platform.code_team_agent" in str(error.value)


def test_an_ablation_is_a_boolean_a_configuration_states_rather_than_a_string(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.code_team_agent:CodeTeamAgent",
            "hyperparameters": {"git_coordination": "off"},
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "git_coordination" in str(error.value)


def test_a_seed_of_zero_is_a_seed_rather_than_an_empty_budget(tmp_path):
    """Zero bounds nothing here; it names one draw among many."""
    path = write_config(
        tmp_path,
        agent={
            "import_path": "evaluation_platform.code_team_agent:CodeTeamAgent",
            "hyperparameters": {"architect_seed": 0, "architects": 2},
        },
    )

    config = load_experiment_config(path, ENVIRONMENT)

    assert config.hyperparameters["architect_seed"] == 0
    with pytest.raises(ConfigurationError):
        load_experiment_config(
            write_config(
                tmp_path,
                agent={
                    "import_path": "evaluation_platform.code_team_agent:CodeTeamAgent",
                    "hyperparameters": {"architects": 0},
                },
            ),
            ENVIRONMENT,
        )


def test_a_benchmark_names_a_registry_dataset_or_a_directory_but_not_both(tmp_path):
    """The two ways a benchmark is named, and the one that is an error."""
    path = write_config(
        tmp_path,
        benchmark={"dataset": "org/name", "path": "benchmarks/humaneval/tasks"},
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "exactly one" in str(error.value)


def test_a_local_benchmark_cannot_state_a_registry_ref(tmp_path):
    """A ref pins a registry version, and a directory has no registry.

    What pins a local benchmark is a digest over its tasks, computed at launch.
    Stating a ref would put a version in the record that nothing resolved.
    """
    path = write_config(
        tmp_path,
        benchmark={"path": "benchmarks/humaneval/tasks", "ref": "sha256:abc"},
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "digest" in str(error.value)


# The revision every Self-Collaboration arm evaluates; see
# `self_collaboration_agent.SELF_COLLABORATION_COMMIT`.
SELF_COLLABORATION_COMMIT_UNDER_TEST = "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"


def test_the_shipped_humaneval_configuration_reproduces_the_papers_settings():
    """The configuration is the experiment, so the paper's numbers are in it.

    Every value below is quoted from the paper rather than chosen here, which
    is what makes the orchestrator the only thing this run varies. The model is
    the one exception, and it is the one the paper's own pin made unavoidable:
    "gpt-3.5-turbo-0301" has been retired, and 0613 is the nearest snapshot
    still served.
    """
    config = load_experiment_config(
        ROOT / "configs" / "humaneval-self-collaboration.yaml", ENVIRONMENT
    )
    hyperparameters = config.hyperparameters

    assert config.benchmark.path == "benchmarks/humaneval/tasks"
    assert config.benchmark.dataset is None
    # Pass@1 under greedy decoding is one attempt, which is what §4.1.4 reports.
    assert config.run["n_attempts"] == 1
    assert config.model_name == "openai/gpt-3.5-turbo-0613"

    # "the maximum number of interactions between roles is limited to 4"
    assert hyperparameters["max_rounds"] == 4
    # "we set max tokens to 512 and temperature to 0 for code generation"
    assert config.generation.temperature == 0.0
    assert config.generation.max_tokens == 512
    assert config.generation.source.endswith("generation-humaneval.yaml")

    # The authors' HumanEval entry point, not the repository-shaped session
    # the NL2RepoBench arms run.
    assert hyperparameters["task_shape"] == "humaneval"
    assert hyperparameters["max_steps"] == 10
    # The tool is still pinned to the authors' own repository at one revision.
    assert config.agent_commit == SELF_COLLABORATION_COMMIT_UNDER_TEST



def test_the_humaneval_shape_refuses_a_test_command_it_cannot_run(tmp_path):
    """The Tester on that entry point writes its own cases and runs them.

    There is no command for a configuration to supply, so one written here is
    refused rather than archived as part of a setup and then ignored.
    """
    path = write_config(
        tmp_path,
        agent={
            "import_path": (
                "evaluation_platform.self_collaboration_agent:"
                "SelfCollaborationAgent"
            ),
            "hyperparameters": {
                "task_shape": "humaneval",
                "test_command": "pytest -q",
            },
        },
    )

    with pytest.raises(ConfigurationError) as error:
        load_experiment_config(path, ENVIRONMENT)

    assert "test_command" in str(error.value)
    assert "humaneval" in str(error.value)


def test_the_repository_shape_is_what_a_configuration_gets_by_default():
    """Adding a second entry point must not move the arms already running.

    Every NL2RepoBench configuration predates `task_shape` and states none, so
    the default is the shape they have always run.
    """
    for name in (
            "math-verify-self-collaboration",
            "nl2repobench-self-collaboration",
    ):
        config = load_experiment_config(
            ROOT / "configs" / f"{name}.yaml", ENVIRONMENT
        )
        assert config.hyperparameters["task_shape"] == "repository"


def test_an_unknown_task_shape_is_rejected_rather_than_run_as_the_default(tmp_path):
    path = write_config(
        tmp_path,
        agent={
            "import_path": (
                "evaluation_platform.self_collaboration_agent:"
                "SelfCollaborationAgent"
            ),
            "hyperparameters": {"task_shape": "mbpp"},
        },
    )

    with pytest.raises(ConfigurationError):
        load_experiment_config(path, ENVIRONMENT)
