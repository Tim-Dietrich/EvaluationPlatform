from pathlib import Path
from typing import Any

import pytest
import yaml

from main import archive_setup, build_harbor_command
from evaluation_platform.experiment_config import (
    DEFAULT_CONFIG_PATH,
    ConfigurationError,
    load_experiment_config,
)


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
    assert config.hyperparameters["temperature"] == 0.0


def test_agent_environment_forwards_only_the_model_backend(tmp_path):
    config = load_experiment_config(write_config(tmp_path), ENVIRONMENT)

    environment = config.to_harbor_config("job")["agents"][0]["env"]

    assert environment == {
        "API_KEY": "${API_KEY}",
        "API_KEY_ENV": "API_KEY",
        "BASE_URL": "https://openrouter.ai/api/v1",
        "MODEL": "test/free-model",
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
