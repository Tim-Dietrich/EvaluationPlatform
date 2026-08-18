from pathlib import Path

import yaml

from main import build_harbor_command, build_harbor_config, validate_environment


ROOT = Path(__file__).parents[1]


def test_env_example_documents_required_and_optional_settings():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "API_KEY=" in env_example
    assert "PYTHONUTF8=1" in env_example
    assert "MODEL_PROVIDER=openrouter" in env_example
    assert "MODEL=moonshotai/kimi-k2.5" in env_example
    assert "BASE_URL=https://openrouter.ai/api/v1" in env_example
    assert "MODEL_PROVIDER=deepseek" in env_example
    assert "MODEL=deepseek-v4-flash" in env_example
    assert "BASE_URL=https://api.deepseek.com" in env_example


def test_experiment_forwards_only_configured_llm_environment():
    config = yaml.safe_load((ROOT / "experiment.yaml").read_text(encoding="utf-8"))

    assert "job_name" not in config
    assert "model_name" not in config["agents"][0]
    assert config["agents"][0]["env"] == {
        "API_KEY": "${API_KEY}",
        "API_KEY_ENV": "API_KEY",
        "BASE_URL": "${BASE_URL:-https://openrouter.ai/api/v1}",
        "MODEL": "${MODEL:-moonshotai/kimi-k2.5}",
    }


def test_launcher_passes_reporting_identity_to_harbor():
    config = build_harbor_config(
        {"MODEL_PROVIDER": "openrouter", "MODEL": "test/free-model"}
    )
    command = build_harbor_command(
        ROOT / "jobs" / "generated.yaml",
        python_executable=r"C:\project\.venv\Scripts\python.exe",
    )

    assert config["agents"][0]["model_name"] == "openrouter/test/free-model"
    assert command[-2:] == ["--config", str(ROOT / "jobs" / "generated.yaml")]


def test_launcher_rejects_incorrect_deepseek_api_path():
    try:
        validate_environment(
            {
                "MODEL_PROVIDER": "deepseek",
                "BASE_URL": "https://api.deepseek.com/api/v1",
            }
        )
    except ValueError as error:
        assert "https://api.deepseek.com" in str(error)
        assert "/api/v1" in str(error)
    else:
        raise AssertionError("Expected invalid DeepSeek API path to fail")


def test_launcher_accepts_official_deepseek_api_url():
    validate_environment(
        {
            "MODEL_PROVIDER": "deepseek",
            "BASE_URL": "https://api.deepseek.com",
        }
    )


def test_readme_runs_harbor_through_dotenv_launcher():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe main.py" in readme
    assert "--env-file" not in readme


def test_readme_points_viewer_at_current_jobs_directory():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "harbor.exe view .\\jobs --jobs" in readme
