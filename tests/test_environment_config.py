from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_env_example_documents_required_and_optional_settings():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "API_KEY=" in env_example
    assert "PYTHONUTF8=1" in env_example
    assert "MODEL=moonshotai/kimi-k2.5" in env_example
    assert "BASE_URL=https://openrouter.ai/api/v1" in env_example


def test_experiment_forwards_only_configured_llm_environment():
    config = yaml.safe_load((ROOT / "experiment.yaml").read_text(encoding="utf-8"))

    assert config["agents"][0]["env"] == {
        "API_KEY": "${API_KEY}",
        "API_KEY_ENV": "API_KEY",
        "BASE_URL": "${BASE_URL:-https://openrouter.ai/api/v1}",
        "MODEL": "${MODEL:-moonshotai/kimi-k2.5}",
    }


def test_readme_runs_harbor_under_dotenv():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "-m dotenv run --" in readme
    assert "--env-file" not in readme
