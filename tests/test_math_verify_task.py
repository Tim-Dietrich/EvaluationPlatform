from pathlib import Path

import tomllib
import yaml

TASK_ROOT = Path("harbor_tasks/math-verify")


def _load_experiment() -> dict:
    return yaml.safe_load(Path("experiment.yaml").read_text(encoding="utf-8"))


def test_experiment_uses_the_local_math_verify_task():
    experiment = _load_experiment()

    assert "datasets" not in experiment
    tasks = experiment["tasks"]
    assert tasks == [{"path": "harbor_tasks/math-verify"}]


def test_experiment_collects_generated_workspace_as_an_artifact():
    experiment = _load_experiment()

    assert experiment["artifacts"] == ["/workspace"]


def test_experiment_uses_self_collaboration_agent():
    experiment = _load_experiment()

    assert (
        experiment["agents"][0]["import_path"]
        == "evaluation_platform.self_collaboration_agent:SelfCollaborationAgent"
    )


def test_task_instruction_is_self_contained():
    instruction = (TASK_ROOT / "instruction.md").read_text(encoding="utf-8")

    assert "Please create a Python project called Math-Verify" in instruction


def test_tester_image_points_at_the_public_ghcr_mirror_not_the_private_one():
    compose = (TASK_ROOT / "environment/docker-compose.yaml").read_text(
        encoding="utf-8"
    )

    assert (
        "ghcr.io/multimodal-art-projection/nl2repobench/math-verify:1.0" in compose
    )
    assert "us-docker.pkg.dev" not in compose


def test_task_declares_environment_and_timeouts():
    config = tomllib.loads((TASK_ROOT / "task.toml").read_text(encoding="utf-8"))

    assert config["metadata"]["source"] == "NL2RepoBench"
    assert config["agent"]["timeout_sec"] > 0
    assert config["verifier"]["timeout_sec"] > 0


def test_tester_reports_fractional_score_out_of_192():
    tester_run = (TASK_ROOT / "environment/tester_run.sh").read_text(
        encoding="utf-8"
    )

    assert "tests/test_all.py" in tester_run
    assert "parse_pytest.py" in tester_run
