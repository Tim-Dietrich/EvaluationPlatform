from pathlib import Path

import yaml


def _load_experiment() -> dict:
    return yaml.safe_load(Path("experiment.yaml").read_text(encoding="utf-8"))


def test_experiment_uses_nl2repobench_registry_dataset():
    experiment = _load_experiment()

    assert "tasks" not in experiment
    datasets = experiment["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "nl2repobench/nl2repobench"


def test_experiment_pins_dataset_to_a_content_digest():
    experiment = _load_experiment()
    ref = experiment["datasets"][0]["ref"]

    assert ref.startswith("sha256:")
    assert len(ref) == len("sha256:") + 64


def test_experiment_restricts_dataset_to_math_verify_task():
    experiment = _load_experiment()

    assert experiment["datasets"][0]["task_names"] == ["nl2repobench/math-verify"]


def test_experiment_collects_generated_workspace_as_an_artifact():
    experiment = _load_experiment()

    assert experiment["artifacts"] == ["/workspace"]


def test_experiment_uses_self_collaboration_agent():
    experiment = _load_experiment()

    assert (
        experiment["agents"][0]["import_path"]
        == "evaluation_platform.self_collaboration_agent:SelfCollaborationAgent"
    )
