import json
from pathlib import Path

import tomllib

TASK_ROOT = Path("harbor_tasks/math-verify")


def test_task_uses_nl2repobench_instruction_and_test_image():
    benchmark_root = Path("benchmarks/NL2RepoBench")
    source_instruction = (
            benchmark_root / "test_files/math-verify/start.md"
    ).read_text(encoding="utf-8")
    instruction = (TASK_ROOT / "instruction.md").read_text(encoding="utf-8")
    assert "Please create a Python project called Math-Verify" in instruction
    assert instruction in source_instruction

    dockerfile = (TASK_ROOT / "environment/Dockerfile").read_text(encoding="utf-8")
    assert "nl2repobench/math-verify:1.0" in dockerfile
    assert "chmod -R 700 /benchmark" in dockerfile

    config = tomllib.loads((TASK_ROOT / "task.toml").read_text(encoding="utf-8"))
    assert config["agent"]["user"] == "agent"
    assert config["verifier"]["user"] == "root"


def test_verifier_preserves_original_commands_and_reports_fractional_score():
    commands = json.loads(
        Path(
            "benchmarks/NL2RepoBench/test_files/math-verify/test_commands.json"
        ).read_text(encoding="utf-8")
    )
    verifier = (TASK_ROOT / "tests/test.sh").read_text(encoding="utf-8")

    for command in commands:
        assert command in verifier
    assert "reward.txt" in verifier
    assert "evaluator-output.json" in verifier


def test_task_collects_workspace_and_declares_timeouts():
    config = tomllib.loads((TASK_ROOT / "task.toml").read_text(encoding="utf-8"))

    assert config["source"] == "NL2RepoBench"
    assert "/app" in config["artifacts"]
    assert config["agent"]["timeout_sec"] > 0
    assert config["verifier"]["timeout_sec"] > 0


def test_experiment_supplies_complete_benchmark_context():
    experiment = Path("experiment.yaml").read_text(encoding="utf-8")

    assert "evaluation_platform.self_collaboration_agent:SelfCollaborationAgent" in experiment
    assert "benchmarks/NL2RepoBench/test_files/math-verify/start.md" in experiment
