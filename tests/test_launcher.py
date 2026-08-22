"""Launching and continuing a run.

A benchmark-scale run is long enough that it will sometimes be interrupted —
a lost connection, a laptop closing, an exhausted credential. Harbor keeps
every trial that already has a result, so continuing costs the tasks that are
left rather than all of them. These tests cover the launcher's side of that:
finding the archived setup, refusing what cannot be resumed, and handing
Harbor the job directory.
"""

import json
from pathlib import Path

import pytest
import yaml

import main
from main import (
    BENCHMARK_FILENAME,
    SNAPSHOT_FILENAME,
    build_resume_command,
    parse_arguments,
)


ROOT = Path(__file__).parents[1]


def write_job(directory: Path, with_config: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if with_config:
        (directory / "config.json").write_text(
            json.dumps({"job_name": directory.name}), encoding="utf-8"
        )
    return directory


def test_resume_hands_harbor_the_job_directory():
    command = build_resume_command(Path("jobs/2026-01-01__00-00-00"))

    assert command[1:] == [
        "job",
        "resume",
        "--job-path",
        str(Path("jobs/2026-01-01__00-00-00")),
    ]
    assert command[0].endswith(("harbor", "harbor.exe"))


def test_resume_runs_harbor_from_the_project_environment():
    """The same executable the launcher uses, so both see the pinned Harbor."""
    assert Path(build_resume_command(Path("jobs/x"))[0]).parent == Path(
        main.build_harbor_command(Path("config.yaml"))[0]
    ).parent


def test_a_job_harbor_never_started_is_not_resumable(tmp_path, capsys):
    """Harbor resumes from its own `config.json` and there is nothing else.

    A run that failed during preparation leaves this project's snapshot behind
    but no Harbor record, and saying so is more useful than a Harbor error
    about a missing file.
    """
    job_dir = write_job(tmp_path / "job", with_config=False)
    (job_dir / SNAPSHOT_FILENAME).write_text("name: x\n", encoding="utf-8")

    assert main.resume(job_dir) == 2
    assert "config.json" in capsys.readouterr().err


def test_resume_makes_the_images_available_again_before_continuing(
        tmp_path, monkeypatch
):
    """A resume can happen days later, after Docker images have been pruned."""
    job_dir = write_job(tmp_path / "job")
    config = yaml.safe_load(
        (ROOT / "configs" / "nl2repobench-self-collaboration.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["model"] = {
        "provider": "openrouter",
        "name": "test/free-model",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "API_KEY",
    }
    (job_dir / SNAPSHOT_FILENAME).write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    prepared = []
    commands = []
    monkeypatch.setenv("API_KEY", "test-credential")
    monkeypatch.setattr(
        main.benchmark_module,
        "prepare",
        lambda settings: prepared.append(settings)
        or main.BenchmarkPreparation(settings.dataset, settings.ref, [], []),
    )
    # Every subprocess the resume makes, so the order of the two is visible.
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda command, **kwargs: commands.append(list(command))
        or type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    assert main.resume(job_dir) == 0
    assert [settings.dataset for settings in prepared] == [
        "nl2repobench/nl2repobench"
    ]
    assert commands[-1][1:3] == ["job", "resume"]


def test_resume_cannot_be_combined_with_a_fresh_setup(tmp_path, capsys):
    """Resuming reads the setup that ran; it is not a chance to change it."""
    exit_code = main.main(
        ["--resume", str(tmp_path), "--config", "configs/whatever.yaml"]
    )

    assert exit_code == 2
    assert "--config" in capsys.readouterr().err


def test_a_run_without_resume_still_defaults_to_the_shipped_configuration():
    assert parse_arguments([]).config is None
    assert parse_arguments([]).resume is None


def test_the_readme_documents_continuing_an_interrupted_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "--resume" in readme


def test_the_archived_names_are_the_ones_resume_looks_for():
    assert SNAPSHOT_FILENAME == "experiment-config.yaml"
    assert BENCHMARK_FILENAME == "benchmark.json"
