import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv

from evaluation_platform import benchmark as benchmark_module
from evaluation_platform.benchmark import (
    BenchmarkError,
    BenchmarkPreparation,
    concurrency_advice,
    docker_capacity,
)
from evaluation_platform.experiment_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_N_CONCURRENT_TRIALS,
    ConfigurationError,
    ExperimentConfig,
    load_experiment_config,
)


ROOT = Path(__file__).parent
SNAPSHOT_FILENAME = "experiment-config.yaml"
BENCHMARK_FILENAME = "benchmark.json"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Harbor experiment from a setup configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Experiment setup configuration to run "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Name of the job directory (default: the launch timestamp).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Continue an interrupted job directory instead of starting a new "
            "run. Trials that already have a result are kept and the rest are "
            "executed; cancelled trials are discarded and retried."
        ),
    )
    return parser.parse_args(argv)


def harbor_executable(python_executable: str = sys.executable) -> str:
    executable_name = "harbor.exe" if os.name == "nt" else "harbor"
    return str(Path(python_executable).with_name(executable_name))


def build_harbor_command(
        config_path: Path,
        python_executable: str = sys.executable,
) -> list[str]:
    return [
        harbor_executable(python_executable),
        "run",
        "--config",
        str(config_path),
    ]


def build_resume_command(
        job_dir: Path,
        python_executable: str = sys.executable,
) -> list[str]:
    return [
        harbor_executable(python_executable),
        "job",
        "resume",
        "--job-path",
        str(job_dir),
    ]


def archive_setup(experiment: ExperimentConfig, job_dir: Path) -> Path:
    """Record the resolved setup next to the job before the run starts.

    Writing the snapshot up front keeps the setup recoverable even when a run
    crashes. Harbor treats a job directory as resumable only once it has
    written its own `config.json`, so this file does not disturb the run.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = job_dir / SNAPSHOT_FILENAME
    snapshot_path.write_text(
        yaml.safe_dump(experiment.to_snapshot(), sort_keys=False),
        encoding="utf-8",
    )
    return snapshot_path


def archive_benchmark(preparation: BenchmarkPreparation, job_dir: Path) -> Path:
    """Record which tasks and images the run resolved the benchmark to."""
    job_dir.mkdir(parents=True, exist_ok=True)
    record_path = job_dir / BENCHMARK_FILENAME
    record_path.write_text(
        json.dumps(preparation.to_dict(), indent=2),
        encoding="utf-8",
    )
    return record_path


def report_capacity(
        experiment: ExperimentConfig,
        preparation: BenchmarkPreparation,
) -> None:
    """Say how the requested concurrency compares with what Docker has.

    A benchmark run is long enough that a concurrency setting the machine
    cannot hold is expensive to discover from its results. This reports the
    comparison and launches anyway: the declared figures are what a task
    author asked for, not a measurement of what the task uses.
    """
    for line in concurrency_advice(
            preparation.demand,
            docker_capacity(),
            experiment.run.get(
                "n_concurrent_trials", DEFAULT_N_CONCURRENT_TRIALS
            ),
    ):
        print(line)


def resume(job_dir: Path) -> int:
    """Continue an interrupted job from the setup archived beside it.

    Harbor resumes from its own `config.json`, keeping every trial that
    already has a result and running the rest, so a run that dies at task 80
    of 104 costs the remaining tasks rather than all of them. The images the
    remaining tasks need are made available again first: a resume can happen
    days later, on a machine whose Docker images have since been pruned.
    """
    if not (job_dir / "config.json").exists():
        print(
            f"{job_dir} is not a resumable job directory: Harbor writes "
            "config.json once a run starts, and this one has none. A run that "
            "failed before that point has to be launched again.",
            file=sys.stderr,
        )
        return 2

    snapshot = job_dir / SNAPSHOT_FILENAME
    if snapshot.exists():
        try:
            experiment = load_experiment_config(snapshot, os.environ)
        except ConfigurationError as error:
            print(f"Configuration error in {snapshot}: {error}", file=sys.stderr)
            return 2
        if experiment.benchmark is not None:
            try:
                preparation = benchmark_module.prepare(experiment.benchmark)
            except BenchmarkError as error:
                print(f"Benchmark error: {error}", file=sys.stderr)
                return 2
            report_capacity(experiment, preparation)

    return subprocess.run(build_resume_command(job_dir), check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    arguments = parse_arguments(argv)

    if arguments.resume is not None:
        if arguments.config is not None or arguments.job_name is not None:
            print(
                "--resume continues a job from the setup archived beside it, "
                "so it cannot be combined with --config or --job-name.",
                file=sys.stderr,
            )
            return 2
        return resume(arguments.resume)

    try:
        experiment = load_experiment_config(
            arguments.config or ROOT / DEFAULT_CONFIG_PATH, os.environ
        )
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    job_name = arguments.job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    jobs_dir = ROOT / experiment.run.get("jobs_dir", "jobs")
    job_dir = jobs_dir / job_name

    if experiment.benchmark is not None:
        try:
            preparation = benchmark_module.prepare(experiment.benchmark)
        except BenchmarkError as error:
            print(f"Benchmark error: {error}", file=sys.stderr)
            return 2
        # Every trial of the run, and its archived setup, stay on the version
        # the benchmark resolved to at launch.
        experiment = experiment.with_pinned_benchmark(preparation.resolved_ref)
        archive_benchmark(preparation, job_dir)
        report_capacity(experiment, preparation)

    harbor_config: dict[str, Any] = experiment.to_harbor_config(job_name)
    archive_setup(experiment, job_dir)

    config_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".yaml",
                prefix="experiment-",
                dir=jobs_dir,
                delete=False,
        ) as config_file:
            yaml.safe_dump(harbor_config, config_file, sort_keys=False)
            config_path = Path(config_file.name)
        result = subprocess.run(build_harbor_command(config_path), check=False)
        return result.returncode
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
