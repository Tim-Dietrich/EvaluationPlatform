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
from evaluation_platform.benchmark import BenchmarkError, BenchmarkPreparation
from evaluation_platform.experiment_config import (
    DEFAULT_CONFIG_PATH,
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
        default=ROOT / DEFAULT_CONFIG_PATH,
        help="Experiment setup configuration to run (default: %(default)s).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Name of the job directory (default: the launch timestamp).",
    )
    return parser.parse_args(argv)


def build_harbor_command(
        config_path: Path,
        python_executable: str = sys.executable,
) -> list[str]:
    executable_name = "harbor.exe" if os.name == "nt" else "harbor"
    harbor_executable = str(Path(python_executable).with_name(executable_name))
    return [
        harbor_executable,
        "run",
        "--config",
        str(config_path),
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


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    arguments = parse_arguments(argv)
    try:
        experiment = load_experiment_config(arguments.config, os.environ)
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
