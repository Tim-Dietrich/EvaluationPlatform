"""Benchmark integration.

Harbor's registry already carries whole benchmarks as digest-pinned datasets,
so a benchmark enters this repository as a dependency rather than as
checked-in task files: an experiment configuration names the dataset and which
of its tasks to run, and Harbor downloads and pins the rest.

What that leaves is one gap, and it is about images rather than tasks.
NL2RepoBench's published tasks name their tester images on a private mirror
that needs credentials this project does not have, while the identical images
are public elsewhere. This module closes that gap without touching the
benchmark: it resolves exactly the tasks a run will execute, reads the images
they name, and makes each one available locally under the name the task
expects, pulled from its public home. Docker Compose uses a locally present
image without contacting a registry, so the benchmark runs unmodified.
"""

import asyncio
import subprocess
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class BenchmarkError(RuntimeError):
    """A benchmark could not be resolved or made runnable."""


@dataclass(frozen=True)
class ImageMirrorRule:
    """Where to obtain an image a task names but cannot pull itself.

    `expects` is the reference prefix as published in the benchmark;
    `pull_from` is the prefix of the same images on a reachable host.
    """

    expects: str
    pull_from: str

    def source_for(self, image: str) -> str | None:
        """The reachable name for `image`, or None if this rule doesn't apply."""
        if not image.startswith(self.expects):
            return None
        return self.pull_from + image[len(self.expects):]


@dataclass(frozen=True)
class BenchmarkSettings:
    """Which benchmark a run uses, and which of its tasks."""

    dataset: str
    ref: str | None = None
    task_names: list[str] = field(default_factory=list)
    exclude_task_names: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    image_mirror: list[ImageMirrorRule] = field(default_factory=list)

    def to_harbor_dataset(self) -> dict[str, Any]:
        dataset: dict[str, Any] = {"name": self.dataset}
        if self.ref is not None:
            dataset["ref"] = self.ref
        if self.task_names:
            dataset["task_names"] = list(self.task_names)
        if self.exclude_task_names:
            dataset["exclude_task_names"] = list(self.exclude_task_names)
        if self.n_tasks is not None:
            dataset["n_tasks"] = self.n_tasks
        return dataset

    def to_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"dataset": self.dataset}
        if self.ref is not None:
            snapshot["ref"] = self.ref
        if self.task_names:
            snapshot["task_names"] = list(self.task_names)
        if self.exclude_task_names:
            snapshot["exclude_task_names"] = list(self.exclude_task_names)
        if self.n_tasks is not None:
            snapshot["n_tasks"] = self.n_tasks
        if self.image_mirror:
            snapshot["image_mirror"] = [
                {"expects": rule.expects, "pull_from": rule.pull_from}
                for rule in self.image_mirror
            ]
        return snapshot


@dataclass(frozen=True)
class MirroredImage:
    """One image made available under the name a task expects."""

    task: str
    expected: str
    pulled_from: str
    digest: str | None
    already_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "expected": self.expected,
            "pulled_from": self.pulled_from,
            "digest": self.digest,
            "already_present": self.already_present,
        }


@dataclass(frozen=True)
class BenchmarkPreparation:
    """What a run resolved the benchmark to, and the images it will use."""

    dataset: str
    resolved_ref: str | None
    task_names: list[str]
    images: list[MirroredImage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "resolved_ref": self.resolved_ref,
            "n_tasks": len(self.task_names),
            "task_names": list(self.task_names),
            "images": [image.to_dict() for image in self.images],
        }


def prepare(
        settings: BenchmarkSettings,
        log: Callable[[str], None] = print,
) -> BenchmarkPreparation:
    """Resolve the selected tasks and make their images locally available."""
    return asyncio.run(prepare_async(settings, log))


async def prepare_async(
        settings: BenchmarkSettings,
        log: Callable[[str], None] = print,
) -> BenchmarkPreparation:
    resolved_ref, tasks = await _resolve_tasks(settings)
    log(
        f"Benchmark {settings.dataset} resolved to {resolved_ref or 'latest'} "
        f"with {len(tasks)} task(s) selected."
    )

    images: list[MirroredImage] = []
    if settings.image_mirror:
        for task_name, task_dir in tasks:
            for expected in _images_named_by(task_dir):
                source = _mirror_source(expected, settings.image_mirror)
                if source is not None:
                    images.append(_make_available(task_name, expected, source, log))

    return BenchmarkPreparation(
        dataset=settings.dataset,
        resolved_ref=resolved_ref,
        task_names=[name for name, _ in tasks],
        images=images,
    )


async def _resolve_tasks(
        settings: BenchmarkSettings,
) -> tuple[str | None, list[tuple[str, Path]]]:
    """Select tasks exactly as the job will, and materialize their packages.

    Harbor's own dataset configuration does the selecting, so a run and its
    preparation can never disagree about which tasks are in scope. The
    packages land in the cache the job reads from, so nothing is downloaded
    twice.
    """
    from harbor.models.job.config import DatasetConfig
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient

    dataset = DatasetConfig(**settings.to_harbor_dataset())
    if not dataset.is_package():
        raise BenchmarkError(
            f"Benchmark {settings.dataset!r} is not a registry dataset. Use an "
            "'org/name' dataset published to Harbor's registry."
        )

    try:
        task_configs = await dataset.get_task_configs()
    except ValueError as error:
        raise BenchmarkError(str(error)) from error

    task_ids = []
    for task_config in task_configs:
        assert task_config.name is not None
        org, _, name = task_config.name.partition("/")
        task_ids.append(PackageTaskId(org=org, name=name, ref=task_config.ref))

    downloads = await TaskClient().download_tasks(task_ids)
    names = [task_config.name or "" for task_config in task_configs]
    # `get_task_configs` resolves a floating ref to the dataset's digest.
    return dataset.ref, list(zip(names, downloads.paths, strict=True))


def _images_named_by(task_dir: Path) -> list[str]:
    """Every image reference a task names, in its compose file and task.toml."""
    images: list[str] = []

    compose_path = task_dir / "environment" / "docker-compose.yaml"
    if compose_path.exists():
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
        for service in (compose.get("services") or {}).values():
            image = (service or {}).get("image")
            if isinstance(image, str):
                images.append(image)

    task_path = task_dir / "task.toml"
    if task_path.exists():
        task = tomllib.loads(task_path.read_text(encoding="utf-8"))
        image = task.get("environment", {}).get("docker_image")
        if isinstance(image, str):
            images.append(image)

    return images


def _mirror_source(image: str, rules: Iterable[ImageMirrorRule]) -> str | None:
    for rule in rules:
        source = rule.source_for(image)
        if source is not None:
            return source
    return None


def _make_available(
        task: str,
        expected: str,
        source: str,
        log: Callable[[str], None],
) -> MirroredImage:
    if _image_exists(expected):
        return MirroredImage(
            task=task,
            expected=expected,
            pulled_from=source,
            digest=_image_digest(expected),
            already_present=True,
        )

    log(f"Pulling {source} for {task} ...")
    _docker(["pull", source], stream=True)
    _docker(["tag", source, expected])
    return MirroredImage(
        task=task,
        expected=expected,
        pulled_from=source,
        digest=_image_digest(source),
        already_present=False,
    )


def _image_exists(image: str) -> bool:
    return _docker(["image", "inspect", image], check=False).returncode == 0


def _image_digest(image: str) -> str | None:
    result = _docker(
        ["image", "inspect", "--format", "{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}", image],
        check=False,
    )
    digest = (result.stdout or "").strip()
    return digest or None


def _docker(
        arguments: list[str],
        check: bool = True,
        stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            capture_output=not stream,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise BenchmarkError(
            "Docker is required to make benchmark images available locally, "
            "but the 'docker' command was not found."
        ) from error

    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise BenchmarkError(
            f"'docker {' '.join(arguments)}' failed"
            + (f": {detail}" if detail else ".")
        )
    return result
