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

Preparation is sized for whole benchmarks rather than single tasks. A
NL2RepoBench tester image is around two gigabytes and every one of the 104
tasks has its own, so the images are pulled concurrently: pulling is bound by
the network, not by this machine, and doing it one image at a time would cost
more wall-clock time than the run it precedes. Preparation also reads what
the selected tasks ask of the machine and weighs it against what Docker
actually has, so a concurrency setting the machine cannot hold is reported
before the run starts rather than discovered hours into it.
"""

import asyncio
import subprocess
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# How many images to pull at once. Pulling is bound by the network and by
# Docker's own extraction, not by this process, so a handful of concurrent
# pulls turns the fixed cost of a whole benchmark from serial into parallel
# without saturating either.
DEFAULT_PULL_CONCURRENCY = 4


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
class TrialDemand:
    """What one trial of the selected tasks asks of the machine.

    Every Harbor task declares its own `cpus` and `memory_mb`. Running tasks
    concurrently multiplies that demand, so the largest task in the selection
    is what a concurrency setting has to be judged against.
    """

    cpus: int | None = None
    memory_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"cpus": self.cpus, "memory_mb": self.memory_mb}


@dataclass(frozen=True)
class MachineCapacity:
    """What Docker itself has to spend, as Docker reports it.

    On Windows and macOS this is the virtual machine's allocation rather than
    the host's hardware, which is exactly the number that matters: containers
    never see more than the VM was given.
    """

    cpus: int | None = None
    memory_mb: int | None = None


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
    demand: TrialDemand = TrialDemand()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "resolved_ref": self.resolved_ref,
            "n_tasks": len(self.task_names),
            "task_names": list(self.task_names),
            "trial_demand": self.demand.to_dict(),
            "images": [image.to_dict() for image in self.images],
        }


def prepare(
        settings: BenchmarkSettings,
        log: Callable[[str], None] = print,
        pull_concurrency: int = DEFAULT_PULL_CONCURRENCY,
) -> BenchmarkPreparation:
    """Resolve the selected tasks and make their images locally available."""
    return asyncio.run(prepare_async(settings, log, pull_concurrency))


async def prepare_async(
        settings: BenchmarkSettings,
        log: Callable[[str], None] = print,
        pull_concurrency: int = DEFAULT_PULL_CONCURRENCY,
) -> BenchmarkPreparation:
    resolved_ref, tasks = await _resolve_tasks(settings)
    log(
        f"Benchmark {settings.dataset} resolved to {resolved_ref or 'latest'} "
        f"with {len(tasks)} task(s) selected."
    )

    images = await _mirror_images(tasks, settings.image_mirror, log, pull_concurrency)

    return BenchmarkPreparation(
        dataset=settings.dataset,
        resolved_ref=resolved_ref,
        task_names=[name for name, _ in tasks],
        images=images,
        demand=_demand_of(task_dir for _, task_dir in tasks),
    )


async def _mirror_images(
        tasks: list[tuple[str, Path]],
        rules: list[ImageMirrorRule],
        log: Callable[[str], None],
        pull_concurrency: int,
) -> list[MirroredImage]:
    """Make every mirrored image the selected tasks name locally available.

    The images are independent of one another, so they are pulled together
    rather than in turn: at benchmark scale this is the difference between
    minutes and hours before the first trial starts. Two tasks naming the same
    image resolve it once.
    """
    wanted: dict[str, tuple[str, str]] = {}
    for task_name, task_dir in tasks:
        for expected in _images_named_by(task_dir):
            source = _mirror_source(expected, rules)
            if source is not None:
                wanted.setdefault(expected, (task_name, source))

    if not wanted:
        return []

    at_once = max(1, min(len(wanted), pull_concurrency))
    limit = asyncio.Semaphore(at_once)
    log(
        f"Preparing {len(wanted)} tester image(s)"
        + (f", {at_once} at a time" if at_once > 1 else "")
        + " ..."
    )

    async def make_available(expected: str, task_name: str, source: str) -> MirroredImage:
        async with limit:
            return await asyncio.to_thread(
                _make_available, task_name, expected, source, log
            )

    return list(
        await asyncio.gather(
            *(
                make_available(expected, task_name, source)
                for expected, (task_name, source) in wanted.items()
            )
        )
    )


def docker_capacity() -> MachineCapacity:
    """What Docker reports it has, or an empty capacity if it cannot say.

    Docker is the only authority worth asking here: on Windows and macOS it
    runs in a virtual machine whose allocation is usually well below the
    host's hardware, and containers never see more than the VM was given.
    """
    result = _docker(
        ["info", "--format", "{{.NCPU}} {{.MemTotal}}"],
        check=False,
    )
    if result.returncode != 0:
        return MachineCapacity()
    fields = (result.stdout or "").split()
    if len(fields) != 2:
        return MachineCapacity()
    try:
        cpus, memory_bytes = int(fields[0]), int(fields[1])
    except ValueError:
        return MachineCapacity()
    return MachineCapacity(
        cpus=cpus or None,
        memory_mb=memory_bytes // (1024 * 1024) or None,
    )


def concurrency_advice(
        demand: TrialDemand,
        capacity: MachineCapacity,
        n_concurrent_trials: int,
) -> list[str]:
    """How the requested concurrency compares with what the machine has.

    Memory is the ceiling worth reporting. Harbor passes a task's `cpus` to
    Docker as a limit, so oversubscribing processors makes trials share and
    slow down; a trial that reaches its memory limit is killed outright and
    the task is scored as an error. A run of a hundred tasks is long enough
    that finding this out from the results is expensive, so it is reported
    before the first container starts. The advice is never a refusal: the
    declared figures are a task author's headroom, not a measurement, and the
    person launching the run knows their machine.
    """
    if demand.memory_mb is None or capacity.memory_mb is None:
        return []

    fits = capacity.memory_mb // demand.memory_mb
    summary = (
        f"Each trial may use up to {demand.memory_mb} MB"
        + (f" and {demand.cpus} CPU(s)" if demand.cpus else "")
        + f"; Docker has {capacity.memory_mb} MB"
        + (f" and {capacity.cpus} CPU(s)" if capacity.cpus else "")
        + f" for {n_concurrent_trials} concurrent trial(s)."
    )
    if fits >= n_concurrent_trials:
        return [summary]

    covered = (
        "Docker has less memory than a single trial's ceiling"
        if fits == 0
        else f"Docker's memory covers about {fits} trial(s) at that ceiling"
    )
    return [
        summary,
        f"{covered}, so the run can overcommit it. A trial that reaches the "
        "ceiling is killed and scored as an error, and one that merely "
        "crowds the machine runs slower. Give Docker more memory, lower "
        "'run.n_concurrent_trials', or set a ceiling this machine can hold "
        "with 'run.environment.override_memory_mb', which changes what the "
        "benchmark measures and so is a choice to record deliberately.",
    ]


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

    image = _task_environment(task_dir).get("docker_image")
    if isinstance(image, str):
        images.append(image)

    return images


def _task_environment(task_dir: Path) -> dict[str, Any]:
    """The `[environment]` table of a task, or an empty one if it has none."""
    task_path = task_dir / "task.toml"
    if not task_path.exists():
        return {}
    task = tomllib.loads(task_path.read_text(encoding="utf-8"))
    environment = task.get("environment")
    return environment if isinstance(environment, dict) else {}


def _demand_of(task_dirs: Iterable[Path]) -> TrialDemand:
    """What a single trial of the largest selected task asks for.

    Concurrency has to hold for every task in the selection, so the maximum
    rather than the average is the figure to plan against.
    """
    cpus: list[int] = []
    memory: list[int] = []
    for task_dir in task_dirs:
        environment = _task_environment(task_dir)
        if isinstance(environment.get("cpus"), int):
            cpus.append(environment["cpus"])
        if isinstance(environment.get("memory_mb"), int):
            memory.append(environment["memory_mb"])
    return TrialDemand(
        cpus=max(cpus) if cpus else None,
        memory_mb=max(memory) if memory else None,
    )


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

    # Concurrent pulls would interleave their progress bars into noise, so the
    # output is captured and each image reports once, when it lands.
    _docker(["pull", source])
    _docker(["tag", source, expected])
    log(f"Pulled {source} for {task}.")
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
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            capture_output=True,
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
