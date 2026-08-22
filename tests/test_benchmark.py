import json
import time
from pathlib import Path

import pytest
import yaml

from evaluation_platform import benchmark as benchmark_module
from evaluation_platform.benchmark import (
    BenchmarkError,
    BenchmarkPreparation,
    BenchmarkSettings,
    ImageMirrorRule,
    MachineCapacity,
    MirroredImage,
    TrialDemand,
    _demand_of,
    _images_named_by,
    _make_available,
    _mirror_images,
    _mirror_source,
    concurrency_advice,
)
from evaluation_platform.experiment_config import load_experiment_config

from main import archive_benchmark


ROOT = Path(__file__).parents[1]
ENVIRONMENT = {"API_KEY": "test-credential"}
NL2REPOBENCH_PRIVATE_PREFIX = (
    "us-docker.pkg.dev/cybertron-gcp-island-test-0rxn/applejax/walker_cheng/"
    "nl2repobench/"
)


def load_shipped(name: str):
    return load_experiment_config(ROOT / "configs" / f"{name}.yaml", ENVIRONMENT)


def write_task(directory: Path, image: str, docker_image: str) -> Path:
    (directory / "environment").mkdir(parents=True)
    (directory / "environment" / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {"working_dir": "/workspace"},
                    "tester": {"image": image},
                }
            }
        ),
        encoding="utf-8",
    )
    (directory / "task.toml").write_text(
        f'[environment]\ndocker_image = "{docker_image}"\n', encoding="utf-8"
    )
    return directory


def test_shipped_configurations_take_the_benchmark_as_a_dependency():
    for name in (
            "math-verify-self-collaboration",
            "nl2repobench-self-collaboration",
    ):
        config = load_shipped(name)

        # No checked-in task files: the benchmark is resolved from Harbor's
        # registry, pinned to one digest.
        assert config.task_path is None
        assert config.benchmark.dataset == "nl2repobench/nl2repobench"
        assert config.benchmark.ref.startswith("sha256:")


def test_full_benchmark_configuration_selects_every_task():
    config = load_shipped("nl2repobench-self-collaboration")

    assert config.benchmark.task_names == []
    assert config.benchmark.n_tasks is None
    assert config.benchmark.to_harbor_dataset() == {
        "name": "nl2repobench/nl2repobench",
        "ref": config.benchmark.ref,
    }


def test_single_task_configuration_narrows_the_same_benchmark():
    single = load_shipped("math-verify-self-collaboration")
    full = load_shipped("nl2repobench-self-collaboration")

    # Comparability rests on both runs resolving the same benchmark version.
    assert single.benchmark.ref == full.benchmark.ref
    assert single.benchmark.task_names == ["nl2repobench/math-verify"]


def test_mirror_rule_maps_the_private_reference_to_its_public_home():
    config = load_shipped("nl2repobench-self-collaboration")
    (rule,) = config.benchmark.image_mirror

    assert rule.source_for(f"{NL2REPOBENCH_PRIVATE_PREFIX}math-verify:1.0") == (
        "ghcr.io/multimodal-art-projection/nl2repobench/math-verify:1.0"
    )
    assert rule.source_for("nikolaik/python-nodejs:python3.12-nodejs22") is None


def test_images_named_by_a_task_cover_compose_and_task_toml(tmp_path):
    write_task(
        tmp_path,
        image=f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
        docker_image="nikolaik/python-nodejs:python3.12-nodejs22",
    )

    assert _images_named_by(tmp_path) == [
        f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
        "nikolaik/python-nodejs:python3.12-nodejs22",
    ]


def test_only_unreachable_images_are_mirrored(tmp_path):
    rules = [
        ImageMirrorRule(
            expects=NL2REPOBENCH_PRIVATE_PREFIX,
            pull_from="ghcr.io/multimodal-art-projection/nl2repobench/",
        )
    ]
    write_task(
        tmp_path,
        image=f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
        docker_image="nikolaik/python-nodejs:python3.12-nodejs22",
    )

    sources = [_mirror_source(image, rules) for image in _images_named_by(tmp_path)]

    assert sources == [
        "ghcr.io/multimodal-art-projection/nl2repobench/box:1.0",
        None,
    ]


def test_a_missing_image_is_pulled_and_tagged_under_the_expected_name(monkeypatch):
    calls = []

    def fake_docker(arguments, check=True):
        calls.append(arguments)
        if arguments[:2] == ["image", "inspect"] and "--format" not in arguments:
            raise AssertionError("unreachable: existence is stubbed separately")
        return type("Result", (), {"returncode": 0, "stdout": "repo@sha256:abc"})()

    monkeypatch.setattr(benchmark_module, "_image_exists", lambda image: False)
    monkeypatch.setattr(benchmark_module, "_docker", fake_docker)

    mirrored = _make_available(
        "nl2repobench/box",
        f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
        "ghcr.io/multimodal-art-projection/nl2repobench/box:1.0",
        log=lambda message: None,
    )

    assert calls[0] == ["pull", "ghcr.io/multimodal-art-projection/nl2repobench/box:1.0"]
    assert calls[1] == [
        "tag",
        "ghcr.io/multimodal-art-projection/nl2repobench/box:1.0",
        f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
    ]
    assert mirrored.already_present is False
    assert mirrored.digest == "repo@sha256:abc"


def test_an_image_already_present_is_not_pulled_again(monkeypatch):
    monkeypatch.setattr(benchmark_module, "_image_exists", lambda image: True)
    monkeypatch.setattr(
        benchmark_module,
        "_docker",
        lambda arguments, check=True: (_ for _ in ()).throw(
            AssertionError(f"unexpected docker call: {arguments}")
        )
        if arguments[0] == "pull"
        else type("Result", (), {"returncode": 0, "stdout": "repo@sha256:abc"})(),
    )

    mirrored = _make_available(
        "nl2repobench/box",
        f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
        "ghcr.io/multimodal-art-projection/nl2repobench/box:1.0",
        log=lambda message: None,
    )

    assert mirrored.already_present is True


def test_a_missing_docker_command_is_reported_as_a_benchmark_error(monkeypatch):
    def missing_docker(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(benchmark_module.subprocess, "run", missing_docker)

    with pytest.raises(BenchmarkError) as error:
        benchmark_module._docker(["pull", "image"])

    assert "Docker" in str(error.value)


def test_a_floating_reference_is_pinned_to_what_it_resolved_to():
    config = load_shipped("nl2repobench-self-collaboration")
    floating = config.with_pinned_benchmark(None)

    pinned = floating.with_pinned_benchmark("sha256:resolved")

    assert pinned.benchmark.ref == "sha256:resolved"
    assert pinned.to_harbor_config("job")["datasets"][0]["ref"] == "sha256:resolved"
    assert pinned.to_snapshot()["benchmark"]["ref"] == "sha256:resolved"


def test_resolved_benchmark_is_recorded_next_to_the_job(tmp_path):
    preparation = BenchmarkPreparation(
        dataset="nl2repobench/nl2repobench",
        resolved_ref="sha256:resolved",
        task_names=["nl2repobench/box", "nl2repobench/emoji"],
        images=[
            MirroredImage(
                task="nl2repobench/box",
                expected=f"{NL2REPOBENCH_PRIVATE_PREFIX}box:1.0",
                pulled_from="ghcr.io/multimodal-art-projection/nl2repobench/box:1.0",
                digest="ghcr.io/multimodal-art-projection/nl2repobench/box@sha256:abc",
                already_present=False,
            )
        ],
    )

    record_path = archive_benchmark(preparation, tmp_path / "job")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["resolved_ref"] == "sha256:resolved"
    assert record["n_tasks"] == 2
    assert record["images"][0]["digest"].endswith("sha256:abc")


def test_a_non_registry_dataset_is_rejected_before_any_download():
    settings = BenchmarkSettings(dataset="local-only")

    with pytest.raises(BenchmarkError):
        benchmark_module.prepare(settings, log=lambda message: None)


def test_tester_images_are_prepared_concurrently(monkeypatch):
    """104 images at roughly two gigabytes each is the fixed cost of a run.

    Pulling is bound by the network rather than by this machine, so preparing
    the images one after another would cost more wall-clock time than the run
    it precedes. What matters is that several are in flight at once.
    """
    import asyncio

    in_flight = 0
    peak = 0

    def slow_make_available(task, expected, source, log):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        time.sleep(0.02)
        in_flight -= 1
        return MirroredImage(
            task=task,
            expected=expected,
            pulled_from=source,
            digest=None,
            already_present=False,
        )

    monkeypatch.setattr(benchmark_module, "_make_available", slow_make_available)
    monkeypatch.setattr(
        benchmark_module,
        "_images_named_by",
        lambda task_dir: [f"{NL2REPOBENCH_PRIVATE_PREFIX}{task_dir.name}:1.0"],
    )
    rules = [
        ImageMirrorRule(
            expects=NL2REPOBENCH_PRIVATE_PREFIX,
            pull_from="ghcr.io/multimodal-art-projection/nl2repobench/",
        )
    ]
    tasks = [(f"nl2repobench/task-{index}", Path(f"task-{index}")) for index in range(8)]

    images = asyncio.run(
        _mirror_images(tasks, rules, log=lambda message: None, pull_concurrency=4)
    )

    assert len(images) == 8
    assert peak > 1
    assert peak <= 4


def test_an_image_two_tasks_share_is_prepared_once(monkeypatch):
    import asyncio

    prepared = []

    monkeypatch.setattr(
        benchmark_module,
        "_make_available",
        lambda task, expected, source, log: prepared.append(expected)
        or MirroredImage(task, expected, source, None, False),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_images_named_by",
        lambda task_dir: [f"{NL2REPOBENCH_PRIVATE_PREFIX}shared:1.0"],
    )
    rules = [
        ImageMirrorRule(
            expects=NL2REPOBENCH_PRIVATE_PREFIX,
            pull_from="ghcr.io/multimodal-art-projection/nl2repobench/",
        )
    ]

    images = asyncio.run(
        _mirror_images(
            [("nl2repobench/a", Path("a")), ("nl2repobench/b", Path("b"))],
            rules,
            log=lambda message: None,
            pull_concurrency=4,
        )
    )

    assert prepared == [f"{NL2REPOBENCH_PRIVATE_PREFIX}shared:1.0"]
    assert len(images) == 1


def test_what_a_trial_asks_for_is_the_largest_task_in_the_selection(tmp_path):
    """Concurrency has to hold for every task, so the maximum is the figure."""
    for name, cpus, memory_mb in (("small", 1, 2048), ("large", 2, 8192)):
        task_dir = tmp_path / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            f"[environment]\ncpus = {cpus}\nmemory_mb = {memory_mb}\n",
            encoding="utf-8",
        )

    assert _demand_of([tmp_path / "small", tmp_path / "large"]) == TrialDemand(
        cpus=2, memory_mb=8192
    )


def test_a_task_without_declared_resources_leaves_the_demand_unknown(tmp_path):
    (tmp_path / "task.toml").write_text('[task]\nname = "x"\n', encoding="utf-8")

    assert _demand_of([tmp_path]) == TrialDemand(cpus=None, memory_mb=None)


def test_concurrency_the_machine_can_hold_is_reported_without_a_remedy():
    advice = concurrency_advice(
        TrialDemand(cpus=2, memory_mb=2048),
        MachineCapacity(cpus=12, memory_mb=16384),
        n_concurrent_trials=4,
    )

    assert len(advice) == 1
    assert "2048 MB" in advice[0]


def test_concurrency_beyond_the_machine_is_reported_before_the_run_starts():
    """Ten hours into a run is an expensive place to learn this."""
    advice = concurrency_advice(
        TrialDemand(cpus=2, memory_mb=8192),
        MachineCapacity(cpus=12, memory_mb=7789),
        n_concurrent_trials=4,
    )

    assert len(advice) == 2
    assert "n_concurrent_trials" in advice[1]
    assert "override_memory_mb" in advice[1]


def test_no_advice_is_given_when_there_is_nothing_to_compare():
    assert concurrency_advice(TrialDemand(), MachineCapacity(), 4) == []
    assert concurrency_advice(TrialDemand(2, 8192), MachineCapacity(), 4) == []


def test_the_benchmark_record_states_what_a_trial_asks_for(tmp_path):
    """The record is what makes a result comparable after the fact.

    A run's concurrency is only interpretable next to what one trial of it
    needed, so the demand is archived with the tasks and images.
    """
    record = archive_benchmark(
        BenchmarkPreparation(
            dataset="nl2repobench/nl2repobench",
            resolved_ref="sha256:pinned",
            task_names=["nl2repobench/math-verify"],
            images=[],
            demand=TrialDemand(cpus=2, memory_mb=8192),
        ),
        tmp_path,
    )

    assert json.loads(record.read_text(encoding="utf-8"))["trial_demand"] == {
        "cpus": 2,
        "memory_mb": 8192,
    }
