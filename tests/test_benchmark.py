import json
from pathlib import Path

import pytest
import yaml

from evaluation_platform import benchmark as benchmark_module
from evaluation_platform.benchmark import (
    BenchmarkError,
    BenchmarkPreparation,
    BenchmarkSettings,
    ImageMirrorRule,
    MirroredImage,
    _images_named_by,
    _make_available,
    _mirror_source,
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

    def fake_docker(arguments, check=True, stream=False):
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
        lambda arguments, check=True, stream=False: (_ for _ in ()).throw(
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
