from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evaluation_platform.experiment_config import (
    ConfigurationError,
    load_experiment_config,
)
from evaluation_platform.model_routing import (
    ROUTING_ENV,
    ServedProviders,
    install_client_routing,
    routing_from_env,
    served_provider,
    with_routing,
)

ROOT = Path(__file__).parents[1]
ENVIRONMENT = {"API_KEY": "test-credential"}
PINNED = {"order": ["baidu/fp8", "siliconflow/fp8"], "allow_fallbacks": False}


def write_config(directory: Path, routing) -> Path:
    model = {
        "provider": "openrouter",
        "name": "test/free-model",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "API_KEY",
    }
    if routing is not None:
        model["routing"] = routing
    document = {
        "name": "unit-test",
        "run": {"jobs_dir": "jobs"},
        "benchmark": {"dataset": "nl2repobench/nl2repobench", "ref": "sha256:pinned"},
        "model": model,
        "agent": {"import_path": "evaluation_platform.single_shot_agent:Agent"},
    }
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def completions():
    """The OpenAI client class, restored after the wrapper is installed on it."""
    from openai.resources.chat import completions as module

    original = module.Completions.create
    calls: list[dict] = []

    def create(self, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(provider="Baidu")

    module.Completions.create = create
    try:
        yield module.Completions, calls
    finally:
        module.Completions.create = original
        if hasattr(module.Completions, "_evaluation_platform_create"):
            del module.Completions._evaluation_platform_create


def test_routing_is_pinned_beside_the_model_it_qualifies():
    """Why this lives in `model` rather than in a solution's hyperparameters.

    A model name does not name a server: the aggregator offers one model from
    many providers at different quantizations, and picks by price unless told
    otherwise. That makes routing part of what "the model" means for a run, and
    not a property of any method — so the test that already requires the model
    blocks to agree across configurations is the test that keeps every arm on
    one server.
    """
    pinned = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = load_experiment_config(path, ENVIRONMENT)
        assert config.model_routing, f"{path.name} leaves the server to price"
        pinned[path.name] = config.model_routing

    assert len(set(map(repr, pinned.values()))) == 1, pinned


def test_every_arm_is_told_the_same_way(tmp_path):
    """One directive, delivered identically whether an arm runs in the task
    container or on the host."""
    config = load_experiment_config(write_config(tmp_path, PINNED), ENVIRONMENT)

    environment = config.to_harbor_config("job")["agents"][0]["env"]

    assert routing_from_env(environment) == PINNED


def test_a_configuration_that_pins_nothing_says_so(tmp_path):
    config = load_experiment_config(write_config(tmp_path, None), ENVIRONMENT)

    assert config.model_routing is None
    assert routing_from_env(
        config.to_harbor_config("job")["agents"][0]["env"]
    ) is None


def test_the_pinned_routing_is_archived_with_the_run(tmp_path):
    config = load_experiment_config(write_config(tmp_path, PINNED), ENVIRONMENT)

    assert config.to_snapshot()["model"]["routing"] == PINNED


@pytest.mark.parametrize(
    "routing",
    [
        # The failure worth catching: the API accepts an unknown routing field
        # and ignores it, leaving the run routed by price while its record
        # says otherwise.
        {"providers": ["baidu/fp8"]},
        {"order": ["baidu/fp8"], "fallbacks": False},
        {},
        {"order": []},
        {"order": "baidu/fp8"},
        {"order": [""]},
        {"order": ["baidu/fp8"], "allow_fallbacks": "no"},
    ],
)
def test_a_directive_the_provider_would_ignore_is_rejected(tmp_path, routing):
    with pytest.raises(ConfigurationError):
        load_experiment_config(write_config(tmp_path, routing), ENVIRONMENT)


def test_routing_joins_whatever_else_the_request_asks_for():
    merged = with_routing({"reasoning": {"enabled": False}}, PINNED)

    assert merged == {"reasoning": {"enabled": False}, "provider": PINNED}


def test_the_experiments_routing_outranks_a_solutions_own():
    """A comparison in which one arm chose its own server is not one."""
    merged = with_routing({"provider": {"order": ["somewhere-else"]}}, PINNED)

    assert merged["provider"] == PINNED


def test_no_routing_leaves_a_request_as_it_was():
    assert with_routing({"reasoning": {"enabled": False}}, None) == {
        "reasoning": {"enabled": False}
    }
    assert with_routing(None, None) is None


def test_a_tool_that_builds_its_own_request_is_routed_anyway(completions):
    """The seam that means no published tool has to be modified.

    Self-Collaboration constructs its own client and its own request body
    several layers below anything this platform hands it. Wrapping the client
    class reaches those requests: what changes is the server they are sent to,
    not their messages, their sampling, or the model that answers.
    """
    client_class, calls = completions
    install_client_routing(PINNED)

    client_class.create(object(), model="m", messages=[])

    assert calls[0]["extra_body"] == {"provider": PINNED}


def test_the_wrapper_reads_back_which_server_answered(completions):
    """Pinning states an intent; this is what proves it held."""
    client_class, _ = completions
    observed = ServedProviders()
    install_client_routing(PINNED, observed)

    client_class.create(object(), model="m", messages=[])
    client_class.create(object(), model="m", messages=[])

    assert observed.names() == ["Baidu"]


def test_installing_twice_does_not_wrap_twice(completions):
    client_class, calls = completions
    install_client_routing(PINNED)
    install_client_routing(PINNED)

    client_class.create(object(), model="m", messages=[])

    assert calls[0]["extra_body"] == {"provider": PINNED}
    assert len(calls) == 1


def test_nothing_is_installed_when_there_is_nothing_to_do(completions):
    client_class, calls = completions

    assert install_client_routing(None) is False

    client_class.create(object(), model="m", messages=[])
    assert "extra_body" not in calls[0]


def test_a_response_that_names_no_provider_is_not_guessed_at():
    assert served_provider(SimpleNamespace()) is None
    assert served_provider(SimpleNamespace(provider="")) is None
    assert served_provider(SimpleNamespace(provider="Baidu")) == "Baidu"


def test_an_unreadable_directive_is_treated_as_none():
    assert routing_from_env({ROUTING_ENV: "not json"}) is None
    assert routing_from_env({ROUTING_ENV: "{}"}) is None
    assert routing_from_env({ROUTING_ENV: "[]"}) is None
    assert routing_from_env({}) is None
