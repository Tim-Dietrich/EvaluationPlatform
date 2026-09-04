import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.name import AgentName
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import (
    ConfigurationError,
    resolve_generation_kwargs,
)
from evaluation_platform.failure_categories import FailureTally
from evaluation_platform.terminus_agent import (
    RESOLVED_SETUP_FILENAME,
    TerminusAgent,
    _counting_calls,
    model_connection,
)


MODEL = "openrouter/moonshotai/kimi-k2.5"
# What `ExperimentConfig.to_harbor_config` hands the agent: the credential
# under the name the configuration chose for it, and the endpoint beside it.
AGENT_ENV = {
    "API_KEY": "secret-credential",
    "API_KEY_ENV": "API_KEY",
    "BASE_URL": "https://openrouter.ai/api/v1",
    "MODEL": "moonshotai/kimi-k2.5",
    "MODEL_ROUTING": '{"order": ["baidu/fp8"], "allow_fallbacks": false}',
}
PINNED = {"order": ["baidu/fp8"], "allow_fallbacks": False}


def terminus(tmp_path, env=None, **hyperparameters) -> TerminusAgent:
    return TerminusAgent(
        logs_dir=tmp_path,
        model_name=MODEL,
        extra_env=dict(AGENT_ENV if env is None else env),
        **hyperparameters,
    )


def test_the_experiments_endpoint_and_credential_reach_the_model(tmp_path):
    """The bridge this adapter exists for.

    Terminus reaches the provider through LiteLLM, which looks for a
    credential under the provider's own environment variable name. This
    platform names its credential in the configuration. Without the bridge the
    two never meet, and the arm would authenticate against whatever happened to
    be in the launcher's environment rather than against the configuration.
    """
    agent = terminus(tmp_path)

    assert agent._llm._api_base == "https://openrouter.ai/api/v1"
    assert agent._llm._llm_kwargs["api_key"] == "secret-credential"


def test_the_credential_is_not_handed_to_the_container(tmp_path):
    """The model call happens on the host, so the container has no use for it.

    This agent is a shell inside that container, and a secret it does not need
    is one more thing a generated command could read.
    """
    agent = terminus(tmp_path)

    assert "API_KEY" not in agent.extra_env
    assert agent.extra_env["BASE_URL"] == "https://openrouter.ai/api/v1"


def test_a_provider_setting_is_stated_once_and_means_the_same_everywhere(tmp_path):
    """`request_extra` is the same key, and the same place, as in the others.

    Three of the solutions send it through the OpenAI client; this one goes
    through LiteLLM, which carries it to the same part of the request body. An
    arm that reasons where another does not is not a comparison of methods.
    """
    agent = terminus(tmp_path, request_extra={"reasoning": {"enabled": False}})

    assert agent._llm_call_kwargs == {
        "extra_body": {"reasoning": {"enabled": False}, "provider": PINNED}
    }


def test_the_experiments_routing_reaches_every_request(tmp_path):
    """The arm that does not go through the OpenAI client is pinned too.

    Terminus reaches the provider through LiteLLM, and Harbor's wrapper merges
    an `extra_body` given here into the body of every request — the same place
    the other four arms' routing ends up. An arm left on the aggregator's
    default would run at whatever quantization price selected that day.
    """
    agent = terminus(tmp_path)

    assert agent.connection.routing == PINNED
    assert agent._llm_call_kwargs == {"extra_body": {"provider": PINNED}}


def test_an_unpinned_experiment_asks_for_nothing(tmp_path):
    agent = terminus(tmp_path, env={"API_KEY_ENV": "API_KEY", "API_KEY": "k"})

    assert agent.connection.routing is None
    assert agent._llm_call_kwargs == {}


def test_a_configured_turn_budget_bounds_a_tool_that_bounds_nothing(tmp_path):
    """Terminus stops at a million turns, which is no bound at all."""
    agent = terminus(tmp_path, max_turns=75)

    assert agent._max_episodes == 75


def test_omitted_settings_leave_terminus_at_its_own_defaults(tmp_path):
    agent = terminus(tmp_path)

    assert agent.hyperparameters == {"parser_name": "json", "enable_summarize": True}


def test_the_shared_sampling_reaches_terminus_by_both_of_its_routes(tmp_path):
    """One set of values, two parameters, because Harbor offers only one.

    Terminus declares a `temperature` and nothing for the other two, so `top_p`
    and `max_tokens` travel as `llm_kwargs`, which Harbor's LiteLLM wrapper
    spreads into the body of every request it sends. What matters is that the
    three values reaching the provider are the three every other arm sends.
    """
    shared = resolve_generation_kwargs({})
    agent = terminus(tmp_path)

    assert agent._temperature == shared["temperature"]
    assert agent._llm._temperature == shared["temperature"]
    assert agent._llm._llm_kwargs["top_p"] == shared["top_p"]
    assert agent._llm._llm_kwargs["max_tokens"] == shared["max_tokens"]
    # The credential travels the same way and is not displaced by them.
    assert agent._llm._llm_kwargs["api_key"] == "secret-credential"


def test_sampling_stated_for_this_arm_alone_is_refused(tmp_path):
    """They travel together, or a run samples at a value nobody chose."""
    with pytest.raises(ConfigurationError):
        terminus(tmp_path, temperature=0.3)


def test_a_token_limit_is_counted_on_the_way_past_and_not_swallowed():
    """Terminus recovers from both of these itself, and must go on doing so.

    It summarizes its history at the context ceiling and salvages what it can
    from a truncated reply. The counter sits in front of that and re-raises, so
    what the agent does is unchanged and what it spent recovering is on record.
    """

    class ContextLengthExceededError(Exception):
        pass

    failures = FailureTally()

    async def refuse(*args, **kwargs):
        raise ContextLengthExceededError("no room")

    counting = _counting_calls(refuse, failures)

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(counting("prompt"))

    assert failures.to_dict()["counts"]["context_exhausted"] == 1


def test_a_call_that_succeeds_is_returned_untouched():
    failures = FailureTally()

    async def answer(prompt):
        return f"answered {prompt}"

    assert asyncio.run(_counting_calls(answer, failures)("x")) == "answered x"
    assert failures.to_dict()["counts"]["context_exhausted"] == 0


@pytest.mark.parametrize(
    "setting",
    [
        # Terminus ships inside Harbor; there is no revision of it to pin.
        {"repository": "https://github.com/laude-institute/harbor.git"},
        {"commit": "a6490a9d0d32f3238cc5b776d2de8d2134d2b138"},
        # Self-Collaboration's, which Terminus has no role for.
        {"analyst_steps": 10},
        {"parser_name": "yaml"},
        {"reasoning_effort": "extreme"},
        {"max_turns": 0},
    ],
)
def test_settings_that_would_decide_nothing_are_rejected(tmp_path, setting):
    with pytest.raises(ConfigurationError):
        terminus(tmp_path, **setting)


def test_the_setup_is_recorded_where_the_other_arms_record_theirs(tmp_path):
    agent = terminus(tmp_path, max_turns=75)

    agent._record_resolved_setup()

    record = json.loads(
        (tmp_path / RESOLVED_SETUP_FILENAME).read_text(encoding="utf-8")
    )
    assert record["solution"] == "Terminus 2"
    # The pin, in place of an upstream revision: Terminus is part of Harbor.
    assert record["harbor_version"]
    assert record["terminus_version"] == agent.version()
    assert record["model"] == MODEL
    assert record["base_url"] == "https://openrouter.ai/api/v1"
    # The directive is on file even though which server answered is LiteLLM's
    # to report rather than this adapter's to read.
    assert record["routing"] == PINNED
    assert record["hyperparameters"]["max_turns"] == 75
    # The sampling this run used, and the route each value took, so a recorded
    # result can be checked against it rather than against a file that has
    # moved on since.
    shared = resolve_generation_kwargs({})
    assert {name: record["generation"][name] for name in shared} == shared
    assert record["generation"]["source"] == "configs/generation.yaml"
    assert set(record["generation"]["applied_via"]) == set(shared)
    # Why a run stopped, in categories that stay apart from a timeout and from
    # a test that failed. Zero before the agent has made a request.
    assert record["failures"]["counts"] == {
        "output_limit_exhausted": 0,
        "context_exhausted": 0,
        "request_timeout": 0,
    }
    # Stated rather than left to be discovered: this arm's tokens are counted
    # by Harbor around LiteLLM, the others' by `model_usage.py` around the
    # OpenAI client.
    assert record["usage_accounting"] == "harbor-litellm"
    # The credential is named, never written down.
    assert "secret-credential" not in json.dumps(record)


def test_the_setup_is_recorded_before_the_agent_is_let_loose(tmp_path, monkeypatch):
    """A run that dies in its first turn still says what it was."""
    seen: list[bool] = []

    async def fake_run(self, instruction, environment, context):
        seen.append((tmp_path / RESOLVED_SETUP_FILENAME).exists())

    monkeypatch.setattr(Terminus2, "run", fake_run)
    agent = terminus(tmp_path)

    asyncio.run(
        agent.run(
            "write it",
            cast(BaseEnvironment, cast(object, object())),
            AgentContext(),
        )
    )

    assert seen == [True]


def test_the_agents_own_settings_outrank_the_launchers_environment():
    """A stray provider credential in the shell must not decide the run."""
    connection = model_connection(
        {"API_KEY_ENV": "API_KEY", "API_KEY": "from-config", "BASE_URL": "https://a"},
        {"API_KEY": "from-shell", "BASE_URL": "https://b"},
    )

    assert connection.api_key == "from-config"
    assert connection.base_url == "https://a"


def test_the_process_environment_is_the_fallback():
    """What makes the agent constructible by hand, without a Harbor job."""
    connection = model_connection(None, {"OPENAI_API_KEY": "from-shell"})

    assert connection.api_key_env == "OPENAI_API_KEY"
    assert connection.api_key == "from-shell"
    assert connection.base_url is None


def test_the_agent_is_not_named_what_harbor_calls_its_own(tmp_path):
    """The one thing that must not be true of this arm's name.

    Harbor's factory prefers a configured agent *name* over an `import_path`
    whenever that name is one of its built-ins. Called plainly `terminus-2`,
    this arm would quietly run Harbor's own Terminus instead: unbridged,
    reaching for whatever credential was in the launcher's environment, and
    recording nothing about the setup. It is the same agent at the same
    version either way, and `resolved-setup.json` names both.
    """
    name = terminus(tmp_path).name()

    assert name not in AgentName.values()
    assert name.startswith("terminus-2")
