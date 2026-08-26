"""Terminus 2 as the agentic baseline.

Single-Shot asks what one request buys. This asks the harder question: what
does the *structure* of a multi-agent solution buy over the same model
iterating on its own? Terminus 2 is Harbor's reference agent — one model, one
shell, one loop, no roles, no plan and no review — so a solution that cannot
beat it has not shown that its roles are doing the work.

The adapter changes nothing about how Terminus behaves. It does two things.

It connects Terminus to the endpoint the experiment configured. Terminus
reaches the provider through LiteLLM, which finds a credential by the
provider's own environment variable name, while this platform names its
credential in the configuration and passes it to the agent as `extra_env`.
Without the bridge the two never meet, and the arm authenticates against
whatever happened to be in the launcher's environment instead of against the
configuration.

It records the setup, as every other arm does. There is no upstream revision
to pin here: Terminus is part of Harbor, so the version that ran is the
version of the Harbor this platform is installed with.

One consequence is worth stating where results are read rather than
discovered later. Terminus runs on the host and calls the provider through
LiteLLM, so what it spent is counted by Harbor's own accounting rather than by
`model_usage.py`. The figures mean the same thing — tokens in, tokens out —
but they are produced by a different counter than the other four arms use.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from evaluation_platform.experiment_config import TERMINUS
from evaluation_platform.model_routing import (
    ROUTING_ENV,
    routing_from_env,
    with_routing,
)

RESOLVED_SETUP_FILENAME = "resolved-setup.json"
# What identifies a Terminus run in place of an upstream revision.
IDENTITY = (
    "the Harbor release this platform is installed with, recorded in the "
    "run's resolved-setup.json"
)


@dataclass(frozen=True)
class ModelConnection:
    """The endpoint and credential an experiment configuration named."""

    base_url: str | None
    api_key_env: str
    api_key: str | None
    routing: dict[str, Any] | None


class TerminusAgent(Terminus2):
    """Harbor's Terminus 2, pointed at this experiment's model."""

    @staticmethod
    def name() -> str:
        """Terminus 2, under a name Harbor will not mistake for its own.

        Harbor's agent factory prefers a configured agent *name* over an
        `import_path` whenever that name is one of its built-ins, so an arm
        called plainly `terminus-2` would quietly run Harbor's own Terminus
        instead of this one — unbridged, against whatever credential happened
        to be in the launcher's environment, and recording nothing. The suffix
        is what keeps the configuration in charge. It is the same agent at the
        same version, and `resolved-setup.json` names both.
        """
        return TERMINUS.label

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hyperparameters = {
            name: kwargs.pop(name) for name in list(kwargs) if name in TERMINUS.types
        }
        TERMINUS.reject_upstream_revision(kwargs, IDENTITY)
        TERMINUS.reject_foreign(kwargs)
        self.hyperparameters = TERMINUS.resolve(hyperparameters)
        self.connection = model_connection(kwargs.get("extra_env"), os.environ)

        settings: dict[str, Any] = {
            name: value
            for name, value in self.hyperparameters.items()
            if name != "request_extra"
        }
        # Harbor's LiteLLM wrapper merges an `extra_body` given here into the
        # body of every request, which is the same place the other solutions'
        # `request_extra` and the experiment's routing end up. Stating either
        # once therefore means the same thing in every arm.
        extra_body = with_routing(
            self.hyperparameters.get("request_extra"), self.connection.routing
        )
        if extra_body:
            settings["llm_call_kwargs"] = {"extra_body": extra_body}
        if self.connection.base_url:
            settings["api_base"] = self.connection.base_url
        if self.connection.api_key:
            # LiteLLM forwards what it is given here to every request, which is
            # what lets the credential stay under the name the configuration
            # chose instead of having to be re-exported as OPENROUTER_API_KEY
            # or its equivalent for whichever provider is in use.
            settings["llm_kwargs"] = {"api_key": self.connection.api_key}

        super().__init__(*args, **settings, **kwargs)

        # The model call happens on the host, so the container has no use for
        # the credential. This agent is a shell in that container, and a secret
        # it does not need is one more thing a generated command could read.
        self._extra_env = {
            name: value
            for name, value in self._extra_env.items()
            if name != self.connection.api_key_env
        }

    async def run(
            self,
            instruction: str,
            environment: BaseEnvironment,
            context: AgentContext,
    ) -> None:
        self._record_resolved_setup()
        await super().run(instruction, environment, context)

    def _record_resolved_setup(self) -> None:
        """Record what actually ran, next to the run's other logs.

        The other arms write this from inside the task container, because that
        is where their tool ends up. Terminus runs on the host, so the record
        is written straight into the trial's agent log directory, which is the
        same place those files arrive.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / RESOLVED_SETUP_FILENAME).write_text(
            json.dumps(
                {
                    "solution": "Terminus 2",
                    # Terminus ships with Harbor, so this is the pin.
                    "harbor_version": _harbor_version(),
                    "terminus_version": self.version(),
                    "model": self.model_name,
                    "base_url": self.connection.base_url,
                    "api_key_env": self.connection.api_key_env,
                    # What this run asked of the aggregator. Which server
                    # actually answered is LiteLLM's to report, and Harbor
                    # records what it reports; the directive is stated here so
                    # the intent is on file either way.
                    "routing": self.connection.routing,
                    "hyperparameters": self.hyperparameters,
                    "reasoning_requested": bool(
                        self.hyperparameters.get("reasoning_effort")
                    ),
                    # Stated rather than assumed: this arm's tokens are counted
                    # by Harbor around LiteLLM, where the other arms' are
                    # counted by `model_usage.py` around the OpenAI client.
                    "usage_accounting": "harbor-litellm",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def model_connection(
        extra_env: Mapping[str, str] | None,
        environment: Mapping[str, str],
) -> ModelConnection:
    """Read the configured endpoint and credential, agent settings first.

    `to_harbor_config` passes the experiment's model settings to the agent as
    environment variables, and Harbor resolves the credential's value into
    them before the agent is built. The process environment is the fallback,
    which is what makes the agent constructible in a test, or by hand, without
    a Harbor job around it.
    """
    sources = (extra_env or {}, environment)

    def value(name: str) -> str | None:
        for source in sources:
            if name in source:
                return source[name]
        return None

    api_key_env = value("API_KEY_ENV") or "OPENAI_API_KEY"
    routing_source = {ROUTING_ENV: value(ROUTING_ENV) or ""}
    return ModelConnection(
        base_url=value("BASE_URL"),
        api_key_env=api_key_env,
        api_key=value(api_key_env),
        routing=routing_from_env(routing_source),
    )


def _harbor_version() -> str | None:
    try:
        return version("harbor")
    except PackageNotFoundError:  # pragma: no cover - Harbor is a dependency.
        return None
