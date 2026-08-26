"""Provider routing, shared by every code generation solution.

A model name does not name a server. An aggregator such as OpenRouter offers
one model from many providers, and they are not interchangeable: they serve
different quantizations of the same weights — fp4, fp8, bf16 — at different
speeds and prices, and the default route is chosen by price. Left alone, two
arms of a comparison can run the same model at different numerical precision,
and so can two requests inside one arm. Nothing records which.

That is a confound in the one variable every configuration here claims to hold
constant, so routing is pinned in the `model` block beside the model it
qualifies, not in any solution's hyperparameters. It is not a property of a
method, and the test that already requires the `model` blocks to agree across
configurations is therefore the test that keeps every arm on one server.

The directive reaches the container as `MODEL_ROUTING`, a JSON object placed
verbatim into each request as its `provider` field. Two ways in, because the
solutions differ in who builds the request:

- Where this platform builds the body, `with_routing` merges it in.
- Where the tool builds its own body, `install_client_routing` adds it to every
  request the OpenAI client sends. This is a change to the request rather than
  to the tool: the messages, the sampling and the model are untouched, and the
  tool's own repository stays exactly as published.

`served_provider` is the other half. Pinning states an intent; reading the
provider back out of each response is what proves it held.

This module must stay importable without Harbor: it is uploaded next to the
runners and imported by its flat name inside the task container.
"""

import json
import os
from typing import Any, Mapping

ROUTING_ENV = "MODEL_ROUTING"
# OpenRouter's provider-routing fields. Checked when a configuration is read so
# that a misspelling is a startup error rather than a directive the API accepts
# and quietly ignores, leaving the run unpinned and nobody any the wiser.
ROUTING_KEYS = frozenset(
    {
        "allow_fallbacks",
        "data_collection",
        "ignore",
        "only",
        "order",
        "quantizations",
        "require_parameters",
        "sort",
        "zdr",
    }
)


def routing_from_env(
        environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """The routing directive the experiment configured, if it configured one."""
    raw = (environ if environ is not None else os.environ).get(ROUTING_ENV, "")
    if not raw.strip():
        return None
    try:
        routing = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return routing if isinstance(routing, dict) and routing else None


def with_routing(
        extra_body: Mapping[str, Any] | None,
        routing: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge the routing directive into one request's extra body.

    The experiment's routing wins over anything a solution asked for under the
    same key: a comparison in which one arm chose its own server is not one.
    """
    if not routing:
        return dict(extra_body) if extra_body else None
    return {**(extra_body or {}), "provider": dict(routing)}


class ServedProviders:
    """Which servers actually answered, accumulated across a run.

    A pin states an intent. This is the observation, and the two are recorded
    side by side: a run whose directive named one endpoint and whose responses
    came from another is a run that did not measure what its setup says.
    """

    def __init__(self) -> None:
        self._names: set[str] = set()

    def record(self, response: Any) -> None:
        name = served_provider(response)
        if name is not None:
            self._names.add(name)

    def names(self) -> list[str]:
        return sorted(self._names)


def install_client_routing(
        routing: Mapping[str, Any] | None,
        observed: ServedProviders | None = None,
) -> bool:
    """Route every request an OpenAI client makes, whoever built the body.

    For a tool that constructs its own requests, and its own client, several
    layers below anything this platform passes it. Wrapping the client class
    reaches all of them without the tool being modified: what changes is the
    server the request is sent to, not the request's messages, its sampling, or
    the model that answers it.

    The same seam reads back which server answered, since it is the one place
    every response of such a tool passes through.

    Returns whether the wrapper was installed, so a run can record that it was.
    """
    if not routing and observed is None:
        return False

    from openai.resources.chat import completions

    original = getattr(completions.Completions, "_evaluation_platform_create", None)
    if original is None:
        original = completions.Completions.create

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        if routing:
            kwargs["extra_body"] = with_routing(kwargs.get("extra_body"), routing)
        response = original(self, *args, **kwargs)
        if observed is not None:
            observed.record(response)
        return response

    completions.Completions._evaluation_platform_create = original
    completions.Completions.create = create
    return bool(routing)


def served_provider(response: Any) -> str | None:
    """Which provider actually answered, as the aggregator reports it.

    OpenRouter returns this beside the completion, and the OpenAI client keeps
    fields it does not know about. A response without one — a different
    aggregator, a direct endpoint — is reported as unknown rather than guessed.
    """
    provider = getattr(response, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider
    return None
