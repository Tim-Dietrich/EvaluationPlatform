"""Model usage accounting, shared by every code generation solution.

This is one record written from two sides. Inside the task container a runner
accumulates the usage that each model response reports; on the host the agent
reads the resulting file back into Harbor's context, which is what the results
view aggregates and displays per run.

Both halves live here so that a token, a cached token, and a cost mean the same
thing whichever solution produced them. Two solutions counting usage in two
places is how a comparison between them quietly stops being one.

The container half is uploaded next to the runner and imported by its flat
name, so this module must stay importable without Harbor: the task image has
only what the runner installs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - the container half has no Harbor.
    from harbor.models.agent.context import AgentContext


@dataclass
class UsageTotals:
    """What a run spent, accumulated across every model response it received."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    responses: int = 0

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            # Reasoning tokens are billed as output tokens and included in the
            # total above. Recording them separately shows how much of the
            # budget a run spent thinking rather than answering, whether that
            # was asked for or inherited from the provider's default.
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
        }


def record_response_usage(response: Any, totals: UsageTotals) -> None:
    """Add one model response's reported usage to the running totals.

    A response that reports no usage at all is counted as nothing rather than
    as zero, so a provider that omits the field cannot be mistaken for one
    that answered for free.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    totals.input_tokens += usage_value(usage, "prompt_tokens")
    totals.output_tokens += usage_value(usage, "completion_tokens")
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    totals.cached_input_tokens += usage_value(prompt_details, "cached_tokens")
    completion_details = getattr(usage, "completion_tokens_details", None)
    totals.reasoning_tokens += usage_value(completion_details, "reasoning_tokens")
    cost = getattr(usage, "cost", None)
    if isinstance(cost, (int, float)):
        totals.cost_usd = (totals.cost_usd or 0.0) + float(cost)
    totals.responses += 1


def usage_value(container: Any, key: str) -> int:
    """Read one integer field from a usage object or the dict form of one."""
    if isinstance(container, dict):
        value = container.get(key)
    else:
        value = getattr(container, key, None)
    return value if isinstance(value, int) else 0


def populate_usage_context(context: "AgentContext", usage_path: Path) -> None:
    """Hand what the run spent to Harbor, if the runner got far enough to say.

    A run that crashed before writing the file, or wrote one that cannot be
    read, leaves Harbor's own fields untouched: an absent figure is reported as
    absent rather than as zero.
    """
    import json

    if not usage_path.exists():
        return
    try:
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(usage.get("input_tokens"), int):
        context.n_input_tokens = usage["input_tokens"]
    if isinstance(usage.get("cached_input_tokens"), int):
        context.n_cache_tokens = usage["cached_input_tokens"]
    if isinstance(usage.get("output_tokens"), int):
        context.n_output_tokens = usage["output_tokens"]
    if isinstance(usage.get("cost_usd"), (int, float)):
        context.cost_usd = float(usage["cost_usd"])
