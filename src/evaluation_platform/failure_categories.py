"""Why a run stopped, in categories that stay distinct from one another.

A reward figure says a run scored badly. It does not say whether the model was
wrong, whether it was cut off mid-sentence at the output ceiling, or whether the
provider refused the request before generating anything at all. Those are three
findings, and reading the first where the truth was the second or third is how a
comparison of methods quietly becomes a comparison of token budgets.

Two of them are recorded here:

  * `output_limit_exhausted` — generation began and stopped at the token cap.
    The reply exists and is truncated. `finish_reason == "length"` on the
    OpenAI-compatible path, `OutputLengthExceededError` on Harbor's LiteLLM one.
  * `context_exhausted` — the request was refused before generation, because the
    prompt plus the requested `max_tokens` did not fit the model's context
    window. There is no reply at all. A 400 whose payload names the context
    length on the OpenAI-compatible path, `ContextLengthExceededError` on
    Harbor's.

Both are kept apart from the two things they are most often confused with. A
timeout is its own category, `request_timeout`: a request that was still being
answered when the clock ran out says nothing about token limits. A functional
test failure is not here at all — that is the verifier's finding about generated
code, produced after this run has ended, and it never reaches these counters.

The container half of every solution imports this module by its flat name next
to its runner, so it must stay importable without Harbor and without the
`openai` package: an exception is classified by what it carries, never by
`isinstance` against a library the task image may not have.
"""

from dataclasses import dataclass, field
from typing import Any

# Generation began and ran into the output ceiling.
OUTPUT_LIMIT_EXHAUSTED = "output_limit_exhausted"
# The request was refused: prompt + `max_tokens` exceeded the context window.
CONTEXT_EXHAUSTED = "context_exhausted"
# The request was still being answered when its clock ran out. Named so that a
# timeout cannot be filed under either of the two above by default.
REQUEST_TIMEOUT = "request_timeout"

CATEGORIES = (OUTPUT_LIMIT_EXHAUSTED, CONTEXT_EXHAUSTED, REQUEST_TIMEOUT)

# What a provider says when the prompt plus the requested output does not fit.
# The same set Harbor's LiteLLM wrapper matches on, so an arm reaching the
# provider through the OpenAI client and an arm reaching it through LiteLLM
# classify one refusal the same way.
_CONTEXT_LENGTH_PHRASES = (
    "context length exceeded",
    "context_length_exceeded",
    "maximum context length",
    "`inputs` tokens + `max_new_tokens`",
    "model's context length",
    "prompt is too long",
    "input is too long for requested model",
    "reduce the length of the messages",
)
_TIMEOUT_PHRASES = ("timed out", "timeout")


def classify_finish_reason(finish_reason: Any) -> str | None:
    """The category a completed response falls into, if it failed at all.

    Only the ceiling is visible here. A response that finished normally, or for
    a reason of the provider's own, is not a failure of this kind and is left
    uncategorised.
    """
    return OUTPUT_LIMIT_EXHAUSTED if finish_reason == "length" else None


def classify_error(error: BaseException) -> str | None:
    """The category a failed request falls into, if it is one of these.

    Read from what the exception carries rather than from its type: the runners
    that use this are five different call paths through three libraries, and an
    error that has crossed a tool's own retry loop arrives re-wrapped as often
    as not. Anything else — a refused credential, an unknown model, a network
    reset — is not one of these categories and returns `None` rather than being
    filed under the nearest one.
    """
    for candidate in _causes(error):
        name = type(candidate).__name__
        text = " ".join(
            str(part)
            for part in (
                candidate,
                getattr(candidate, "message", ""),
                getattr(candidate, "body", ""),
            )
            if part
        ).lower()

        # Harbor's own names, for the arm that reaches the provider through it.
        if name == "ContextLengthExceededError":
            return CONTEXT_EXHAUSTED
        if name == "OutputLengthExceededError":
            return OUTPUT_LIMIT_EXHAUSTED
        if any(phrase in text for phrase in _CONTEXT_LENGTH_PHRASES):
            return CONTEXT_EXHAUSTED
        # A timeout is checked last: a provider that refuses an oversized
        # request slowly is still refusing it for the size.
        if "Timeout" in name or any(phrase in text for phrase in _TIMEOUT_PHRASES):
            return REQUEST_TIMEOUT
    return None


def _causes(error: BaseException) -> list[BaseException]:
    """The exception and what it was raised from, outermost first."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


@dataclass
class FailureTally:
    """How often each category occurred over one run, and what it looked like.

    A count rather than a flag: a multi-agent run makes hundreds of requests,
    and one truncated reply among them is a different finding from every reply
    being truncated. The first example of each is kept because a category
    without a message is hard to act on when reading a run months later.
    """

    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)

    def record(self, category: str | None, detail: Any = None) -> str | None:
        """Count one occurrence, and return the category for the caller to act on."""
        if category is None:
            return None
        self.counts[category] = self.counts.get(category, 0) + 1
        if category not in self.examples and detail is not None:
            self.examples[category] = str(detail)[:500]
        return category

    def record_finish_reason(self, finish_reason: Any) -> str | None:
        return self.record(
            classify_finish_reason(finish_reason),
            f"finish_reason={finish_reason!r}",
        )

    def record_error(self, error: BaseException) -> str | None:
        return self.record(
            classify_error(error), f"{type(error).__name__}: {error}"
        )

    def to_dict(self) -> dict[str, Any]:
        """The run's failure categories, every one of them stated.

        Categories that did not occur are written as zero rather than left out,
        so a run that hit no ceiling and a run whose logging predates this file
        do not look alike in the record.
        """
        return {
            "counts": {
                category: self.counts.get(category, 0) for category in CATEGORIES
            },
            "examples": dict(self.examples),
            # Said out loud where the categories are read: whether the generated
            # code passes the benchmark's hidden tests is the verifier's finding
            # and is never one of the counts above.
            "excludes": "functional test outcomes, which the verifier reports",
        }
