"""The failure categories, and the two things they must not be confused with."""

import pytest

from evaluation_platform.failure_categories import (
    CONTEXT_EXHAUSTED,
    OUTPUT_LIMIT_EXHAUSTED,
    REQUEST_TIMEOUT,
    FailureTally,
    classify_error,
    classify_finish_reason,
)


class BadRequest(Exception):
    """An OpenAI-style 400, carrying the provider's payload the way one does."""

    def __init__(self, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status_code = 400
        self.body = body


class ContextLengthExceededError(Exception):
    """Harbor's own, by name: the arm that goes through LiteLLM raises this."""


class OutputLengthExceededError(Exception):
    """Harbor's own, likewise."""


def test_a_reply_that_stopped_at_the_ceiling_is_its_own_category():
    """Generation happened and was cut off. There is a reply, and it is partial."""
    assert classify_finish_reason("length") == OUTPUT_LIMIT_EXHAUSTED
    assert classify_finish_reason("stop") is None
    assert classify_finish_reason(None) is None
    # Not a failure of this kind: the model called a tool and stopped to wait.
    assert classify_finish_reason("tool_calls") is None


def test_a_request_refused_for_not_fitting_the_context_is_its_own_category():
    """No generation happened at all. The two are different findings.

    Prompt plus `max_tokens` overflowed the window, so the provider refused
    before writing a token. Recorded apart from a truncated reply because the
    remedy is different: one wants a smaller prompt, the other a larger ceiling.
    """
    assert classify_error(
        BadRequest("This model's maximum context length is 65536 tokens")
    ) == CONTEXT_EXHAUSTED
    assert classify_error(
        BadRequest("400", body="{'message': 'prompt is too long'}")
    ) == CONTEXT_EXHAUSTED
    assert classify_error(ContextLengthExceededError()) == CONTEXT_EXHAUSTED
    assert classify_error(OutputLengthExceededError("truncated")) == (
        OUTPUT_LIMIT_EXHAUSTED
    )


def test_a_timeout_is_kept_apart_from_both():
    """A request still being answered when the clock ran out says nothing about
    token limits, and must not be filed under either."""
    assert classify_error(TimeoutError("request timed out")) == REQUEST_TIMEOUT
    assert classify_error(RuntimeError("Read timeout")) == REQUEST_TIMEOUT


def test_a_slow_refusal_for_size_is_still_a_refusal_for_size():
    """A provider that takes its time refusing an oversized prompt refused it
    for the size, so the size is what is recorded."""
    assert classify_error(
        BadRequest("timed out after maximum context length exceeded")
    ) == CONTEXT_EXHAUSTED


def test_an_error_of_another_kind_is_left_uncategorised():
    """A refused credential is not a token limit, and is not filed as the
    nearest one available."""
    assert classify_error(BadRequest("invalid api key")) is None
    assert classify_error(RuntimeError("connection reset by peer")) is None


def test_a_category_survives_being_re_wrapped_by_a_tools_own_retry_loop():
    """The runners sit outside three tools' retry loops, and an error that has
    crossed one arrives wrapped as often as not."""
    cause = BadRequest("maximum context length is 65536 tokens")
    try:
        try:
            raise cause
        except BadRequest as error:
            raise RuntimeError("Failed to call LLM API with tools") from error
    except RuntimeError as wrapped:
        assert classify_error(wrapped) == CONTEXT_EXHAUSTED


def test_the_tally_counts_rather_than_flags():
    """One truncated reply among hundreds is a different finding from every
    reply being truncated, so the record is a count."""
    failures = FailureTally()

    failures.record_finish_reason("length")
    failures.record_finish_reason("length")
    failures.record_finish_reason("stop")
    failures.record_error(ContextLengthExceededError("no room"))

    assert failures.to_dict()["counts"] == {
        OUTPUT_LIMIT_EXHAUSTED: 2,
        CONTEXT_EXHAUSTED: 1,
        REQUEST_TIMEOUT: 0,
    }
    assert "no room" in failures.to_dict()["examples"][CONTEXT_EXHAUSTED]


def test_a_run_that_hit_nothing_says_so_rather_than_saying_nothing():
    """Every category is written, at zero, so a clean run and a run whose
    logging predates these categories do not look alike in the record."""
    assert FailureTally().to_dict()["counts"] == {
        OUTPUT_LIMIT_EXHAUSTED: 0,
        CONTEXT_EXHAUSTED: 0,
        REQUEST_TIMEOUT: 0,
    }


def test_a_failing_test_is_the_verifiers_finding_and_never_one_of_these():
    """The line these counters do not cross, stated in the record itself.

    Whether the generated code passes the benchmark's hidden tests is decided
    after this run has ended, by a verifier this module never sees. Nothing a
    test run reports can reach these counts.
    """
    failures = FailureTally()

    failures.record_error(AssertionError("2 failed, 5 passed"))
    failures.record_error(RuntimeError("pytest exited with status 1"))

    assert failures.to_dict()["counts"] == {
        OUTPUT_LIMIT_EXHAUSTED: 0,
        CONTEXT_EXHAUSTED: 0,
        REQUEST_TIMEOUT: 0,
    }
    assert "functional test" in failures.to_dict()["excludes"]


@pytest.mark.parametrize(
    "module",
    ["failure_categories"],
)
def test_the_container_half_stays_importable_without_harbor(module):
    """The runners import this by its flat name inside the task image, which
    has only what they install: no Harbor, and no `openai` until pip runs."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).parents[1] / "src" / "evaluation_platform" / f"{module}.py"
    ).read_text(encoding="utf-8")
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported <= {"dataclasses", "typing"}
