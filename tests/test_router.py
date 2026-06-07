"""Tests for llm_fallback_router.Router."""

from __future__ import annotations

import pytest

from llm_fallback_router import (
    AllProvidersFailedError,
    Attempt,
    Provider,
    RouteResult,
    Router,
    default_is_retryable,
)


# ---- fake errors -----------------------------------------------------------


class FakeRateLimitError(Exception):
    status_code = 429


class FakeOverloadedError(Exception):
    pass


class FakeAuthError(Exception):
    status_code = 401


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class FakeHttpStatusError(Exception):
    def __init__(self, status: int):
        super().__init__(f"http {status}")
        self.response = FakeResponse(status)


# ---- happy path ------------------------------------------------------------


def test_first_provider_wins_returns_route_result():
    def primary(req):
        return {"text": "hello", "from": "primary"}

    def secondary(req):
        raise AssertionError("must not be called")

    router = Router(
        [
            Provider("primary", primary),
            Provider("secondary", secondary),
        ]
    )
    out = router.complete({"messages": []})
    assert isinstance(out, RouteResult)
    assert out.provider == "primary"
    assert out.tries == 1
    assert out.response["text"] == "hello"
    assert len(out.attempts) == 1
    assert out.attempts[0].ok is True


def test_falls_through_on_retryable_error():
    def primary(req):
        raise FakeRateLimitError("rate limited")

    def secondary(req):
        return {"text": "ok"}

    router = Router(
        [
            Provider("primary", primary),
            Provider("secondary", secondary),
        ]
    )
    out = router.complete({"messages": []})
    assert out.provider == "secondary"
    assert out.tries == 2
    assert out.attempts[0].ok is False
    assert out.attempts[0].error_type == "FakeRateLimitError"
    assert out.attempts[1].ok is True


def test_falls_through_multiple_times():
    def p1(req):
        raise FakeRateLimitError()

    def p2(req):
        raise FakeOverloadedError("overloaded")

    def p3(req):
        return "win"

    router = Router(
        [
            Provider("a", p1),
            Provider("b", p2),
            Provider("c", p3),
        ]
    )
    out = router.complete({})
    assert out.provider == "c"
    assert out.tries == 3


# ---- non-retryable behavior ------------------------------------------------


def test_non_retryable_surfaces_immediately():
    def p1(req):
        raise FakeAuthError("bad key")

    def p2(req):
        return "should not see this"

    router = Router([Provider("a", p1), Provider("b", p2)])
    with pytest.raises(FakeAuthError):
        router.complete({})


def test_all_fail_raises_all_providers_failed():
    def p1(req):
        raise FakeRateLimitError()

    def p2(req):
        raise FakeOverloadedError("hot")

    router = Router([Provider("a", p1), Provider("b", p2)])
    with pytest.raises(AllProvidersFailedError) as exc:
        router.complete({})

    assert len(exc.value.attempts) == 2
    assert exc.value.attempts[0].error_type == "FakeRateLimitError"
    assert exc.value.attempts[1].error_type == "FakeOverloadedError"
    # the message names each failed provider and its error type
    msg = str(exc.value)
    assert "a(FakeRateLimitError)" in msg
    assert "b(FakeOverloadedError)" in msg


# ---- default_is_retryable predicate ---------------------------------------


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 529])
def test_default_is_retryable_status_codes(status):
    err = FakeHttpStatusError(status)
    assert default_is_retryable(err) is True


def test_default_is_retryable_class_name_keywords():
    assert default_is_retryable(FakeRateLimitError()) is True
    assert default_is_retryable(FakeOverloadedError()) is True


def test_default_is_retryable_rejects_401():
    assert default_is_retryable(FakeAuthError()) is False


def test_default_is_retryable_handles_native_network_errors():
    assert default_is_retryable(TimeoutError("slow")) is True
    assert default_is_retryable(ConnectionError("nope")) is True
    assert default_is_retryable(OSError("socket")) is True


def test_default_is_retryable_resolves_status_via_response_attr():
    # status_code lives on exc.response, not directly on exc (httpx style)
    assert default_is_retryable(FakeHttpStatusError(503)) is True
    assert default_is_retryable(FakeHttpStatusError(404)) is False


def test_default_is_retryable_rejects_plain_exception():
    assert default_is_retryable(ValueError("nope")) is False


# ---- custom predicates ----------------------------------------------------


def test_per_provider_is_retryable_overrides_global():
    def p1(req):
        raise FakeAuthError("normally non-retryable")

    def p2(req):
        return "second"

    router = Router(
        [
            Provider("a", p1, is_retryable=lambda e: True),
            Provider("b", p2),
        ]
    )
    out = router.complete({})
    assert out.provider == "b"


def test_global_is_retryable_overrides_default():
    calls = []

    def predicate(exc):
        calls.append(type(exc).__name__)
        return False

    def p1(req):
        raise FakeRateLimitError("would be retryable by default")

    def p2(req):
        return "second"

    router = Router(
        [Provider("a", p1), Provider("b", p2)],
        is_retryable=predicate,
    )
    with pytest.raises(FakeRateLimitError):
        router.complete({})
    assert calls == ["FakeRateLimitError"]


# ---- on_attempt audit hook -------------------------------------------------


def test_on_attempt_called_for_both_success_and_failure():
    seen: list[Attempt] = []

    def p1(req):
        raise FakeRateLimitError()

    def p2(req):
        return "ok"

    router = Router(
        [Provider("a", p1), Provider("b", p2)],
        on_attempt=seen.append,
    )
    router.complete({})
    assert [a.provider for a in seen] == ["a", "b"]
    assert seen[0].ok is False
    assert seen[1].ok is True


# ---- misc ------------------------------------------------------------------


def test_empty_providers_raises():
    with pytest.raises(ValueError):
        Router([])


def test_providers_property_returns_copy():
    p = Provider("a", lambda r: "x")
    router = Router([p])
    out = router.providers
    out.append(Provider("b", lambda r: "y"))
    assert len(router._providers) == 1  # internal unchanged


def test_attempt_records_latency_in_milliseconds():
    import time

    def p1(req):
        time.sleep(0.01)
        return "ok"

    router = Router([Provider("a", p1)])
    out = router.complete({})
    assert out.attempts[0].latency_ms >= 5
