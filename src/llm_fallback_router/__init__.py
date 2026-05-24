"""llm-fallback-router - multi-provider failover for LLM calls.

You want Anthropic. You also don't want a 503 from Anthropic to break a
production agent. `Router` takes an ordered list of providers and falls
through on retryable errors.

    from llm_fallback_router import Router, Provider

    router = Router([
        Provider("anthropic", anthropic_call),
        Provider("openai",    openai_call),
        Provider("gemini",    gemini_call),
    ])

    result = router.complete({"messages": [...], "max_tokens": 256})
    print(result.provider, result.tries)

`anthropic_call`, `openai_call`, etc. are user-supplied callables of shape
`(request: dict) -> object`. The router stays vendor-agnostic on purpose -
it doesn't know what's inside the request and doesn't translate between
providers. You handle the per-provider request shape; the router handles
the failover and audit.

Retryable predicate defaults to a small list that covers Anthropic /
OpenAI / Google / Bedrock conventions, but you can pass your own
`is_retryable(exc) -> bool`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "Router",
    "Provider",
    "Attempt",
    "RouteResult",
    "AllProvidersFailedError",
    "default_is_retryable",
]


# ---- types -----------------------------------------------------------------


CallFn = Callable[[dict], Any]
RetryableFn = Callable[[BaseException], bool]
OnAttemptFn = Callable[["Attempt"], None]


@dataclass(frozen=True)
class Provider:
    name: str
    call: CallFn
    is_retryable: RetryableFn | None = None


@dataclass(frozen=True)
class Attempt:
    provider: str
    ok: bool
    latency_ms: int
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RouteResult:
    provider: str
    response: Any
    tries: int
    attempts: list[Attempt] = field(default_factory=list)


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider has been tried and all failed."""

    def __init__(self, attempts: list[Attempt]):
        self.attempts = attempts
        names = ", ".join(f"{a.provider}({a.error_type})" for a in attempts)
        super().__init__(f"all providers failed: {names}")


# ---- retryable defaults ----------------------------------------------------


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

_RETRYABLE_TYPE_KEYWORDS = (
    "RateLimit",
    "ServiceUnavailable",
    "Overloaded",
    "Timeout",
    "APIConnectionError",
    "InternalServer",
    "ThrottlingException",
    "ModelStreamErrorException",
    "ServiceQuotaExceeded",
)


def default_is_retryable(exc: BaseException) -> bool:
    """Reasonable default retryable predicate.

    True for HTTP-ish status codes 408/409/425/429/500/502/503/504/529 and
    for a handful of vendor-specific exception class names. Network-level
    errors (ConnectionError, TimeoutError, OSError) are retryable too.
    """
    # native python network errors
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return True

    # status_code attr (httpx, anthropic, openai all expose this)
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True

    name = type(exc).__name__
    return any(kw in name for kw in _RETRYABLE_TYPE_KEYWORDS)


# ---- router ----------------------------------------------------------------


class Router:
    """Try providers in order, fall through on retryable errors.

    Args:
        providers: ordered list of `Provider`s. The router calls `provider.call`
            for each in turn until one succeeds or the list is exhausted.
        is_retryable: optional global retryable predicate. If a per-provider
            `is_retryable` is set, that wins for that provider; otherwise this
            global predicate is used; otherwise `default_is_retryable`.
        on_attempt: optional callback invoked after every attempt (success or
            failure). Useful for audit logs / metrics.
    """

    def __init__(
        self,
        providers: Iterable[Provider],
        *,
        is_retryable: RetryableFn | None = None,
        on_attempt: OnAttemptFn | None = None,
    ) -> None:
        self._providers: list[Provider] = list(providers)
        if not self._providers:
            raise ValueError("Router requires at least one provider")
        self._is_retryable = is_retryable or default_is_retryable
        self._on_attempt = on_attempt

    @property
    def providers(self) -> list[Provider]:
        return list(self._providers)

    def complete(self, request: dict) -> RouteResult:
        """Send `request` through the chain until one provider succeeds.

        Raises the original exception on a non-retryable error from any
        provider. Raises `AllProvidersFailedError` if every provider failed
        with a retryable error.
        """
        attempts: list[Attempt] = []
        for i, p in enumerate(self._providers):
            t0 = perf_counter()
            try:
                resp = p.call(request)
            except BaseException as exc:  # noqa: BLE001 - we re-raise below
                t1 = perf_counter()
                attempt = Attempt(
                    provider=p.name,
                    ok=False,
                    latency_ms=int((t1 - t0) * 1000),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:512],
                )
                attempts.append(attempt)
                if self._on_attempt:
                    self._on_attempt(attempt)

                pred = p.is_retryable or self._is_retryable
                if not pred(exc):
                    # non-retryable: surface the original exception
                    raise
                # retryable: keep trying the next provider
                continue
            t1 = perf_counter()
            attempt = Attempt(
                provider=p.name, ok=True, latency_ms=int((t1 - t0) * 1000)
            )
            attempts.append(attempt)
            if self._on_attempt:
                self._on_attempt(attempt)
            return RouteResult(
                provider=p.name,
                response=resp,
                tries=i + 1,
                attempts=attempts,
            )

        # every provider failed with a retryable error
        raise AllProvidersFailedError(attempts)
