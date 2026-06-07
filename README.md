# llm-fallback-router

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-fallback-router.svg)](https://pypi.org/project/llm-fallback-router/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Multi-provider failover for LLM calls.** Try Anthropic, fall back to OpenAI or Gemini or Bedrock on retryable errors. Per-attempt audit log. Zero runtime deps. BYO clients.

```python
from llm_fallback_router import Router, Provider

def anthropic_call(req):
    return anthropic.Anthropic().messages.create(**req)

def openai_call(req):
    return openai.OpenAI().responses.create(**openai_translate(req))

router = Router([
    Provider("anthropic", anthropic_call),
    Provider("openai",    openai_call),
])

result = router.complete({"model": "claude-opus-4-8", "messages": [...]})
print(result.provider, result.tries)   # "anthropic", 1 — or "openai", 2 on failover
```

## Why

In production, the question is never "what's my best provider" — it's "what do I do when my best provider is having a bad ten minutes." Half the LLM-resilience guides on the internet are slow tutorial code that does this in 30 lines and forgets to record what happened.

`llm-fallback-router` is the 100-line version that:

- Tries providers in order, falls through on retryable errors only
- Defaults to a status-code list that covers Anthropic / OpenAI / Google / Bedrock
- Lets you pass `is_retryable(exc) -> bool` if your predicate is weirder
- Calls `on_attempt(Attempt)` for every try so you can audit / meter
- Stays out of your way on request shape — you decide what the request dict looks like and you write the per-provider translation, because that part is genuinely vendor-specific

## Install

```bash
pip install llm-fallback-router
```

## API

```python
@dataclass(frozen=True)
class Provider:
    name: str
    call: Callable[[dict], Any]
    is_retryable: Callable[[BaseException], bool] | None = None

router = Router(
    providers,
    is_retryable=default_is_retryable,
    on_attempt=None,
)

result: RouteResult = router.complete(request: dict)
```

`RouteResult` has `.provider` (winner), `.response` (raw provider reply), `.tries`, and `.attempts: list[Attempt]` with per-attempt latency and error info.

`default_is_retryable` covers:

- HTTP statuses 408, 409, 425, 429, 500, 502, 503, 504, 529
- exception class names containing `RateLimit`, `ServiceUnavailable`, `Overloaded`, `Timeout`, `APIConnectionError`, `InternalServer`, `ThrottlingException`, `ModelStreamErrorException`, `ServiceQuotaExceeded`
- native `TimeoutError` / `ConnectionError` / `OSError`

If the last provider in the chain fails with a retryable error, the router raises `AllProvidersFailedError` with `.attempts` so you can see the full chain.

If any provider fails with a *non-retryable* error (e.g. 401 bad auth), the router re-raises immediately — you don't want a 401 on Anthropic to silently drain budget on three other providers.

## Companion libraries

- [`llm-retry`](https://github.com/MukundaKatta/llm-retry) — exponential-backoff retry of a *single* provider; pair this for `retry-then-fail-over` semantics.
- [`llm-circuit-breaker`](https://github.com/MukundaKatta/llm-circuit-breaker) — open the circuit on a provider after N failures so the router can skip it cleanly.
- [`agenttrace`](https://github.com/MukundaKatta/agenttrace) — wire `on_attempt` into per-run cost+latency aggregation.

## License

MIT
