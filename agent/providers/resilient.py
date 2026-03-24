"""Resilient provider wrapper — retry, backoff, and provider fallback.

Wraps any BaseProvider with:
  1. Retries with exponential backoff on transient errors
  2. Timeout enforcement
  3. Fallback to an alternate provider if the primary keeps failing
  4. Error classification (transient vs permanent)
  5. Circuit breaker — if a provider fails N times in a row, skip it temporarily

Usage:
    from agent.providers.resilient import ResilientProvider

    provider = ResilientProvider(
        primary=claude_cli_provider,
        fallback=google_provider,
        fallback_model="gemini-2.5-flash",
    )

    # Works exactly like a normal provider
    response = await provider.complete(messages=..., model="sonnet")
"""

import asyncio
import time

import structlog

from .base import BaseProvider, LLMResponse

log = structlog.get_logger()

# Errors that are worth retrying
TRANSIENT_ERRORS = (
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
    RuntimeError,  # Catches our CLI subprocess errors
)

# Errors that should NOT be retried (bad request, auth, etc)
PERMANENT_ERROR_STRINGS = [
    "invalid_request",
    "authentication",
    "unauthorized",
    "invalid_api_key",
    "not_found",
    "permission",
]


def _is_permanent(error: Exception) -> bool:
    """Check if an error is permanent (retrying won't help)."""
    msg = str(error).lower()
    return any(s in msg for s in PERMANENT_ERROR_STRINGS)


class CircuitBreaker:
    """Tracks consecutive failures and temporarily disables a provider."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            log.warning("circuit_breaker_opened",
                        failures=self.consecutive_failures,
                        cooldown=self.cooldown_seconds)

    def is_available(self) -> bool:
        if not self.is_open:
            return True
        # Check if cooldown has elapsed
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.cooldown_seconds:
            log.info("circuit_breaker_half_open", elapsed=f"{elapsed:.0f}s")
            return True  # Allow one attempt
        return False


class ResilientProvider(BaseProvider):
    """Wraps a provider with retry logic and optional fallback."""

    def __init__(
        self,
        primary: BaseProvider,
        fallback: BaseProvider | None = None,
        fallback_model: str = "",
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 120.0,
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

        self.primary_circuit = CircuitBreaker()
        self.fallback_circuit = CircuitBreaker() if fallback else None

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # Try primary provider
        if self.primary_circuit.is_available():
            try:
                result = await self._try_with_retries(
                    provider=self.primary,
                    label="primary",
                    messages=messages,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                )
                self.primary_circuit.record_success()
                return result
            except Exception as e:
                self.primary_circuit.record_failure()
                log.warning("primary_provider_failed",
                            error=str(e),
                            will_fallback=self.fallback is not None)

                if not self.fallback or _is_permanent(e):
                    raise

        # Try fallback provider
        if self.fallback and self.fallback_circuit.is_available():
            fallback_model = self.fallback_model or model
            log.info("using_fallback_provider", model=fallback_model)

            try:
                result = await self._try_with_retries(
                    provider=self.fallback,
                    label="fallback",
                    messages=messages,
                    system=system,
                    model=fallback_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                )
                self.fallback_circuit.record_success()

                # Tag the response so the caller knows it was a fallback
                result.model = f"fallback:{result.model}"
                return result
            except Exception as e:
                self.fallback_circuit.record_failure()
                log.error("fallback_provider_also_failed", error=str(e))
                raise

        # Both providers unavailable
        raise RuntimeError(
            "All providers unavailable. Primary circuit breaker is open, "
            f"cooldown remaining: {self.primary_circuit.cooldown_seconds}s"
        )

    async def _try_with_retries(
        self,
        provider: BaseProvider,
        label: str,
        **kwargs,
    ) -> LLMResponse:
        """Try a provider with retries and exponential backoff."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    provider.complete(**kwargs),
                    timeout=self.timeout,
                )
                if attempt > 0:
                    log.info("retry_succeeded",
                             provider=label, attempt=attempt + 1)
                return result

            except Exception as e:
                last_error = e

                # Don't retry permanent errors
                if _is_permanent(e):
                    log.error("permanent_error",
                              provider=label, error=str(e))
                    raise

                # Don't retry if we've exhausted attempts
                if attempt >= self.max_retries:
                    break

                # Calculate backoff delay
                delay = min(
                    self.base_delay * (2 ** attempt),
                    self.max_delay,
                )

                log.warning("retrying_after_error",
                            provider=label,
                            attempt=attempt + 1,
                            max_retries=self.max_retries,
                            delay=f"{delay:.1f}s",
                            error=str(e)[:200])

                await asyncio.sleep(delay)

        raise last_error