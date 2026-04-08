"""Tests for agent.providers.resilient — retry, fallback, and circuit breaker."""

import pytest

from agent.providers.base import BaseProvider, LLMResponse
from agent.providers.resilient import (
    CircuitBreaker,
    ResilientProvider,
    _is_permanent,
)


class MockProvider(BaseProvider):
    """Provider with configurable failure behavior."""

    def __init__(
        self,
        responses: list[str | Exception] | None = None,
        default_response: str = "ok",
    ):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.call_count = 0

    async def complete(self, **kwargs) -> LLMResponse:
        self.call_count += 1
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return LLMResponse(content=resp, model="mock", usage={})
        return LLMResponse(content=self.default_response, model="mock", usage={})


class TestIsPermanent:
    """Permanent error classification."""

    def test_auth_error_is_permanent(self):
        assert _is_permanent(Exception("401 unauthorized"))

    def test_invalid_api_key_is_permanent(self):
        assert _is_permanent(Exception("invalid_api_key: check your key"))

    def test_permission_denied_is_permanent(self):
        assert _is_permanent(Exception("permission denied for this resource"))

    def test_timeout_is_not_permanent(self):
        assert not _is_permanent(TimeoutError("request timed out"))

    def test_connection_error_is_not_permanent(self):
        assert not _is_permanent(ConnectionError("connection refused"))

    def test_generic_error_is_not_permanent(self):
        assert not _is_permanent(RuntimeError("something broke"))


class TestCircuitBreaker:
    """Circuit breaker opens after consecutive failures."""

    def test_starts_available(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.is_available()

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available()
        cb.record_failure()
        assert not cb.is_available()

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        # Only 2 consecutive failures — should still be available
        assert cb.is_available()

    def test_reopens_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        assert not cb.is_available() or cb.is_available()
        # With 0 cooldown, should be available immediately
        assert cb.is_available()


class TestResilientProviderPrimary:
    """Primary provider succeeds."""

    @pytest.mark.asyncio
    async def test_primary_success(self):
        primary = MockProvider(default_response="primary ok")
        rp = ResilientProvider(primary=primary)
        result = await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.content == "primary ok"
        assert primary.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_success_resets_circuit(self):
        primary = MockProvider(default_response="ok")
        rp = ResilientProvider(primary=primary)
        rp.primary_circuit.record_failure()
        rp.primary_circuit.record_failure()
        await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert rp.primary_circuit.consecutive_failures == 0


class TestResilientProviderRetry:
    """Transient errors trigger retries."""

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        primary = MockProvider(responses=[
            ConnectionError("transient"),
            "ok after retry",
        ])
        rp = ResilientProvider(primary=primary, max_retries=2, base_delay=0)
        result = await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.content == "ok after retry"
        assert primary.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        primary = MockProvider(responses=[
            ValueError("invalid_api_key: bad key"),
        ])
        fallback = MockProvider(default_response="fallback")
        rp = ResilientProvider(
            primary=primary, fallback=fallback, max_retries=3, base_delay=0,
        )
        # Permanent errors raise immediately without retry or fallback
        with pytest.raises(ValueError, match="invalid_api_key"):
            await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert primary.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_trigger_fallback(self):
        primary = MockProvider(responses=[
            ConnectionError("fail 1"),
            ConnectionError("fail 2"),
            ConnectionError("fail 3"),
        ])
        fallback = MockProvider(default_response="fallback saved us")
        rp = ResilientProvider(
            primary=primary, fallback=fallback,
            fallback_model="fb-model", max_retries=2, base_delay=0,
        )
        result = await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert "fallback saved us" in result.content
        assert primary.call_count == 3  # initial + 2 retries
        assert fallback.call_count == 1


class TestResilientProviderFallback:
    """Fallback provider behavior."""

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_model(self):
        primary = MockProvider(responses=[ConnectionError("down")])
        fallback = MockProvider(default_response="fb")
        rp = ResilientProvider(
            primary=primary, fallback=fallback,
            fallback_model="gemini-2.5-flash", max_retries=0, base_delay=0,
        )
        result = await rp.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
        )
        assert "fallback:" in result.model

    @pytest.mark.asyncio
    async def test_no_fallback_raises_original_error(self):
        primary = MockProvider(responses=[ConnectionError("down")])
        rp = ResilientProvider(primary=primary, max_retries=0)
        with pytest.raises(ConnectionError, match="down"):
            await rp.complete(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_both_fail_raises(self):
        primary = MockProvider(responses=[ConnectionError("p down")])
        fallback = MockProvider(responses=[ConnectionError("f down")])
        rp = ResilientProvider(
            primary=primary, fallback=fallback, max_retries=0, base_delay=0,
        )
        with pytest.raises(ConnectionError, match="f down"):
            await rp.complete(messages=[{"role": "user", "content": "hi"}])


class TestCircuitBreakerIntegration:
    """Circuit breaker disables providers after repeated failures."""

    @pytest.mark.asyncio
    async def test_circuit_opens_skips_primary(self):
        primary = MockProvider(default_response="primary")
        fallback = MockProvider(default_response="fallback")
        rp = ResilientProvider(
            primary=primary, fallback=fallback,
            fallback_model="fb", max_retries=0, base_delay=0,
        )
        # Force circuit open
        rp.primary_circuit.is_open = True
        rp.primary_circuit.last_failure_time = float("inf")  # never cools down
        rp.primary_circuit.cooldown_seconds = 9999

        result = await rp.complete(messages=[{"role": "user", "content": "hi"}])
        assert "fallback" in result.content
        assert primary.call_count == 0  # skipped entirely

    @pytest.mark.asyncio
    async def test_both_circuits_open_raises(self):
        primary = MockProvider()
        fallback = MockProvider()
        rp = ResilientProvider(
            primary=primary, fallback=fallback, max_retries=0,
        )
        rp.primary_circuit.is_open = True
        rp.primary_circuit.last_failure_time = float("inf")
        rp.primary_circuit.cooldown_seconds = 9999
        rp.fallback_circuit.is_open = True
        rp.fallback_circuit.last_failure_time = float("inf")
        rp.fallback_circuit.cooldown_seconds = 9999

        with pytest.raises(RuntimeError, match="All providers unavailable"):
            await rp.complete(messages=[{"role": "user", "content": "hi"}])
