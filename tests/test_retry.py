"""
Tests for retry logic.
"""

import pytest
import time
from unittest.mock import Mock, patch

from antigravity_sdk.retry import (
    RetryConfig,
    DEFAULT_RETRY_CONFIG,
    NO_RETRY_CONFIG,
    retry_sync,
    retry_async,
)
from antigravity_sdk.exceptions import (
    RateLimitError,
    ServerError,
    AuthenticationError,
)


class TestRetryConfig:
    """Tests for RetryConfig."""
    
    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_values(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
    
    def test_calculate_delay_exponential(self):
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        # Attempt 0: 1 * 2^0 = 1
        assert config.calculate_delay(0) == 1.0
        # Attempt 1: 1 * 2^1 = 2
        assert config.calculate_delay(1) == 2.0
        # Attempt 2: 1 * 2^2 = 4
        assert config.calculate_delay(2) == 4.0
    
    def test_calculate_delay_respects_max(self):
        config = RetryConfig(base_delay=10.0, max_delay=30.0, jitter=False)
        
        # Attempt 3: 10 * 2^3 = 80, capped at 30
        assert config.calculate_delay(3) == 30.0
    
    def test_calculate_delay_with_retry_after(self):
        config = RetryConfig(max_delay=60.0)
        
        # Server says wait 45 seconds
        assert config.calculate_delay(0, retry_after=45.0) == 45.0
        
        # Server says wait 120 seconds, capped at max_delay
        assert config.calculate_delay(0, retry_after=120.0) == 60.0
    
    def test_calculate_delay_with_jitter(self):
        config = RetryConfig(base_delay=10.0, jitter=True)
        
        delays = [config.calculate_delay(0) for _ in range(100)]
        
        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # But stay within ±25% of base
        assert all(7.5 <= d <= 12.5 for d in delays)
    
    def test_should_retry_retryable_exceptions(self):
        config = RetryConfig(max_retries=3)
        
        assert config.should_retry(RateLimitError("rate"), attempt=0) is True
        assert config.should_retry(ServerError("server"), attempt=0) is True
    
    def test_should_retry_non_retryable_exceptions(self):
        config = RetryConfig(max_retries=3)
        
        # AuthenticationError should not be retried
        assert config.should_retry(AuthenticationError("auth"), attempt=0) is False
    
    def test_should_retry_exhausted(self):
        config = RetryConfig(max_retries=3)
        
        # After 3 attempts, should not retry even retryable errors
        assert config.should_retry(RateLimitError("rate"), attempt=3) is False


class TestRetrySyncDecorator:
    """Tests for sync retry decorator."""
    
    def test_success_no_retry(self):
        call_count = 0
        
        @retry_sync(RetryConfig(max_retries=3))
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_then_success(self):
        call_count = 0
        
        @retry_sync(RetryConfig(max_retries=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServerError("Temporary error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3
    
    def test_exhausted_retries(self):
        call_count = 0
        
        @retry_sync(RetryConfig(max_retries=2, base_delay=0.01))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ServerError("Always fails")
        
        with pytest.raises(ServerError):
            always_fails()
        
        # Initial + 2 retries = 3 calls
        assert call_count == 3
    
    def test_non_retryable_exception_not_retried(self):
        call_count = 0
        
        @retry_sync(RetryConfig(max_retries=3, base_delay=0.01))
        def auth_error():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid token")
        
        with pytest.raises(AuthenticationError):
            auth_error()
        
        # Should not retry auth errors
        assert call_count == 1


class TestRetryAsyncDecorator:
    """Tests for async retry decorator."""
    
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0
        
        @retry_async(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        call_count = 0
        
        @retry_async(RetryConfig(max_retries=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"
        
        result = await flaky_func()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        call_count = 0
        
        @retry_async(RetryConfig(max_retries=2, base_delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ServerError("Always fails")
        
        with pytest.raises(ServerError):
            await always_fails()
        
        assert call_count == 3


class TestDefaultConfigs:
    """Tests for default configurations."""
    
    def test_default_retry_config(self):
        assert DEFAULT_RETRY_CONFIG.max_retries == 3
        assert DEFAULT_RETRY_CONFIG.base_delay == 1.0
    
    def test_no_retry_config(self):
        assert NO_RETRY_CONFIG.max_retries == 0
        assert NO_RETRY_CONFIG.should_retry(ServerError("test"), attempt=0) is False
