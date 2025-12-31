"""
Antigravity SDK - Retry Logic

Exponential backoff with jitter for resilient API calls.
Handles transient errors and rate limits automatically.
"""

import random
import time
import asyncio
import logging
from typing import (
    TypeVar, Callable, Optional, Tuple, Type, Union,
    Awaitable, Any
)
from functools import wraps

from .exceptions import (
    AntigravityError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    OverloadedError,
    TimeoutError,
    ConnectionError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Exceptions that should be retried
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    OverloadedError,
    TimeoutError,
    ConnectionError,
)

# HTTP exceptions from httpx that should be retried
try:
    import httpx
    HTTPX_RETRYABLE = (
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    )
except ImportError:
    HTTPX_RETRYABLE = ()


class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2)
        jitter: Add randomness to delays (default: True)
        retry_on: Tuple of exception types to retry on
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or (RETRYABLE_EXCEPTIONS + HTTPX_RETRYABLE)
    
    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate delay for a given retry attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            retry_after: Server-specified retry delay (overrides calculation)
        
        Returns:
            Delay in seconds
        """
        # If server specifies retry-after, use it (with a cap)
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25% randomness)
        if self.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should be retried.
        
        Args:
            exception: The exception that was raised
            attempt: Current attempt number (0-indexed)
        
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if we've exhausted retries
        if attempt >= self.max_retries:
            return False
        
        # Check if exception type is retryable
        return isinstance(exception, self.retry_on)


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()

# No retry configuration
NO_RETRY_CONFIG = RetryConfig(max_retries=0)


def retry_sync(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for synchronous functions with retry logic.
    
    Example:
        >>> @retry_sync(RetryConfig(max_retries=5))
        ... def make_request():
        ...     return client.chat("Hello")
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        raise
                    
                    # Get retry_after if available
                    retry_after = None
                    if isinstance(e, RateLimitError):
                        retry_after = e.retry_after
                    
                    delay = config.calculate_delay(attempt, retry_after)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} after {delay:.2f}s "
                        f"due to {type(e).__name__}: {e}"
                    )
                    
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result or exception")
        
        return wrapper
    return decorator


def retry_async(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for async functions with retry logic.
    
    Example:
        >>> @retry_async(RetryConfig(max_retries=5))
        ... async def make_request():
        ...     return await client.chat("Hello")
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        raise
                    
                    # Get retry_after if available
                    retry_after = None
                    if isinstance(e, RateLimitError):
                        retry_after = e.retry_after
                    
                    delay = config.calculate_delay(attempt, retry_after)
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} after {delay:.2f}s "
                        f"due to {type(e).__name__}: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry loop completed without result or exception")
        
        return wrapper
    return decorator


async def retry_async_generator(
    func: Callable[..., Any],
    config: Optional[RetryConfig] = None,
    *args: Any,
    **kwargs: Any,
):
    """Retry logic for async generators (streaming).
    
    Note: Only retries if the generator fails before yielding anything.
    Once streaming starts, interruptions are not retried.
    
    Example:
        >>> async for chunk in retry_async_generator(client.stream, config, "Hello"):
        ...     print(chunk, end="")
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        try:
            async for item in func(*args, **kwargs):
                yield item
            return  # Successfully completed
        
        except Exception as e:
            last_exception = e
            
            if not config.should_retry(e, attempt):
                raise
            
            retry_after = None
            if isinstance(e, RateLimitError):
                retry_after = e.retry_after
            
            delay = config.calculate_delay(attempt, retry_after)
            
            logger.warning(
                f"Stream retry {attempt + 1}/{config.max_retries} after {delay:.2f}s "
                f"due to {type(e).__name__}: {e}"
            )
            
            await asyncio.sleep(delay)
    
    if last_exception:
        raise last_exception
