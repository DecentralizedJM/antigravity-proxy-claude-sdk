"""
Antigravity SDK - Custom Exceptions

Hierarchical exception classes for granular error handling.
All exceptions inherit from AntigravityError for easy catching.
"""

from typing import Optional, Dict, Any


class AntigravityError(Exception):
    """Base exception for all Antigravity SDK errors.
    
    Catch this to handle any SDK-related error.
    
    Example:
        >>> try:
        ...     response = client.chat("Hello")
        ... except AntigravityError as e:
        ...     print(f"SDK error: {e}")
    """
    
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(status: {self.status_code})")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, status_code={self.status_code})"


# ============================================================================
# Authentication Errors
# ============================================================================

class AuthenticationError(AntigravityError):
    """Authentication failed - invalid or expired token.
    
    This typically means:
    - The proxy couldn't authenticate with Antigravity
    - OAuth tokens need to be refreshed
    - Account needs re-authentication
    
    Solution: Check proxy logs, run `antigravity-claude-proxy accounts verify`
    """
    pass


class AuthorizationError(AntigravityError):
    """Authorization failed - insufficient permissions.
    
    The authenticated user doesn't have permission for this operation.
    """
    pass


# ============================================================================
# Rate Limiting Errors
# ============================================================================

class RateLimitError(AntigravityError):
    """Rate limit exceeded - too many requests.
    
    Attributes:
        retry_after: Seconds to wait before retrying (if provided)
    
    Solution: Wait for retry_after seconds, or the SDK will auto-retry
    """
    
    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
    
    def __str__(self) -> str:
        base = super().__str__()
        if self.retry_after:
            return f"{base} (retry after {self.retry_after}s)"
        return base


class QuotaExceededError(RateLimitError):
    """Daily/monthly quota exceeded.
    
    Unlike RateLimitError, this means the account has hit its usage limit.
    May need to wait until quota resets or use a different account.
    """
    pass


# ============================================================================
# Request Errors
# ============================================================================

class InvalidRequestError(AntigravityError):
    """The request was malformed or invalid.
    
    Common causes:
    - Invalid model name
    - max_tokens too large
    - Invalid message format
    - Missing required parameters
    """
    pass


class ContentFilterError(AntigravityError):
    """Content was blocked by safety filters.
    
    The request or response triggered content safety filters.
    """
    pass


class ContextLengthError(AntigravityError):
    """The conversation is too long for the model's context window.
    
    Solutions:
    - Truncate conversation history
    - Use summarization
    - Start a new conversation
    """
    pass


# ============================================================================
# Server Errors
# ============================================================================

class ServerError(AntigravityError):
    """Server-side error (5xx).
    
    The proxy or upstream API encountered an error.
    Usually transient - retry after a short delay.
    """
    pass


class ServiceUnavailableError(ServerError):
    """Service temporarily unavailable.
    
    The proxy or Antigravity API is temporarily down.
    Retry with exponential backoff.
    """
    pass


class OverloadedError(ServerError):
    """Server is overloaded.
    
    Too many requests are being processed.
    Retry with exponential backoff.
    """
    pass


# ============================================================================
# Connection Errors
# ============================================================================

class ConnectionError(AntigravityError):
    """Failed to connect to the proxy server.
    
    Common causes:
    - Proxy server not running
    - Wrong URL
    - Network issues
    - Firewall blocking connection
    
    Solution: Verify proxy is running at the configured URL
    """
    pass


class TimeoutError(AntigravityError):
    """Request timed out.
    
    The request took too long to complete.
    
    Solutions:
    - Increase timeout setting
    - Use streaming for long responses
    - Reduce max_tokens
    """
    pass


# ============================================================================
# Streaming Errors  
# ============================================================================

class StreamError(AntigravityError):
    """Error during streaming response.
    
    The stream was interrupted or contained invalid data.
    """
    pass


class StreamInterruptedError(StreamError):
    """Stream was interrupted before completion."""
    pass


# ============================================================================
# Tool Errors
# ============================================================================

class ToolError(AntigravityError):
    """Error related to tool use."""
    pass


class ToolExecutionError(ToolError):
    """Error executing a tool.
    
    The tool function raised an exception.
    """
    
    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_input: Dict[str, Any],
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.tool_input = tool_input


class ToolNotFoundError(ToolError):
    """Model tried to use a tool that wasn't provided."""
    
    def __init__(self, tool_name: str, **kwargs):
        super().__init__(f"Tool not found: {tool_name}", **kwargs)
        self.tool_name = tool_name


# ============================================================================
# Utility Functions
# ============================================================================

def raise_for_status(status_code: int, response_body: Optional[Dict[str, Any]] = None):
    """Raise appropriate exception based on HTTP status code.
    
    Args:
        status_code: HTTP status code
        response_body: Parsed JSON response body (if available)
    
    Raises:
        Appropriate AntigravityError subclass
    """
    if status_code < 400:
        return
    
    # Try to extract error message from response
    message = "Unknown error"
    error_type = None
    
    if response_body:
        if "error" in response_body:
            error = response_body["error"]
            if isinstance(error, dict):
                message = error.get("message", message)
                error_type = error.get("type")
            else:
                message = str(error)
        elif "message" in response_body:
            message = response_body["message"]
    
    # Map status codes to exceptions
    if status_code == 400:
        if error_type == "invalid_request_error":
            raise InvalidRequestError(message, status_code=status_code, response_body=response_body)
        raise InvalidRequestError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 401:
        raise AuthenticationError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 403:
        if "content" in message.lower() or "safety" in message.lower():
            raise ContentFilterError(message, status_code=status_code, response_body=response_body)
        raise AuthorizationError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 404:
        raise InvalidRequestError(f"Not found: {message}", status_code=status_code, response_body=response_body)
    
    elif status_code == 413:
        raise ContextLengthError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 429:
        # Try to extract retry-after
        retry_after = None
        if response_body and "retry_after" in response_body:
            retry_after = float(response_body["retry_after"])
        
        if "quota" in message.lower():
            raise QuotaExceededError(message, status_code=status_code, response_body=response_body, retry_after=retry_after)
        raise RateLimitError(message, status_code=status_code, response_body=response_body, retry_after=retry_after)
    
    elif status_code == 500:
        raise ServerError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 502:
        raise ServiceUnavailableError(f"Bad gateway: {message}", status_code=status_code, response_body=response_body)
    
    elif status_code == 503:
        raise ServiceUnavailableError(message, status_code=status_code, response_body=response_body)
    
    elif status_code == 504:
        raise TimeoutError(f"Gateway timeout: {message}", status_code=status_code, response_body=response_body)
    
    elif status_code == 529:
        raise OverloadedError(message, status_code=status_code, response_body=response_body)
    
    else:
        raise AntigravityError(message, status_code=status_code, response_body=response_body)
