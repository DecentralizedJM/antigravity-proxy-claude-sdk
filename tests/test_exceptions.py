"""
Tests for exception classes.
"""

import pytest
from antigravity_sdk.exceptions import (
    AntigravityError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    InvalidRequestError,
    ServerError,
    ConnectionError,
    TimeoutError,
    ToolExecutionError,
    raise_for_status,
)


class TestAntigravityError:
    """Tests for base exception."""
    
    def test_basic_creation(self):
        error = AntigravityError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.status_code is None
    
    def test_with_status_code(self):
        error = AntigravityError("Bad request", status_code=400)
        assert str(error) == "Bad request (status: 400)"
        assert error.status_code == 400
    
    def test_with_response_body(self):
        error = AntigravityError(
            "Error",
            status_code=500,
            response_body={"error": {"type": "server_error"}}
        )
        assert error.response_body["error"]["type"] == "server_error"
    
    def test_repr(self):
        error = AntigravityError("Test", status_code=404)
        assert "AntigravityError" in repr(error)
        assert "404" in repr(error)


class TestRateLimitError:
    """Tests for rate limit exception."""
    
    def test_with_retry_after(self):
        error = RateLimitError(
            "Rate limited",
            status_code=429,
            retry_after=30.5
        )
        assert error.retry_after == 30.5
        assert "30.5s" in str(error)
    
    def test_without_retry_after(self):
        error = RateLimitError("Rate limited", status_code=429)
        assert error.retry_after is None


class TestToolExecutionError:
    """Tests for tool execution exception."""
    
    def test_creation(self):
        error = ToolExecutionError(
            "Division by zero",
            tool_name="calculator",
            tool_input={"operation": "divide", "a": 1, "b": 0}
        )
        assert error.tool_name == "calculator"
        assert error.tool_input["b"] == 0


class TestRaiseForStatus:
    """Tests for raise_for_status utility."""
    
    def test_success_codes_no_raise(self):
        # Should not raise for success codes
        raise_for_status(200)
        raise_for_status(201)
        raise_for_status(204)
    
    def test_400_invalid_request(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            raise_for_status(400, {"error": {"message": "Invalid model"}})
        assert "Invalid model" in str(exc_info.value)
    
    def test_401_authentication(self):
        with pytest.raises(AuthenticationError):
            raise_for_status(401, {"error": {"message": "Invalid token"}})
    
    def test_429_rate_limit(self):
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, {"error": {"message": "Too many requests"}, "retry_after": 60})
        assert exc_info.value.retry_after == 60
    
    def test_429_quota_exceeded(self):
        with pytest.raises(QuotaExceededError):
            raise_for_status(429, {"error": {"message": "Quota exceeded for today"}})
    
    def test_500_server_error(self):
        with pytest.raises(ServerError):
            raise_for_status(500, {"error": {"message": "Internal error"}})
    
    def test_unknown_error_code(self):
        with pytest.raises(AntigravityError) as exc_info:
            raise_for_status(418, {"message": "I'm a teapot"})
        assert exc_info.value.status_code == 418


class TestExceptionHierarchy:
    """Test that exceptions can be caught by parent classes."""
    
    def test_catch_by_base(self):
        """All SDK exceptions should be catchable by AntigravityError."""
        exceptions = [
            AuthenticationError("auth"),
            RateLimitError("rate"),
            QuotaExceededError("quota"),
            InvalidRequestError("invalid"),
            ServerError("server"),
            ConnectionError("connection"),
            TimeoutError("timeout"),
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except AntigravityError:
                pass  # Should be caught
    
    def test_rate_limit_hierarchy(self):
        """QuotaExceededError should be catchable as RateLimitError."""
        try:
            raise QuotaExceededError("Quota exceeded")
        except RateLimitError:
            pass  # Should be caught
    
    def test_server_error_hierarchy(self):
        """ServiceUnavailableError should be catchable as ServerError."""
        from antigravity_sdk.exceptions import ServiceUnavailableError
        
        try:
            raise ServiceUnavailableError("Service down")
        except ServerError:
            pass  # Should be caught
