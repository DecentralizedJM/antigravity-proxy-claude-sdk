"""
Tests for synchronous and async clients.

Uses respx to mock HTTP requests without needing a running server.
"""

import pytest
import json
from unittest.mock import Mock, patch

import httpx
import respx

from antigravity_sdk import (
    AntigravityClient,
    AsyncAntigravityClient,
    Tool,
    Conversation,
    RateLimitError,
    AuthenticationError,
    ConnectionError,
)
from antigravity_sdk.retry import RetryConfig, NO_RETRY_CONFIG


# Sample API responses
SAMPLE_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Hello! How can I help you today?"}
    ],
    "model": "claude-sonnet-4-5-thinking",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 15,
    }
}

THINKING_RESPONSE = {
    "id": "msg_456",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "Let me analyze this..."},
        {"type": "text", "text": "The answer is 42."}
    ],
    "model": "claude-opus-4-5-thinking",
    "stop_reason": "end_turn",
}

TOOL_USE_RESPONSE = {
    "id": "msg_789",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "I'll check the weather for you."},
        {
            "type": "tool_use",
            "id": "tool_abc",
            "name": "get_weather",
            "input": {"location": "Tokyo"}
        }
    ],
    "model": "claude-sonnet-4-5-thinking",
    "stop_reason": "tool_use",
}


class TestAntigravityClient:
    """Tests for synchronous client."""
    
    def test_init_defaults(self):
        client = AntigravityClient()
        assert client.base_url == "http://localhost:8080"
        assert "claude" in client.model.lower() or "gemini" in client.model.lower()
    
    def test_init_custom(self):
        client = AntigravityClient(
            base_url="https://my-proxy.example.com",
            model="gemini-3-flash",
            timeout=60.0,
        )
        assert client.base_url == "https://my-proxy.example.com"
        assert client.model == "gemini-3-flash"
        assert client.timeout == 60.0
    
    @respx.mock
    def test_chat_success(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat("Hello!")
        
        assert response.id == "msg_123"
        assert response.text == "Hello! How can I help you today?"
        assert response.usage.input_tokens == 10
    
    @respx.mock
    def test_chat_with_system(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat("Hello!", system="Be concise.")
        
        # Verify system was sent
        request = respx.calls.last.request
        body = json.loads(request.content)
        assert body["system"] == "Be concise."
    
    @respx.mock
    def test_chat_with_thinking(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=THINKING_RESPONSE)
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat("What is 6 * 7?", model="claude-opus-4-5-thinking")
        
        assert response.thinking == "Let me analyze this..."
        assert response.text == "The answer is 42."
    
    @respx.mock
    def test_chat_with_tool_calls(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=TOOL_USE_RESPONSE)
        )
        
        tool = Tool.create(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string"}},
            required=["location"]
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat("What's the weather in Tokyo?", tools=[tool])
        
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].input == {"location": "Tokyo"}
    
    @respx.mock
    def test_chat_with_conversation(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        conv = client.create_conversation(system="Be helpful.")
        
        response = client.chat("Hello!", conversation=conv)
        
        # Conversation should be updated
        assert conv.message_count == 2  # User + Assistant
    
    @respx.mock
    def test_auth_error(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid token"}})
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        
        with pytest.raises(AuthenticationError):
            client.chat("Hello!")
    
    @respx.mock
    def test_rate_limit_error(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(429, json={
                "error": {"message": "Rate limited"},
                "retry_after": 30
            })
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        
        with pytest.raises(RateLimitError) as exc_info:
            client.chat("Hello!")
        
        assert exc_info.value.retry_after == 30
    
    @respx.mock
    def test_health_check_success(self):
        respx.get("http://localhost:8080/health").mock(
            return_value=httpx.Response(200)
        )
        
        client = AntigravityClient()
        assert client.health() is True
    
    @respx.mock
    def test_health_check_failure(self):
        respx.get("http://localhost:8080/health").mock(
            return_value=httpx.Response(500)
        )
        
        client = AntigravityClient()
        assert client.health() is False
    
    def test_context_manager(self):
        with AntigravityClient() as client:
            assert client._client is not None
        # Client should be closed after context


class TestAsyncAntigravityClient:
    """Tests for async client."""
    
    def test_init_defaults(self):
        client = AsyncAntigravityClient()
        assert client.base_url == "http://localhost:8080"
    
    @pytest.mark.asyncio
    @respx.mock
    async def test_chat_success(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        
        async with AsyncAntigravityClient(retry_config=NO_RETRY_CONFIG) as client:
            response = await client.chat("Hello!")
        
        assert response.text == "Hello! How can I help you today?"
    
    @pytest.mark.asyncio
    @respx.mock
    async def test_chat_with_conversation(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        
        async with AsyncAntigravityClient(retry_config=NO_RETRY_CONFIG) as client:
            conv = client.get_or_create_conversation(user_id=123)
            response = await client.chat("Hello!", conversation=conv)
        
        assert conv.message_count == 2
    
    @pytest.mark.asyncio
    @respx.mock
    async def test_health_check(self):
        respx.get("http://localhost:8080/health").mock(
            return_value=httpx.Response(200)
        )
        
        async with AsyncAntigravityClient() as client:
            healthy = await client.health()
        
        assert healthy is True
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncAntigravityClient() as client:
            assert client is not None


class TestChatWithTools:
    """Tests for automatic tool execution."""
    
    @respx.mock
    def test_chat_with_tools_single_round(self):
        # First call returns tool use
        # Second call returns final response
        respx.post("http://localhost:8080/v1/messages").mock(
            side_effect=[
                httpx.Response(200, json=TOOL_USE_RESPONSE),
                httpx.Response(200, json={
                    "id": "msg_final",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The weather in Tokyo is sunny!"}],
                    "model": "claude-sonnet-4-5-thinking",
                    "stop_reason": "end_turn",
                }),
            ]
        )
        
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny, 72°F"
        
        tool = Tool.create(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string"}},
            required=["location"]
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat_with_tools(
            "What's the weather in Tokyo?",
            tools=[tool],
            tool_handlers={"get_weather": get_weather}
        )
        
        assert response.text == "The weather in Tokyo is sunny!"
        assert not response.has_tool_calls
    
    @respx.mock
    def test_chat_with_tools_error_handling(self):
        respx.post("http://localhost:8080/v1/messages").mock(
            side_effect=[
                httpx.Response(200, json=TOOL_USE_RESPONSE),
                httpx.Response(200, json={
                    "id": "msg_final",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Sorry, there was an error."}],
                    "model": "claude-sonnet-4-5-thinking",
                    "stop_reason": "end_turn",
                }),
            ]
        )
        
        def failing_tool(location: str) -> str:
            raise ValueError("API error")
        
        tool = Tool.create(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string"}},
        )
        
        client = AntigravityClient(retry_config=NO_RETRY_CONFIG)
        response = client.chat_with_tools(
            "What's the weather?",
            tools=[tool],
            tool_handlers={"get_weather": failing_tool}
        )
        
        # Should still get a response (model handles the error)
        assert response is not None


class TestConversationManagement:
    """Tests for conversation management in clients."""
    
    def test_create_conversation(self):
        client = AntigravityClient(default_system="Default")
        conv = client.create_conversation()
        
        assert conv.system == "Default"
    
    def test_create_conversation_custom_system(self):
        client = AntigravityClient(default_system="Default")
        conv = client.create_conversation(system="Custom")
        
        assert conv.system == "Custom"
    
    def test_get_or_create_conversation(self):
        client = AntigravityClient()
        
        conv1 = client.get_or_create_conversation(user_id=123)
        conv2 = client.get_or_create_conversation(user_id=123)
        
        assert conv1 is conv2
    
    def test_clear_conversation(self):
        client = AntigravityClient()
        
        conv = client.get_or_create_conversation(user_id=123)
        conv.add_user("Hello!")
        
        cleared = client.clear_conversation(user_id=123)
        
        assert cleared is True
        assert conv.is_empty
