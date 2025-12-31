"""
Tests for Pydantic models.
"""

import pytest
from antigravity_sdk.models import (
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    Tool,
    ToolInputSchema,
    ChatResponse,
    Usage,
    AvailableModels,
)


class TestMessage:
    """Tests for Message model."""
    
    def test_user_message_factory(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"
    
    def test_assistant_message_factory(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_tool_result_message(self):
        msg = Message.tool_result("tool_123", "Result data", is_error=False)
        assert msg.role == Role.USER
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert msg.content[0].tool_use_id == "tool_123"
        assert msg.content[0].content == "Result data"
        assert msg.content[0].is_error is False


class TestContentBlocks:
    """Tests for content block models."""
    
    def test_text_block(self):
        block = TextBlock(text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"
    
    def test_thinking_block(self):
        block = ThinkingBlock(thinking="Let me think...", signature="sig_123")
        assert block.type == "thinking"
        assert block.thinking == "Let me think..."
        assert block.signature == "sig_123"
    
    def test_tool_use_block(self):
        block = ToolUseBlock(
            id="tool_456",
            name="get_weather",
            input={"location": "Tokyo"}
        )
        assert block.type == "tool_use"
        assert block.id == "tool_456"
        assert block.name == "get_weather"
        assert block.input == {"location": "Tokyo"}
    
    def test_tool_result_block(self):
        block = ToolResultBlock(
            tool_use_id="tool_456",
            content="Sunny, 72°F",
            is_error=False
        )
        assert block.type == "tool_result"
        assert block.tool_use_id == "tool_456"
        assert block.content == "Sunny, 72°F"


class TestTool:
    """Tests for Tool model."""
    
    def test_tool_create_factory(self):
        tool = Tool.create(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            required=["location"]
        )
        
        assert tool.name == "get_weather"
        assert tool.description == "Get the current weather"
        assert tool.input_schema.type == "object"
        assert "location" in tool.input_schema.properties
        assert tool.input_schema.required == ["location"]
    
    def test_tool_serialization(self):
        tool = Tool.create(
            name="test_tool",
            description="A test tool",
            parameters={"arg": {"type": "string"}},
            required=["arg"]
        )
        
        data = tool.model_dump()
        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"


class TestChatResponse:
    """Tests for ChatResponse model."""
    
    def test_text_property(self):
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[
                TextBlock(text="Hello "),
                TextBlock(text="world!"),
            ],
            model="claude-sonnet-4-5-thinking",
        )
        
        assert response.text == "Hello world!"
    
    def test_thinking_property(self):
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[
                ThinkingBlock(thinking="Analyzing the question..."),
                TextBlock(text="The answer is 42."),
            ],
            model="claude-opus-4-5-thinking",
        )
        
        assert response.thinking == "Analyzing the question..."
        assert response.text == "The answer is 42."
    
    def test_tool_calls_property(self):
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[
                TextBlock(text="I'll check the weather."),
                ToolUseBlock(
                    id="tool_789",
                    name="get_weather",
                    input={"location": "Tokyo"}
                ),
            ],
            model="claude-sonnet-4-5-thinking",
        )
        
        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
    
    def test_no_tool_calls(self):
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[TextBlock(text="Just text")],
            model="claude-sonnet-4-5",
        )
        
        assert response.has_tool_calls is False
        assert len(response.tool_calls) == 0
    
    def test_is_complete(self):
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[TextBlock(text="Done")],
            model="claude-sonnet-4-5",
            stop_reason="end_turn",
        )
        
        assert response.is_complete is True
    
    def test_with_dict_content_blocks(self):
        """Test handling of dict content blocks (from API)."""
        response = ChatResponse(
            id="msg_123",
            role="assistant",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "thinking", "thinking": "Hmm..."},
            ],
            model="claude-opus-4-5-thinking",
        )
        
        assert response.text == "Hello"
        assert response.thinking == "Hmm..."


class TestUsage:
    """Tests for Usage model."""
    
    def test_usage_creation(self):
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=25,
        )
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_input_tokens == 25
        assert usage.cache_creation_input_tokens is None


class TestAvailableModels:
    """Tests for AvailableModels constants."""
    
    def test_all_models(self):
        models = AvailableModels.all()
        assert len(models) >= 6
        assert "claude-opus-4-5-thinking" in models
        assert "gemini-3-flash" in models
    
    def test_claude_models(self):
        models = AvailableModels.claude_models()
        assert all("claude" in m for m in models)
    
    def test_gemini_models(self):
        models = AvailableModels.gemini_models()
        assert all("gemini" in m for m in models)
