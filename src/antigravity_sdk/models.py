"""
Antigravity SDK - Data Models

Pydantic models for request/response handling with full type safety.
Supports text, thinking blocks, tool use, and streaming.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class Role(str, Enum):
    """Message roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(str, Enum):
    """Reasons why the model stopped generating."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class ContentType(str, Enum):
    """Types of content blocks."""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"


# ============================================================================
# Content Blocks
# ============================================================================

class TextBlock(BaseModel):
    """A text content block."""
    type: Literal["text"] = "text"
    text: str
    
    model_config = ConfigDict(frozen=True)


class ThinkingBlock(BaseModel):
    """A thinking/reasoning content block (extended thinking models)."""
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None  # For multi-turn thinking continuity
    
    model_config = ConfigDict(frozen=True)


class ToolUseBlock(BaseModel):
    """A tool use request from the model."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]
    
    model_config = ConfigDict(frozen=True)


class ToolResultBlock(BaseModel):
    """Result of a tool execution to send back to the model."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: bool = False
    
    model_config = ConfigDict(frozen=True)


class ImageBlock(BaseModel):
    """An image content block."""
    type: Literal["image"] = "image"
    source: Dict[str, Any]  # {"type": "base64", "media_type": "...", "data": "..."}
    
    model_config = ConfigDict(frozen=True)


# Union type for all content blocks
ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock]


# ============================================================================
# Messages
# ============================================================================

class Message(BaseModel):
    """A message in a conversation."""
    role: Role
    content: Union[str, List[ContentBlock]]
    
    model_config = ConfigDict(use_enum_values=True)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)
    
    @classmethod
    def tool_result(cls, tool_use_id: str, result: str, is_error: bool = False) -> "Message":
        """Create a tool result message."""
        return cls(
            role=Role.USER,
            content=[ToolResultBlock(
                tool_use_id=tool_use_id,
                content=result,
                is_error=is_error
            )]
        )


# ============================================================================
# Tools
# ============================================================================

class ToolParameter(BaseModel):
    """A parameter for a tool."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


class ToolInputSchema(BaseModel):
    """JSON Schema for tool input."""
    type: Literal["object"] = "object"
    properties: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None


class Tool(BaseModel):
    """Definition of a tool the model can use."""
    name: str
    description: str
    input_schema: ToolInputSchema
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        required: Optional[List[str]] = None
    ) -> "Tool":
        """Convenience method to create a tool definition.
        
        Example:
            >>> tool = Tool.create(
            ...     name="get_weather",
            ...     description="Get the current weather for a location",
            ...     parameters={
            ...         "location": {"type": "string", "description": "City name"},
            ...         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            ...     },
            ...     required=["location"]
            ... )
        """
        return cls(
            name=name,
            description=description,
            input_schema=ToolInputSchema(
                properties=parameters,
                required=required or []
            )
        )


# ============================================================================
# API Response
# ============================================================================

class Usage(BaseModel):
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Response from a chat completion."""
    id: str
    role: str = "assistant"
    content: List[ContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def text(self) -> str:
        """Get the text content from the response.
        
        Concatenates all text blocks, ignoring thinking and tool use.
        """
        texts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)
    
    @property
    def thinking(self) -> Optional[str]:
        """Get the thinking content from the response (if any)."""
        for block in self.content:
            if isinstance(block, ThinkingBlock):
                return block.thinking
            elif isinstance(block, dict) and block.get("type") == "thinking":
                return block.get("thinking")
        return None
    
    @property
    def tool_calls(self) -> List[ToolUseBlock]:
        """Get all tool use blocks from the response."""
        tools = []
        for block in self.content:
            if isinstance(block, ToolUseBlock):
                tools.append(block)
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append(ToolUseBlock(**block))
        return tools
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0
    
    @property
    def is_complete(self) -> bool:
        """Check if the response completed normally (not truncated or tool use)."""
        return self.stop_reason == StopReason.END_TURN.value


# ============================================================================
# Streaming Events
# ============================================================================

class StreamEvent(BaseModel):
    """A streaming event from the API."""
    type: str
    index: Optional[int] = None
    content_block: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None
    message: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None


class StreamChunk(BaseModel):
    """A processed chunk from the stream."""
    type: Literal["text", "thinking", "tool_use", "done"] 
    text: Optional[str] = None
    thinking: Optional[str] = None
    tool_call: Optional[ToolUseBlock] = None
    usage: Optional[Usage] = None


# ============================================================================
# Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for model behavior."""
    model: str = "claude-opus-4-5-thinking"
    max_tokens: int = 8192
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    
    # Extended thinking settings
    thinking_budget: Optional[int] = None  # Max thinking tokens
    
    model_config = ConfigDict(frozen=True)


# ============================================================================
# Available Models
# ============================================================================

class AvailableModels:
    """Constants for available models."""
    
    # Claude models (via Antigravity)
    CLAUDE_OPUS_THINKING = "claude-opus-4-5-thinking"
    CLAUDE_SONNET_THINKING = "claude-sonnet-4-5-thinking"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    
    # Gemini models (via Antigravity)  
    GEMINI_FLASH = "gemini-3-flash"
    GEMINI_PRO_LOW = "gemini-3-pro-low"
    GEMINI_PRO_HIGH = "gemini-3-pro-high"
    
    @classmethod
    def all(cls) -> List[str]:
        """Get all available model names."""
        return [
            cls.CLAUDE_OPUS_THINKING,
            cls.CLAUDE_SONNET_THINKING,
            cls.CLAUDE_SONNET,
            cls.GEMINI_FLASH,
            cls.GEMINI_PRO_LOW,
            cls.GEMINI_PRO_HIGH,
        ]
    
    @classmethod
    def claude_models(cls) -> List[str]:
        """Get Claude model names."""
        return [
            cls.CLAUDE_OPUS_THINKING,
            cls.CLAUDE_SONNET_THINKING,
            cls.CLAUDE_SONNET,
        ]
    
    @classmethod
    def gemini_models(cls) -> List[str]:
        """Get Gemini model names."""
        return [
            cls.GEMINI_FLASH,
            cls.GEMINI_PRO_LOW,
            cls.GEMINI_PRO_HIGH,
        ]
