"""
Antigravity SDK - Python SDK for Antigravity Claude Proxy

Use Claude and Gemini models in your Python applications through
the Antigravity proxy server.

Example:
    >>> from antigravity_sdk import AntigravityClient
    >>> 
    >>> client = AntigravityClient()
    >>> response = client.chat("Hello, Claude!")
    >>> print(response.text)

Async Example:
    >>> from antigravity_sdk import AsyncAntigravityClient
    >>> 
    >>> async with AsyncAntigravityClient() as client:
    ...     response = await client.chat("Hello!")
    ...     print(response.text)
"""

__version__ = "1.0.0"
__author__ = "DecentralizedJM"

# Core clients
from .client import AntigravityClient
from .async_client import AsyncAntigravityClient

# Data models
from .models import (
    # Content blocks
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    ContentBlock,
    # Messages
    Message,
    Role,
    # Tools
    Tool,
    ToolInputSchema,
    # Response
    ChatResponse,
    Usage,
    StopReason,
    # Streaming
    StreamEvent,
    StreamChunk,
    # Configuration
    ModelConfig,
    AvailableModels,
)

# Conversation management
from .conversation import (
    Conversation,
    ConversationConfig,
    ConversationStore,
)

# Exceptions
from .exceptions import (
    AntigravityError,
    # Authentication
    AuthenticationError,
    AuthorizationError,
    # Rate limiting
    RateLimitError,
    QuotaExceededError,
    # Request errors
    InvalidRequestError,
    ContentFilterError,
    ContextLengthError,
    # Server errors
    ServerError,
    ServiceUnavailableError,
    OverloadedError,
    # Connection errors
    ConnectionError,
    TimeoutError,
    # Streaming
    StreamError,
    StreamInterruptedError,
    # Tools
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)

# Retry configuration
from .retry import (
    RetryConfig,
    DEFAULT_RETRY_CONFIG,
    NO_RETRY_CONFIG,
)


__all__ = [
    # Version
    "__version__",
    
    # Clients
    "AntigravityClient",
    "AsyncAntigravityClient",
    
    # Content blocks
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "ContentBlock",
    
    # Messages
    "Message",
    "Role",
    
    # Tools
    "Tool",
    "ToolInputSchema",
    
    # Response
    "ChatResponse",
    "Usage",
    "StopReason",
    
    # Streaming
    "StreamEvent",
    "StreamChunk",
    
    # Configuration
    "ModelConfig",
    "AvailableModels",
    
    # Conversation
    "Conversation",
    "ConversationConfig",
    "ConversationStore",
    
    # Exceptions
    "AntigravityError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "QuotaExceededError",
    "InvalidRequestError",
    "ContentFilterError",
    "ContextLengthError",
    "ServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "ConnectionError",
    "TimeoutError",
    "StreamError",
    "StreamInterruptedError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    
    # Retry
    "RetryConfig",
    "DEFAULT_RETRY_CONFIG",
    "NO_RETRY_CONFIG",
]
