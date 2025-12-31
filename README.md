# Antigravity Proxy Claude SDK

A production-ready Python SDK for the Antigravity Proxy, enabling seamless access to Claude and Gemini models through Google Cloud Code.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Hints](https://img.shields.io/badge/type%20hints-yes-brightgreen.svg)](https://www.python.org/dev/peps/pep-0484/)

## Features

- 🔄 **Sync & Async Clients** - Full support for both synchronous and asynchronous operations
- 💬 **Conversation Management** - Built-in multi-turn conversation handling with automatic trimming
- 🛠️ **Tool Calling** - First-class support for Claude's tool use with automatic execution
- 🔁 **Automatic Retries** - Exponential backoff with jitter for resilient API calls
- 📊 **Streaming** - Real-time token streaming for responsive UIs
- 🎯 **Type Safety** - Full Pydantic models with type hints throughout
- 🤖 **Telegram Ready** - Designed for high-concurrency bot applications
- 🧪 **Fully Tested** - Comprehensive test suite with pytest

## Installation

```bash
# Basic installation
pip install antigravity-proxy-sdk

# With Telegram bot support
pip install antigravity-proxy-sdk[telegram]

# Development installation
pip install antigravity-proxy-sdk[dev]
```

Or install from source:

```bash
git clone https://github.com/DecentralizedJM/antigravity-proxy-claude-sdk.git
cd antigravity-proxy-claude-sdk
pip install -e ".[dev]"
```

## Quick Start

### Simple Chat

```python
from antigravity_sdk import AntigravityClient

client = AntigravityClient()

response = client.chat("What is the capital of France?")
print(response.text)  # "The capital of France is Paris."
```

### With System Prompt

```python
response = client.chat(
    "Explain quantum computing",
    system="You are a teacher for 5th graders. Use simple words."
)
```

### Streaming Responses

```python
for chunk in client.stream("Write a poem about Python"):
    print(chunk, end="", flush=True)
```

### Async Client (for Telegram bots)

```python
import asyncio
from antigravity_sdk import AsyncAntigravityClient

async def main():
    async with AsyncAntigravityClient() as client:
        response = await client.chat("Hello!")
        print(response.text)

asyncio.run(main())
```

### Multi-Turn Conversations

```python
client = AntigravityClient()
conv = client.create_conversation(system="You are a helpful math tutor.")

# First turn
response1 = client.chat("What is 15% of 80?", conversation=conv)
print(response1.text)

# Second turn - remembers context
response2 = client.chat("How did you calculate that?", conversation=conv)
print(response2.text)
```

### Tool Calling

```python
from antigravity_sdk import AntigravityClient, Tool

# Define a tool
weather_tool = Tool.create(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "location": {"type": "string", "description": "City name"}
    },
    required=["location"]
)

# Tool handler
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

client = AntigravityClient()

# Auto-executes tools and returns final response
response = client.chat_with_tools(
    "What's the weather in Tokyo?",
    tools=[weather_tool],
    tool_handlers={"get_weather": get_weather}
)
print(response.text)
```

## Configuration

### Client Options

```python
from antigravity_sdk import AntigravityClient
from antigravity_sdk.retry import RetryConfig

client = AntigravityClient(
    base_url="http://localhost:8080",        # Proxy URL
    model="claude-sonnet-4-5-thinking",          # Default model
    max_tokens=4096,                         # Max response tokens
    timeout=30.0,                            # Request timeout
    default_system="You are helpful.",       # Default system prompt
    retry_config=RetryConfig(                # Retry configuration
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
    ),
    conversation_max_messages=50,            # Auto-trim conversations
)
```

### Available Models

```python
from antigravity_sdk import AvailableModels

# Claude models
AvailableModels.CLAUDE_OPUS_THINKING
AvailableModels.CLAUDE_SONNET_THINKING

# Gemini models  
AvailableModels.GEMINI_FLASH
AvailableModels.GEMINI_PRO
```

### Environment Variables

```bash
export ANTIGRAVITY_PROXY_URL="http://localhost:8080"
```

## Error Handling

```python
from antigravity_sdk import (
    AntigravityClient,
    AntigravityError,
    RateLimitError,
    AuthenticationError,
    ConnectionError,
)

client = AntigravityClient()

try:
    response = client.chat("Hello!")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError:
    print("Invalid credentials")
except ConnectionError:
    print("Cannot connect to proxy")
except AntigravityError as e:
    print(f"API error: {e}")
```

## Telegram Bot Example

```python
from telegram.ext import Application, MessageHandler, filters
from antigravity_sdk import AsyncAntigravityClient

client = AsyncAntigravityClient()

async def handle_message(update, context):
    user_id = update.effective_user.id
    conv = client.get_or_create_conversation(user_id=user_id)
    
    response = await client.chat(
        update.message.text,
        conversation=conv
    )
    
    await update.message.reply_text(response.text)

app = Application.builder().token("YOUR_TOKEN").build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.run_polling()
```

See [examples/telegram_bot.py](examples/telegram_bot.py) for a complete example.

## API Reference

### AntigravityClient

| Method | Description |
|--------|-------------|
| `chat(message, **kwargs)` | Single message chat |
| `chat_with_tools(message, tools, handlers)` | Chat with automatic tool execution |
| `stream(message, **kwargs)` | Stream response tokens |
| `stream_full(message, **kwargs)` | Stream with final response object |
| `create_conversation(system)` | Create a new conversation |
| `get_or_create_conversation(user_id)` | Get/create conversation by user ID |
| `clear_conversation(user_id)` | Clear a user's conversation |
| `health()` | Check proxy health |

### AsyncAntigravityClient

Same methods as `AntigravityClient` but async:

```python
response = await client.chat("Hello!")
async for chunk in client.stream("Hello!"):
    print(chunk)
```

### ChatResponse

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Response ID |
| `text` | `str` | Combined text content |
| `thinking` | `str\|None` | Thinking block content |
| `tool_calls` | `list[ToolUseBlock]` | Tool use requests |
| `has_tool_calls` | `bool` | Whether tools were called |
| `usage` | `Usage` | Token usage stats |
| `model` | `str` | Model used |
| `stop_reason` | `str` | Why generation stopped |

### Conversation

| Method | Description |
|--------|-------------|
| `add_user(content)` | Add user message |
| `add_assistant(content)` | Add assistant message |
| `add_tool_result(tool_use_id, content)` | Add tool result |
| `clear()` | Clear all messages |
| `export_json()` | Export to JSON |
| `from_json(data)` | Import from JSON |

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=antigravity_sdk

# Type checking
mypy src/

# Linting
ruff check src/
```

## Project Structure

```
antigravity-proxy-claude-sdk/
├── src/antigravity_sdk/
│   ├── __init__.py          # Package exports
│   ├── models.py             # Pydantic data models
│   ├── exceptions.py         # Exception hierarchy
│   ├── retry.py              # Retry logic with backoff
│   ├── conversation.py       # Conversation management
│   ├── client.py             # Synchronous client
│   └── async_client.py       # Async client
├── tests/
│   ├── test_models.py
│   ├── test_exceptions.py
│   ├── test_retry.py
│   ├── test_conversation.py
│   └── test_client.py
├── examples/
│   ├── basic_usage.py
│   └── telegram_bot.py
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.9+
- httpx >= 0.25.0
- pydantic >= 2.0.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `pytest` and `mypy`
5. Submit a pull request
