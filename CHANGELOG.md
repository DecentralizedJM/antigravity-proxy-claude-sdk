# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added

- **Core SDK**
  - `AntigravityClient` - Synchronous client for API calls
  - `AsyncAntigravityClient` - Async client for Telegram bots and concurrent applications
  - Full Pydantic models for all API types
  - Comprehensive exception hierarchy

- **Conversation Management**
  - `Conversation` class for multi-turn conversations
  - `ConversationStore` for managing multiple user conversations
  - Automatic message trimming to stay within token limits
  - Export/import conversations as JSON

- **Tool Calling**
  - `Tool` model with factory method for easy creation
  - `chat_with_tools()` for automatic tool execution
  - Support for both sync and async tool handlers
  - Proper error handling for tool failures

- **Retry Logic**
  - Exponential backoff with jitter
  - Configurable retry attempts and delays
  - Automatic retry on rate limits and server errors
  - Support for streaming with retries

- **Streaming**
  - `stream()` method for real-time token streaming
  - `stream_full()` for streaming with final response object
  - Async streaming support

- **Error Handling**
  - `AntigravityError` - Base exception
  - `AuthenticationError` - Auth failures (401)
  - `RateLimitError` - Rate limits with retry_after (429)
  - `QuotaExceededError` - Quota exceeded
  - `InvalidRequestError` - Bad requests (400)
  - `ContentFilterError` - Content policy violations
  - `ServerError` - Server errors (5xx)
  - `ConnectionError` - Network issues
  - `TimeoutError` - Request timeouts
  - `ToolExecutionError` - Tool execution failures

- **Examples**
  - `basic_usage.py` - Core SDK features
  - `telegram_bot.py` - Complete Telegram bot example

- **Testing**
  - Full test suite with pytest
  - HTTP mocking with respx
  - Tests for all modules

### Models Supported

- Claude Opus 4.5 with thinking
- Claude Sonnet 4.5 with thinking
- Gemini 3 Flash
- Gemini 3 Pro

## [Unreleased]

### Planned

- Token counting utilities
- Response caching
- Batch processing
- Webhook support
- OpenAI-compatible interface
