"""
Antigravity SDK - Async Client

Production-ready async client for Telegram bots and async applications.
Includes retry logic, tool support, and conversation management.
"""

import os
import json
import logging
from typing import List, Optional, AsyncIterator, Dict, Any, Callable, Union, Awaitable

import httpx

from .models import (
    Message, ChatResponse, ContentBlock, TextBlock, ThinkingBlock,
    ToolUseBlock, Tool, Usage, AvailableModels, Role,
)
from .exceptions import (
    AntigravityError, ConnectionError, TimeoutError, StreamError,
    raise_for_status,
)
from .retry import RetryConfig, DEFAULT_RETRY_CONFIG, retry_async
from .conversation import Conversation, ConversationStore

logger = logging.getLogger(__name__)


class AsyncAntigravityClient:
    """Async client for the Antigravity Claude Proxy.
    
    Perfect for Telegram bots and async applications.
    
    Features:
    - Fully async with httpx
    - Automatic retry with exponential backoff
    - Tool/function calling support
    - Conversation history management
    - Streaming responses
    
    Example:
        >>> async with AsyncAntigravityClient() as client:
        ...     response = await client.chat("Hello!")
        ...     print(response.text)
        
        # With conversation (for Telegram bots)
        >>> conv = client.get_or_create_conversation(user_id=12345)
        >>> response = await client.chat("Hi!", conversation=conv)
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-5-thinking"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        retry_config: Optional[RetryConfig] = None,
        default_system: Optional[str] = None,
    ):
        """Initialize the async client.
        
        Args:
            base_url: Proxy URL. Defaults to ANTIGRAVITY_PROXY_URL env var or localhost:8080
            model: Default model to use
            timeout: Request timeout in seconds
            retry_config: Configuration for automatic retries
            default_system: Default system prompt for conversations
        """
        self.base_url = (
            base_url 
            or os.getenv("ANTIGRAVITY_PROXY_URL") 
            or "http://localhost:8080"
        ).rstrip("/")
        self.model = model or os.getenv("ANTIGRAVITY_MODEL") or self.DEFAULT_MODEL
        self.timeout = timeout
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self.default_system = default_system
        
        self._client: Optional[httpx.AsyncClient] = None
        
        # Conversation store for multi-turn chats
        self._conversation_store = ConversationStore(
            default_system=default_system
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                follow_redirects=True,
            )
        return self._client
    
    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        if system:
            payload["system"] = system
        
        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema.model_dump(),
                }
                for t in tools
            ]
        
        payload.update(kwargs)
        return payload
    
    def _parse_response(self, data: Dict[str, Any]) -> ChatResponse:
        """Parse API response into ChatResponse."""
        content = []
        
        for block in data.get("content", []):
            block_type = block.get("type")
            
            if block_type == "text":
                content.append(TextBlock(text=block.get("text", "")))
            
            elif block_type == "thinking":
                content.append(ThinkingBlock(
                    thinking=block.get("thinking", ""),
                    signature=block.get("signature"),
                ))
            
            elif block_type == "tool_use":
                content.append(ToolUseBlock(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))
        
        usage = None
        if "usage" in data:
            usage = Usage(**data["usage"])
        
        return ChatResponse(
            id=data.get("id", ""),
            role=data.get("role", "assistant"),
            content=content,
            model=data.get("model", self.model),
            stop_reason=data.get("stop_reason"),
            usage=usage,
        )
    
    async def _make_request(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> ChatResponse:
        """Make a non-streaming API request with retry logic."""
        payload = self._build_payload(messages, stream=False, **kwargs)
        client = await self._get_client()
        
        @retry_async(self.retry_config)
        async def do_request() -> ChatResponse:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.ConnectError as e:
                raise ConnectionError(f"Failed to connect to proxy at {self.base_url}") from e
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Request timed out after {self.timeout}s") from e
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"error": response.text}
            
            if response.status_code >= 400:
                raise_for_status(response.status_code, data)
            
            return self._parse_response(data)
        
        return await do_request()
    
    async def chat(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        messages: Optional[List[Message]] = None,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat message and get a response.
        
        Args:
            message: The user message to send
            model: Override the default model
            system: System prompt (overrides conversation/default system)
            conversation: Conversation object for multi-turn chats
            messages: Raw message history (alternative to conversation)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            tools: List of tools the model can use
            **kwargs: Additional API parameters
            
        Returns:
            ChatResponse with the assistant's reply
            
        Example:
            >>> async with AsyncAntigravityClient() as client:
            ...     response = await client.chat("Hello!")
            ...     print(response.text)
        """
        effective_system = system
        if effective_system is None and conversation:
            effective_system = conversation.system
        if effective_system is None:
            effective_system = self.default_system
        
        if conversation:
            conversation.add_user(message)
            msgs = conversation.to_messages_list()
        elif messages:
            msgs = [{"role": m.role, "content": m.content} for m in messages]
            msgs.append({"role": "user", "content": message})
        else:
            msgs = [{"role": "user", "content": message}]
        
        response = await self._make_request(
            messages=msgs,
            model=model,
            system=effective_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )
        
        if conversation:
            content_for_storage = []
            for block in response.content:
                if hasattr(block, "model_dump"):
                    content_for_storage.append(block.model_dump())
                elif isinstance(block, dict):
                    content_for_storage.append(block)
            conversation.add_assistant(content_for_storage)
        
        return response
    
    async def chat_with_tools(
        self,
        message: str,
        tools: List[Tool],
        tool_handlers: Dict[str, Callable[..., Union[str, Awaitable[str]]]],
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        max_tokens: int = 8192,
        max_tool_rounds: int = 10,
        **kwargs,
    ) -> ChatResponse:
        """Chat with automatic tool execution.
        
        Supports both sync and async tool handlers.
        
        Args:
            message: The user message
            tools: List of available tools
            tool_handlers: Dict mapping tool names to handler functions
            model: Override the default model
            system: System prompt
            conversation: Conversation for multi-turn
            max_tokens: Maximum tokens per response
            max_tool_rounds: Maximum tool execution rounds
            **kwargs: Additional API parameters
            
        Returns:
            Final ChatResponse after all tool executions
            
        Example:
            >>> async def get_weather(location: str) -> str:
            ...     # Can be async!
            ...     return f"Weather in {location}: Sunny, 72°F"
            >>> 
            >>> tools = [Tool.create("get_weather", "Get weather", 
            ...     {"location": {"type": "string"}}, ["location"])]
            >>> handlers = {"get_weather": get_weather}
            >>> 
            >>> response = await client.chat_with_tools(
            ...     "What's the weather in Tokyo?",
            ...     tools=tools,
            ...     tool_handlers=handlers
            ... )
        """
        import asyncio
        
        response = await self.chat(
            message,
            model=model,
            system=system,
            conversation=conversation,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        
        rounds = 0
        while response.has_tool_calls and rounds < max_tool_rounds:
            rounds += 1
            logger.debug(f"Tool execution round {rounds}")
            
            tool_results = []
            for tool_call in response.tool_calls:
                handler = tool_handlers.get(tool_call.name)
                if handler is None:
                    result = f"Error: Unknown tool '{tool_call.name}'"
                    is_error = True
                else:
                    try:
                        # Support both sync and async handlers
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(**tool_call.input)
                        else:
                            result = handler(**tool_call.input)
                        is_error = False
                    except Exception as e:
                        result = f"Error executing {tool_call.name}: {str(e)}"
                        is_error = True
                        logger.exception(f"Tool execution error: {tool_call.name}")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": str(result),
                    "is_error": is_error,
                })
            
            if conversation:
                conversation.add_message(Message(
                    role=Role.USER,
                    content=tool_results
                ))
                msgs = conversation.to_messages_list()
            else:
                msgs = [{"role": "user", "content": message}]
                assistant_content = [block.model_dump() for block in response.content]
                msgs.append({"role": "assistant", "content": assistant_content})
                msgs.append({"role": "user", "content": tool_results})
            
            response = await self._make_request(
                messages=msgs,
                model=model,
                system=system or (conversation.system if conversation else self.default_system),
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            )
            
            if conversation:
                content_for_storage = [
                    block.model_dump() if hasattr(block, "model_dump") else block
                    for block in response.content
                ]
                conversation.add_assistant(content_for_storage)
        
        return response
    
    async def stream(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token.
        
        Args:
            message: The user message
            model: Override the default model
            system: System prompt
            max_tokens: Maximum tokens in response
            **kwargs: Additional API parameters
            
        Yields:
            Text chunks as they arrive
            
        Example:
            >>> async for chunk in client.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        client = await self._get_client()
        payload = self._build_payload(
            messages=[{"role": "user", "content": message}],
            model=model,
            system=system or self.default_system,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to proxy at {self.base_url}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out") from e
    
    async def stream_full(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream with full event data including thinking blocks.
        
        Yields dictionaries with type and content:
        - {"type": "text", "text": "..."}
        - {"type": "thinking", "thinking": "..."}
        - {"type": "tool_use", "tool": {...}}
        - {"type": "done", "usage": {...}}
        """
        client = await self._get_client()
        payload = self._build_payload(
            messages=[{"role": "user", "content": message}],
            model=model,
            system=system or self.default_system,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        current_block_type = None
        
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)
                
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(data)
                        event_type = event.get("type")
                        
                        if event_type == "content_block_start":
                            block = event.get("content_block", {})
                            current_block_type = block.get("type")
                        
                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            delta_type = delta.get("type")
                            
                            if delta_type == "text_delta":
                                yield {"type": "text", "text": delta.get("text", "")}
                            elif delta_type == "thinking_delta":
                                yield {"type": "thinking", "thinking": delta.get("thinking", "")}
                            elif delta_type == "input_json_delta":
                                yield {"type": "tool_input", "partial": delta.get("partial_json", "")}
                        
                        elif event_type == "message_delta":
                            usage = event.get("usage")
                            if usage:
                                yield {"type": "done", "usage": usage}
                    
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to proxy at {self.base_url}") from e
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Stream timed out") from e
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def create_conversation(
        self,
        system: Optional[str] = None,
        **kwargs,
    ) -> Conversation:
        """Create a new conversation.
        
        Args:
            system: System prompt for this conversation
            **kwargs: Additional Conversation arguments
            
        Returns:
            New Conversation instance
        """
        return Conversation(
            system=system or self.default_system,
            **kwargs,
        )
    
    def get_conversation(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
    ) -> Optional[Conversation]:
        """Get an existing conversation from the store."""
        return self._conversation_store.get(user_id=user_id, chat_id=chat_id, key=key)
    
    def get_or_create_conversation(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Conversation:
        """Get or create a conversation in the store.
        
        Useful for Telegram bots managing multiple users.
        """
        return self._conversation_store.get_or_create(
            user_id=user_id,
            chat_id=chat_id,
            key=key,
            system=system or self.default_system,
        )
    
    def clear_conversation(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bool:
        """Clear a conversation's history."""
        return self._conversation_store.clear(user_id=user_id, chat_id=chat_id)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    async def health(self) -> bool:
        """Check if the proxy is healthy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """Get list of available models from the proxy."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [m.get("id") for m in data.get("data", [])]
        except Exception:
            pass
        return AvailableModels.all()
    
    async def close(self):
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    def __repr__(self) -> str:
        return f"AsyncAntigravityClient(base_url={self.base_url!r}, model={self.model!r})"
