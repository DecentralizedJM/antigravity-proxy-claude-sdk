"""
Antigravity SDK - Synchronous Client

Production-ready synchronous client with retry logic, tool support,
conversation management, and comprehensive error handling.
"""

import os
import json
import logging
from typing import List, Optional, Iterator, Dict, Any, Callable, Union

import httpx

from .models import (
    Message, ChatResponse, ContentBlock, TextBlock, ThinkingBlock,
    ToolUseBlock, Tool, Usage, AvailableModels, Role,
)
from .exceptions import (
    AntigravityError, ConnectionError, TimeoutError, StreamError,
    raise_for_status,
)
from .retry import RetryConfig, DEFAULT_RETRY_CONFIG, retry_sync
from .conversation import Conversation, ConversationStore

logger = logging.getLogger(__name__)


class AntigravityClient:
    """Synchronous client for the Antigravity Claude Proxy.
    
    Features:
    - Automatic retry with exponential backoff
    - Tool/function calling support
    - Conversation history management
    - Streaming responses
    - Comprehensive error handling
    
    Example:
        >>> client = AntigravityClient()
        >>> response = client.chat("Hello, Claude!")
        >>> print(response.text)
        
        # With conversation history
        >>> conv = client.create_conversation(system="You are helpful.")
        >>> response = client.chat("Hi!", conversation=conv)
        >>> response = client.chat("What did I just say?", conversation=conv)
        
        # With tools
        >>> tools = [Tool.create("get_time", "Get current time", {})]
        >>> response = client.chat("What time is it?", tools=tools)
        >>> if response.has_tool_calls:
        ...     # Handle tool calls
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
        """Initialize the client.
        
        Args:
            base_url: Proxy URL. Defaults to ANTIGRAVITY_PROXY_URL env var or localhost:8080
            model: Default model to use. Defaults to ANTIGRAVITY_MODEL env var
            timeout: Request timeout in seconds (default 300s for long responses)
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
        
        # HTTP client
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
        )
        
        # Conversation store for multi-turn chats
        self._conversation_store = ConversationStore(
            default_system=default_system
        )
    
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
        
        # Add any extra parameters
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
        
        # Parse usage if present
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
    
    def _make_request(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> ChatResponse:
        """Make a non-streaming API request with retry logic."""
        payload = self._build_payload(messages, stream=False, **kwargs)
        
        @retry_sync(self.retry_config)
        def do_request() -> ChatResponse:
            try:
                response = self._client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.ConnectError as e:
                raise ConnectionError(f"Failed to connect to proxy at {self.base_url}") from e
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Request timed out after {self.timeout}s") from e
            
            # Parse response body
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"error": response.text}
            
            # Check for errors
            if response.status_code >= 400:
                raise_for_status(response.status_code, data)
            
            return self._parse_response(data)
        
        return do_request()
    
    def chat(
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
            >>> response = client.chat("Hello!")
            >>> print(response.text)
            
            >>> # With thinking model
            >>> response = client.chat("Solve this puzzle...", 
            ...     model="claude-opus-4-5-thinking")
            >>> print(f"Thinking: {response.thinking}")
            >>> print(f"Answer: {response.text}")
        """
        # Determine system prompt
        effective_system = system
        if effective_system is None and conversation:
            effective_system = conversation.system
        if effective_system is None:
            effective_system = self.default_system
        
        # Build messages list
        if conversation:
            # Add user message to conversation
            conversation.add_user(message)
            msgs = conversation.to_messages_list()
        elif messages:
            msgs = [{"role": m.role, "content": m.content} for m in messages]
            msgs.append({"role": "user", "content": message})
        else:
            msgs = [{"role": "user", "content": message}]
        
        # Make request
        response = self._make_request(
            messages=msgs,
            model=model,
            system=effective_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )
        
        # Update conversation with response
        if conversation:
            # Convert content blocks to dict format for storage
            content_for_storage = []
            for block in response.content:
                if hasattr(block, "model_dump"):
                    content_for_storage.append(block.model_dump())
                elif isinstance(block, dict):
                    content_for_storage.append(block)
            conversation.add_assistant(content_for_storage)
        
        return response
    
    def chat_with_tools(
        self,
        message: str,
        tools: List[Tool],
        tool_handlers: Dict[str, Callable[..., str]],
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        conversation: Optional[Conversation] = None,
        max_tokens: int = 8192,
        max_tool_rounds: int = 10,
        **kwargs,
    ) -> ChatResponse:
        """Chat with automatic tool execution.
        
        Handles the tool use loop automatically:
        1. Send message
        2. If model requests tools, execute them
        3. Send results back
        4. Repeat until model gives final response
        
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
            >>> def get_weather(location: str) -> str:
            ...     return f"Weather in {location}: Sunny, 72°F"
            >>> 
            >>> tools = [Tool.create("get_weather", "Get weather", 
            ...     {"location": {"type": "string"}}, ["location"])]
            >>> handlers = {"get_weather": get_weather}
            >>> 
            >>> response = client.chat_with_tools(
            ...     "What's the weather in Tokyo?",
            ...     tools=tools,
            ...     tool_handlers=handlers
            ... )
            >>> print(response.text)  # "The weather in Tokyo is Sunny, 72°F"
        """
        # Initial request
        response = self.chat(
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
            
            # Execute each tool call
            tool_results = []
            for tool_call in response.tool_calls:
                handler = tool_handlers.get(tool_call.name)
                if handler is None:
                    result = f"Error: Unknown tool '{tool_call.name}'"
                    is_error = True
                else:
                    try:
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
            
            # Add tool results to conversation if using one
            if conversation:
                conversation.add_message(Message(
                    role=Role.USER,
                    content=tool_results
                ))
                msgs = conversation.to_messages_list()
            else:
                # Build messages with tool results
                msgs = [{"role": "user", "content": message}]
                # Add assistant response with tool calls
                assistant_content = [block.model_dump() for block in response.content]
                msgs.append({"role": "assistant", "content": assistant_content})
                # Add tool results
                msgs.append({"role": "user", "content": tool_results})
            
            # Continue the conversation
            response = self._make_request(
                messages=msgs,
                model=model,
                system=system or (conversation.system if conversation else self.default_system),
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            )
            
            # Update conversation with new response
            if conversation:
                content_for_storage = [
                    block.model_dump() if hasattr(block, "model_dump") else block
                    for block in response.content
                ]
                conversation.add_assistant(content_for_storage)
        
        return response
    
    def stream(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        **kwargs,
    ) -> Iterator[str]:
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
            >>> for chunk in client.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        payload = self._build_payload(
            messages=[{"role": "user", "content": message}],
            model=model,
            system=system or self.default_system,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    # Read the error response
                    response.read()
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)
                
                for line in response.iter_lines():
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
    
    def stream_full(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """Stream with full event data including thinking blocks.
        
        Yields dictionaries with type and content:
        - {"type": "text", "text": "..."}
        - {"type": "thinking", "thinking": "..."}
        - {"type": "tool_use", "tool": {...}}
        - {"type": "done", "usage": {...}}
        """
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
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"error": response.text}
                    raise_for_status(response.status_code, data)
                
                for line in response.iter_lines():
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
        """Get an existing conversation from the store.
        
        Args:
            user_id: User identifier
            chat_id: Chat/group identifier
            key: Direct key
            
        Returns:
            Conversation if found, None otherwise
        """
        return self._conversation_store.get(user_id=user_id, chat_id=chat_id, key=key)
    
    def get_or_create_conversation(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Conversation:
        """Get or create a conversation in the store.
        
        Useful for bot applications managing multiple users.
        
        Args:
            user_id: User identifier
            chat_id: Chat/group identifier  
            key: Direct key
            system: System prompt for new conversations
            
        Returns:
            Existing or new Conversation
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
        """Clear a conversation's history.
        
        Returns:
            True if cleared, False if not found
        """
        return self._conversation_store.clear(user_id=user_id, chat_id=chat_id)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def health(self) -> bool:
        """Check if the proxy is healthy.
        
        Returns:
            True if proxy is reachable and healthy
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models from the proxy.
        
        Returns:
            List of model identifiers
        """
        try:
            response = self._client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [m.get("id") for m in data.get("data", [])]
        except Exception:
            pass
        
        # Fallback to known models
        return AvailableModels.all()
    
    def close(self):
        """Close the HTTP client and release resources."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self) -> str:
        return f"AntigravityClient(base_url={self.base_url!r}, model={self.model!r})"
