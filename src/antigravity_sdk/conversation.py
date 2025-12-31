"""
Antigravity SDK - Conversation Manager

Manages multi-turn conversations with automatic history tracking,
token estimation, and context window management.
"""

import hashlib
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Message, Role, ContentBlock, TextBlock, ToolUseBlock, ToolResultBlock


@dataclass
class ConversationConfig:
    """Configuration for conversation management.
    
    Attributes:
        max_messages: Maximum messages to keep in history (default: 50)
        max_tokens_estimate: Estimated max tokens for context (default: 100000)
        chars_per_token: Approximate characters per token (default: 4)
        include_system_in_history: Whether to track system prompts (default: False)
        auto_trim: Automatically trim history when limits exceeded (default: True)
    """
    max_messages: int = 50
    max_tokens_estimate: int = 100000
    chars_per_token: int = 4
    include_system_in_history: bool = False
    auto_trim: bool = True


class Conversation:
    """Manages a multi-turn conversation with history.
    
    Handles:
    - Message history tracking
    - Automatic context window management
    - Tool use/result pairing
    - Conversation persistence (via export/import)
    
    Example:
        >>> conv = Conversation(system="You are a helpful assistant.")
        >>> conv.add_user("Hello!")
        >>> response = client.chat_with_conversation(conv)
        >>> print(response.text)
        >>> # Response is automatically added to history
    """
    
    def __init__(
        self,
        system: Optional[str] = None,
        config: Optional[ConversationConfig] = None,
        conversation_id: Optional[str] = None,
    ):
        """Initialize a conversation.
        
        Args:
            system: System prompt for the conversation
            config: Conversation configuration
            conversation_id: Unique ID (auto-generated if not provided)
        """
        self.system = system
        self.config = config or ConversationConfig()
        self.conversation_id = conversation_id or self._generate_id()
        self._messages: List[Message] = []
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at
    
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique conversation ID."""
        import uuid
        return f"conv_{uuid.uuid4().hex[:12]}"
    
    @property
    def messages(self) -> List[Message]:
        """Get the message history."""
        return self._messages.copy()
    
    @property
    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self._messages)
    
    @property
    def is_empty(self) -> bool:
        """Check if conversation has no messages."""
        return len(self._messages) == 0
    
    @property
    def estimated_tokens(self) -> int:
        """Estimate the token count of the conversation."""
        total_chars = 0
        
        # Count system prompt
        if self.system:
            total_chars += len(self.system)
        
        # Count messages
        for msg in self._messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        total_chars += len(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        total_chars += len(block["text"])
        
        return total_chars // self.config.chars_per_token
    
    def _update_timestamp(self):
        """Update the last modified timestamp."""
        self._updated_at = datetime.now(timezone.utc)
    
    def add_user(self, content: str) -> "Conversation":
        """Add a user message to the conversation.
        
        Args:
            content: The user's message text
        
        Returns:
            Self for chaining
        """
        self._messages.append(Message.user(content))
        self._update_timestamp()
        self._maybe_trim()
        return self
    
    def add_assistant(self, content: Union[str, List[ContentBlock]]) -> "Conversation":
        """Add an assistant message to the conversation.
        
        Args:
            content: The assistant's response (text or content blocks)
        
        Returns:
            Self for chaining
        """
        self._messages.append(Message(role=Role.ASSISTANT, content=content))
        self._update_timestamp()
        self._maybe_trim()
        return self
    
    def add_tool_result(
        self,
        tool_use_id: str,
        result: str,
        is_error: bool = False
    ) -> "Conversation":
        """Add a tool result message.
        
        Args:
            tool_use_id: The ID of the tool use being responded to
            result: The tool execution result
            is_error: Whether the tool execution failed
        
        Returns:
            Self for chaining
        """
        self._messages.append(Message.tool_result(tool_use_id, result, is_error))
        self._update_timestamp()
        return self
    
    def add_message(self, message: Message) -> "Conversation":
        """Add a pre-constructed message.
        
        Args:
            message: The message to add
        
        Returns:
            Self for chaining
        """
        self._messages.append(message)
        self._update_timestamp()
        self._maybe_trim()
        return self
    
    def _maybe_trim(self):
        """Trim history if it exceeds limits."""
        if not self.config.auto_trim:
            return
        
        # Trim by message count
        if len(self._messages) > self.config.max_messages:
            # Keep most recent messages, but ensure we don't break tool use pairs
            excess = len(self._messages) - self.config.max_messages
            self._messages = self._safe_trim(self._messages, excess)
        
        # Trim by estimated tokens
        while (
            self.estimated_tokens > self.config.max_tokens_estimate
            and len(self._messages) > 2
        ):
            self._messages = self._safe_trim(self._messages, 2)
    
    def _safe_trim(self, messages: List[Message], count: int) -> List[Message]:
        """Safely trim messages without breaking tool use/result pairs.
        
        Args:
            messages: List of messages
            count: Number of messages to remove from the start
        
        Returns:
            Trimmed message list
        """
        if count <= 0:
            return messages
        
        # Find a safe trim point (don't split tool use from result)
        trim_index = count
        
        for i in range(min(count, len(messages))):
            msg = messages[i]
            if isinstance(msg.content, list):
                # Check if this contains tool use - if so, include next message too
                has_tool_use = any(
                    isinstance(block, ToolUseBlock) or 
                    (isinstance(block, dict) and block.get("type") == "tool_use")
                    for block in msg.content
                )
                if has_tool_use and i + 1 < len(messages):
                    trim_index = max(trim_index, i + 2)
        
        return messages[trim_index:]
    
    def clear(self) -> "Conversation":
        """Clear all messages from the conversation.
        
        Returns:
            Self for chaining
        """
        self._messages.clear()
        self._update_timestamp()
        return self
    
    def fork(self, system: Optional[str] = None) -> "Conversation":
        """Create a copy of this conversation.
        
        Args:
            system: New system prompt (uses original if not provided)
        
        Returns:
            New Conversation instance with copied history
        """
        new_conv = Conversation(
            system=system or self.system,
            config=self.config,
        )
        new_conv._messages = self._messages.copy()
        return new_conv
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for msg in reversed(self._messages):
            if msg.role == Role.USER:
                if isinstance(msg.content, str):
                    return msg.content
                # Check for text in content blocks
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        return block.text
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message text."""
        for msg in reversed(self._messages):
            if msg.role == Role.ASSISTANT:
                if isinstance(msg.content, str):
                    return msg.content
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        return block.text
        return None
    
    def to_messages_list(self) -> List[Dict[str, Any]]:
        """Convert conversation to API message format.
        
        Returns:
            List of message dictionaries for API calls
        """
        result = []
        for msg in self._messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Convert content blocks to dicts
                content_list = []
                for block in msg.content:
                    if hasattr(block, "model_dump"):
                        content_list.append(block.model_dump())
                    elif isinstance(block, dict):
                        content_list.append(block)
                result.append({"role": msg.role, "content": content_list})
        return result
    
    def export(self) -> Dict[str, Any]:
        """Export conversation to a serializable dictionary.
        
        Useful for saving conversations to disk or database.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "conversation_id": self.conversation_id,
            "system": self.system,
            "messages": self.to_messages_list(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
        }
    
    @classmethod
    def from_export(
        cls,
        data: Dict[str, Any],
        config: Optional[ConversationConfig] = None
    ) -> "Conversation":
        """Create a conversation from exported data.
        
        Args:
            data: Exported conversation dictionary
            config: Optional configuration override
        
        Returns:
            Restored Conversation instance
        """
        conv = cls(
            system=data.get("system"),
            config=config,
            conversation_id=data.get("conversation_id"),
        )
        
        # Restore messages
        for msg_data in data.get("messages", []):
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")
            conv._messages.append(Message(role=Role(role), content=content))
        
        # Restore timestamps if available
        if "created_at" in data:
            conv._created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            conv._updated_at = datetime.fromisoformat(data["updated_at"])
        
        return conv
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __repr__(self) -> str:
        return (
            f"Conversation(id={self.conversation_id!r}, "
            f"messages={len(self._messages)}, "
            f"tokens≈{self.estimated_tokens})"
        )


class ConversationStore:
    """Simple in-memory store for multiple conversations.
    
    Useful for bots managing conversations with multiple users.
    
    Example:
        >>> store = ConversationStore()
        >>> conv = store.get_or_create(user_id=12345)
        >>> conv.add_user("Hello!")
    """
    
    def __init__(
        self,
        default_system: Optional[str] = None,
        config: Optional[ConversationConfig] = None,
        max_conversations: int = 1000,
    ):
        """Initialize the conversation store.
        
        Args:
            default_system: Default system prompt for new conversations
            config: Default configuration for new conversations
            max_conversations: Maximum conversations to keep (LRU eviction)
        """
        self.default_system = default_system
        self.config = config or ConversationConfig()
        self.max_conversations = max_conversations
        self._conversations: Dict[str, Conversation] = {}
    
    def _make_key(self, user_id: Optional[int] = None, chat_id: Optional[int] = None) -> str:
        """Create a unique key for a conversation."""
        parts = []
        if user_id is not None:
            parts.append(f"user:{user_id}")
        if chat_id is not None:
            parts.append(f"chat:{chat_id}")
        return ":".join(parts) if parts else "default"
    
    def get(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
    ) -> Optional[Conversation]:
        """Get an existing conversation.
        
        Args:
            user_id: User identifier
            chat_id: Chat/group identifier
            key: Direct key (overrides user_id/chat_id)
        
        Returns:
            Conversation if found, None otherwise
        """
        k = key or self._make_key(user_id, chat_id)
        return self._conversations.get(k)
    
    def get_or_create(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Conversation:
        """Get an existing conversation or create a new one.
        
        Args:
            user_id: User identifier
            chat_id: Chat/group identifier
            key: Direct key (overrides user_id/chat_id)
            system: System prompt (uses default if not provided)
        
        Returns:
            Existing or new Conversation
        """
        k = key or self._make_key(user_id, chat_id)
        
        if k not in self._conversations:
            # Evict oldest if at capacity
            if len(self._conversations) >= self.max_conversations:
                self._evict_oldest()
            
            self._conversations[k] = Conversation(
                system=system or self.default_system,
                config=self.config,
                conversation_id=k,
            )
        
        return self._conversations[k]
    
    def delete(
        self,
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
        key: Optional[str] = None,
    ) -> bool:
        """Delete a conversation.
        
        Returns:
            True if conversation was deleted, False if not found
        """
        k = key or self._make_key(user_id, chat_id)
        if k in self._conversations:
            del self._conversations[k]
            return True
        return False
    
    def clear(self, user_id: Optional[int] = None, chat_id: Optional[int] = None) -> bool:
        """Clear a conversation's messages (keeps the conversation).
        
        Returns:
            True if conversation was cleared, False if not found
        """
        conv = self.get(user_id=user_id, chat_id=chat_id)
        if conv:
            conv.clear()
            return True
        return False
    
    def _evict_oldest(self):
        """Remove the oldest conversation (by last update time)."""
        if not self._conversations:
            return
        
        oldest_key = min(
            self._conversations.keys(),
            key=lambda k: self._conversations[k]._updated_at
        )
        del self._conversations[oldest_key]
    
    @property
    def count(self) -> int:
        """Get the number of stored conversations."""
        return len(self._conversations)
    
    def __len__(self) -> int:
        return len(self._conversations)
