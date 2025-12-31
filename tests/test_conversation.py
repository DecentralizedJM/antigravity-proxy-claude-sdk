"""
Tests for conversation management.
"""

import pytest
from datetime import datetime, timezone

from antigravity_sdk.conversation import (
    Conversation,
    ConversationConfig,
    ConversationStore,
)
from antigravity_sdk.models import Message, Role, TextBlock, ToolUseBlock


class TestConversation:
    """Tests for Conversation class."""
    
    def test_creation(self):
        conv = Conversation(system="You are helpful.")
        assert conv.system == "You are helpful."
        assert conv.is_empty
        assert conv.message_count == 0
    
    def test_auto_generated_id(self):
        conv1 = Conversation()
        conv2 = Conversation()
        assert conv1.conversation_id != conv2.conversation_id
        assert conv1.conversation_id.startswith("conv_")
    
    def test_add_user_message(self):
        conv = Conversation()
        result = conv.add_user("Hello!")
        
        assert result is conv  # Returns self for chaining
        assert conv.message_count == 1
        assert not conv.is_empty
    
    def test_add_assistant_message(self):
        conv = Conversation()
        conv.add_user("Hello!")
        conv.add_assistant("Hi there!")
        
        assert conv.message_count == 2
    
    def test_add_tool_result(self):
        conv = Conversation()
        conv.add_user("What's the weather?")
        conv.add_assistant([
            TextBlock(text="Let me check."),
            ToolUseBlock(id="tool_1", name="get_weather", input={"location": "NYC"})
        ])
        conv.add_tool_result("tool_1", "Sunny, 72°F")
        
        assert conv.message_count == 3
    
    def test_get_last_messages(self):
        conv = Conversation()
        conv.add_user("Hello!")
        conv.add_assistant("Hi!")
        conv.add_user("How are you?")
        
        assert conv.get_last_user_message() == "How are you?"
        assert conv.get_last_assistant_message() == "Hi!"
    
    def test_clear(self):
        conv = Conversation()
        conv.add_user("Hello!")
        conv.add_assistant("Hi!")
        
        conv.clear()
        
        assert conv.is_empty
        assert conv.message_count == 0
    
    def test_fork(self):
        conv = Conversation(system="Original system")
        conv.add_user("Hello!")
        
        forked = conv.fork()
        
        assert forked.conversation_id != conv.conversation_id
        assert forked.system == "Original system"
        assert forked.message_count == 1
        
        # Modifications to fork don't affect original
        forked.add_user("Another message")
        assert conv.message_count == 1
        assert forked.message_count == 2
    
    def test_fork_with_new_system(self):
        conv = Conversation(system="Original")
        forked = conv.fork(system="New system")
        
        assert forked.system == "New system"
    
    def test_to_messages_list(self):
        conv = Conversation()
        conv.add_user("Hello!")
        conv.add_assistant("Hi there!")
        
        msgs = conv.to_messages_list()
        
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello!"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there!"}
    
    def test_estimated_tokens(self):
        conv = Conversation(system="You are a helpful assistant.")
        conv.add_user("Hello, how are you today?")
        conv.add_assistant("I'm doing well, thank you for asking!")
        
        tokens = conv.estimated_tokens
        assert tokens > 0
        # Rough estimate: ~20 words × ~1.5 tokens/word ÷ 4 chars/token
        assert tokens < 100
    
    def test_chaining(self):
        conv = (
            Conversation(system="Be concise")
            .add_user("Hello!")
            .add_assistant("Hi!")
            .add_user("Bye!")
        )
        
        assert conv.message_count == 3


class TestConversationConfig:
    """Tests for ConversationConfig."""
    
    def test_defaults(self):
        config = ConversationConfig()
        assert config.max_messages == 50
        assert config.auto_trim is True
    
    def test_custom_config(self):
        config = ConversationConfig(
            max_messages=20,
            max_tokens_estimate=50000,
        )
        assert config.max_messages == 20
        assert config.max_tokens_estimate == 50000


class TestConversationTrimming:
    """Tests for automatic conversation trimming."""
    
    def test_trim_by_message_count(self):
        config = ConversationConfig(max_messages=5, auto_trim=True)
        conv = Conversation(config=config)
        
        # Add more than max messages
        for i in range(10):
            conv.add_user(f"Message {i}")
        
        assert conv.message_count <= 5
    
    def test_no_trim_when_disabled(self):
        config = ConversationConfig(max_messages=5, auto_trim=False)
        conv = Conversation(config=config)
        
        for i in range(10):
            conv.add_user(f"Message {i}")
        
        assert conv.message_count == 10


class TestConversationExport:
    """Tests for conversation export/import."""
    
    def test_export(self):
        conv = Conversation(system="Test system")
        conv.add_user("Hello!")
        conv.add_assistant("Hi!")
        
        data = conv.export()
        
        assert data["system"] == "Test system"
        assert data["conversation_id"] == conv.conversation_id
        assert len(data["messages"]) == 2
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_import(self):
        original = Conversation(system="Test")
        original.add_user("Hello!")
        original.add_assistant("World!")
        
        exported = original.export()
        restored = Conversation.from_export(exported)
        
        assert restored.system == original.system
        assert restored.conversation_id == original.conversation_id
        assert restored.message_count == original.message_count


class TestConversationStore:
    """Tests for ConversationStore."""
    
    def test_get_or_create(self):
        store = ConversationStore(default_system="Default prompt")
        
        conv = store.get_or_create(user_id=123)
        
        assert conv.system == "Default prompt"
        assert store.count == 1
    
    def test_get_existing(self):
        store = ConversationStore()
        
        conv1 = store.get_or_create(user_id=123)
        conv1.add_user("Hello!")
        
        conv2 = store.get_or_create(user_id=123)
        
        assert conv1 is conv2
        assert conv2.message_count == 1
    
    def test_get_nonexistent(self):
        store = ConversationStore()
        
        conv = store.get(user_id=999)
        
        assert conv is None
    
    def test_different_users(self):
        store = ConversationStore()
        
        conv1 = store.get_or_create(user_id=1)
        conv2 = store.get_or_create(user_id=2)
        
        assert conv1 is not conv2
        assert store.count == 2
    
    def test_user_and_chat_key(self):
        store = ConversationStore()
        
        # Same user, different chats
        conv1 = store.get_or_create(user_id=1, chat_id=100)
        conv2 = store.get_or_create(user_id=1, chat_id=200)
        
        assert conv1 is not conv2
    
    def test_delete(self):
        store = ConversationStore()
        
        store.get_or_create(user_id=123)
        assert store.count == 1
        
        deleted = store.delete(user_id=123)
        
        assert deleted is True
        assert store.count == 0
        assert store.get(user_id=123) is None
    
    def test_clear(self):
        store = ConversationStore()
        
        conv = store.get_or_create(user_id=123)
        conv.add_user("Hello!")
        
        store.clear(user_id=123)
        
        # Conversation still exists but is empty
        conv = store.get(user_id=123)
        assert conv is not None
        assert conv.is_empty
    
    def test_max_conversations_eviction(self):
        store = ConversationStore(max_conversations=3)
        
        # Add 5 conversations
        for i in range(5):
            store.get_or_create(user_id=i)
        
        # Should only keep 3
        assert store.count == 3
    
    def test_custom_key(self):
        store = ConversationStore()
        
        conv = store.get_or_create(key="custom:key:123")
        
        retrieved = store.get(key="custom:key:123")
        assert retrieved is conv
