#!/usr/bin/env python3
"""
Basic usage examples for the Antigravity SDK.

This file demonstrates the core functionality of the SDK including:
- Simple chat
- System prompts
- Streaming responses
- Multi-turn conversations
- Tool calling
"""

from antigravity_sdk import (
    AntigravityClient,
    Tool,
    Conversation,
    AvailableModels,
)


def example_simple_chat():
    """Basic single-turn chat."""
    print("=" * 60)
    print("Example: Simple Chat")
    print("=" * 60)
    
    client = AntigravityClient()
    
    response = client.chat("What is the capital of France?")
    
    print(f"Response: {response.text}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
    print()


def example_with_system_prompt():
    """Chat with a system prompt."""
    print("=" * 60)
    print("Example: System Prompt")
    print("=" * 60)
    
    client = AntigravityClient()
    
    response = client.chat(
        "Explain quantum entanglement",
        system="You are a science teacher for 5th graders. Use simple words and fun analogies."
    )
    
    print(f"Response: {response.text}")
    print()


def example_streaming():
    """Stream responses token by token."""
    print("=" * 60)
    print("Example: Streaming")
    print("=" * 60)
    
    client = AntigravityClient()
    
    print("Streaming: ", end="", flush=True)
    for chunk in client.stream("Write a haiku about programming"):
        print(chunk, end="", flush=True)
    print("\n")


def example_streaming_with_metadata():
    """Stream with full response metadata at the end."""
    print("=" * 60)
    print("Example: Streaming with Metadata")
    print("=" * 60)
    
    client = AntigravityClient()
    
    print("Streaming: ", end="", flush=True)
    response = client.stream_full("Tell me a joke about AI")
    
    for chunk in response:
        print(chunk, end="", flush=True)
    print()
    
    # Get the full response after streaming completes
    full_response = response.get_response()
    print(f"Total tokens: {full_response.usage.total_tokens}")
    print()


def example_multi_turn_conversation():
    """Multi-turn conversation with memory."""
    print("=" * 60)
    print("Example: Multi-Turn Conversation")
    print("=" * 60)
    
    client = AntigravityClient()
    conv = client.create_conversation(system="You are a helpful math tutor.")
    
    # First turn
    response1 = client.chat("What is 15% of 80?", conversation=conv)
    print(f"User: What is 15% of 80?")
    print(f"Assistant: {response1.text}")
    
    # Second turn - references first
    response2 = client.chat("How did you calculate that?", conversation=conv)
    print(f"\nUser: How did you calculate that?")
    print(f"Assistant: {response2.text}")
    
    # Third turn - follow-up
    response3 = client.chat("What if it was 20% instead?", conversation=conv)
    print(f"\nUser: What if it was 20% instead?")
    print(f"Assistant: {response3.text}")
    
    print(f"\nConversation has {conv.message_count} messages")
    print()


def example_tool_calling():
    """Using tools for function calling."""
    print("=" * 60)
    print("Example: Tool Calling")
    print("=" * 60)
    
    # Define a weather tool
    weather_tool = Tool.create(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city and country, e.g., 'Tokyo, Japan'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        required=["location"]
    )
    
    # Tool handler function
    def get_weather(location: str, unit: str = "celsius") -> str:
        # In reality, this would call a weather API
        return f"Weather in {location}: Sunny, 22°{'C' if unit == 'celsius' else 'F'}"
    
    client = AntigravityClient()
    
    # Let the SDK handle tool execution automatically
    response = client.chat_with_tools(
        "What's the weather like in Paris?",
        tools=[weather_tool],
        tool_handlers={"get_weather": get_weather}
    )
    
    print(f"Response: {response.text}")
    print()


def example_calculator_tool():
    """Calculator tool example."""
    print("=" * 60)
    print("Example: Calculator Tool")
    print("=" * 60)
    
    calc_tool = Tool.create(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g., '2 + 2 * 3'"
            }
        },
        required=["expression"]
    )
    
    def calculate(expression: str) -> str:
        try:
            # WARNING: eval is dangerous! Use a proper math parser in production
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    client = AntigravityClient()
    
    response = client.chat_with_tools(
        "What is 15% of 200 plus 50?",
        tools=[calc_tool],
        tool_handlers={"calculate": calculate}
    )
    
    print(f"Response: {response.text}")
    print()


def example_different_models():
    """Using different models."""
    print("=" * 60)
    print("Example: Different Models")
    print("=" * 60)
    
    client = AntigravityClient()
    
    # Check available models
    print(f"Available Claude models: {AvailableModels.CLAUDE_MODELS}")
    print(f"Available Gemini models: {AvailableModels.GEMINI_MODELS}")
    
    # Use a specific model
    response = client.chat(
        "Hello!",
        model="claude-sonnet-4-5-thinking"
    )
    print(f"\nUsed model: {response.model}")
    print()


def example_error_handling():
    """Handling errors gracefully."""
    print("=" * 60)
    print("Example: Error Handling")
    print("=" * 60)
    
    from antigravity_sdk import (
        AntigravityError,
        RateLimitError,
        AuthenticationError,
    )
    
    client = AntigravityClient(
        base_url="http://localhost:8080"  # Assumes local proxy
    )
    
    try:
        response = client.chat("Hello!")
        print(f"Response: {response.text}")
    except RateLimitError as e:
        print(f"Rate limited! Retry after {e.retry_after} seconds")
    except AuthenticationError as e:
        print(f"Auth failed: {e}")
    except AntigravityError as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    print()


def example_conversation_export():
    """Exporting and importing conversations."""
    print("=" * 60)
    print("Example: Conversation Export/Import")
    print("=" * 60)
    
    client = AntigravityClient()
    
    # Create and populate a conversation
    conv = client.create_conversation(system="You are helpful.")
    conv.add_user("Hello!")
    conv.add_assistant("Hi there! How can I help?")
    
    # Export to JSON
    exported = conv.export_json()
    print(f"Exported: {exported[:100]}...")
    
    # Import into a new conversation
    new_conv = Conversation.from_json(exported)
    print(f"Imported conversation with {new_conv.message_count} messages")
    print()


if __name__ == "__main__":
    print("\n🚀 Antigravity SDK - Basic Usage Examples\n")
    
    # Uncomment the examples you want to run:
    
    example_simple_chat()
    example_with_system_prompt()
    example_streaming()
    example_streaming_with_metadata()
    example_multi_turn_conversation()
    example_tool_calling()
    example_calculator_tool()
    example_different_models()
    example_error_handling()
    example_conversation_export()
    
    print("✅ All examples completed!")
