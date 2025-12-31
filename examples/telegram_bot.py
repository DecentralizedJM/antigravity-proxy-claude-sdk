#!/usr/bin/env python3
"""
Telegram Bot Example using Antigravity SDK.

This example shows how to build a Claude-powered Telegram bot with:
- Async operations for high concurrency
- Per-user conversation memory
- Admin commands
- Error handling
- Rate limiting awareness

Requirements:
    pip install python-telegram-bot antigravity-proxy-sdk

Set environment variable:
    export TELEGRAM_BOT_TOKEN="your-token-here"
"""

import os
import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from antigravity_sdk import (
    AsyncAntigravityClient,
    RateLimitError,
    AntigravityError,
)
from antigravity_sdk.retry import RetryConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
PROXY_URL = os.environ.get("ANTIGRAVITY_PROXY_URL", "http://localhost:8080")
ADMIN_USER_IDS = [int(x) for x in os.environ.get("ADMIN_USER_IDS", "").split(",") if x]

# System prompt for the assistant
SYSTEM_PROMPT = """You are a friendly and helpful AI assistant in a Telegram chat.

Guidelines:
- Be concise since this is a chat interface
- Use emoji occasionally to be friendly 😊
- If you don't know something, admit it
- Never pretend to be human
- Keep responses under 500 words unless asked for detail

Remember: You're chatting with {user_name}."""

# Global client instance
claude: Optional[AsyncAntigravityClient] = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Hello {user.first_name}!\n\n"
        "I'm an AI assistant powered by Claude. Just send me a message and I'll help you!\n\n"
        "Commands:\n"
        "/clear - Clear conversation history\n"
        "/help - Show this help message"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "🤖 *AI Assistant Help*\n\n"
        "Just send me any message and I'll respond!\n\n"
        "*Commands:*\n"
        "/start - Welcome message\n"
        "/clear - Clear our conversation history\n"
        "/help - This help message\n\n"
        "*Tips:*\n"
        "• I remember our conversation context\n"
        "• Be specific for better answers\n"
        "• Ask follow-up questions!",
        parse_mode="Markdown"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command - resets conversation."""
    user_id = update.effective_user.id
    
    if claude.clear_conversation(user_id=user_id):
        await update.message.reply_text(
            "🧹 Conversation cleared! Starting fresh."
        )
    else:
        await update.message.reply_text(
            "No conversation to clear."
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command (admin only)."""
    user_id = update.effective_user.id
    
    if user_id not in ADMIN_USER_IDS:
        await update.message.reply_text("⛔ Admin only command.")
        return
    
    # Get conversation store stats
    store = claude._conversation_store
    await update.message.reply_text(
        f"📊 *Bot Statistics*\n\n"
        f"Active conversations: {len(store._conversations)}\n"
        f"Max conversations: {store._max_conversations}",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    user = update.effective_user
    user_message = update.message.text
    
    logger.info(f"Message from {user.id} ({user.first_name}): {user_message[:50]}...")
    
    # Show typing indicator
    await update.message.chat.send_action("typing")
    
    try:
        # Get or create conversation for this user
        conversation = claude.get_or_create_conversation(
            user_id=user.id,
            system=SYSTEM_PROMPT.format(user_name=user.first_name)
        )
        
        # Send to Claude
        response = await claude.chat(
            user_message,
            conversation=conversation
        )
        
        # Reply to user
        await update.message.reply_text(response.text)
        
        logger.info(f"Response to {user.id}: {len(response.text)} chars, "
                   f"{response.usage.total_tokens if response.usage else '?'} tokens")
        
    except RateLimitError as e:
        wait_time = e.retry_after or 60
        await update.message.reply_text(
            f"⏳ I'm a bit busy right now. Please try again in {wait_time} seconds."
        )
        logger.warning(f"Rate limited: retry after {wait_time}s")
        
    except AntigravityError as e:
        await update.message.reply_text(
            "😅 Sorry, I encountered an error. Please try again."
        )
        logger.error(f"API error: {e}")
        
    except Exception as e:
        await update.message.reply_text(
            "❌ Something went wrong. Please try again later."
        )
        logger.exception(f"Unexpected error: {e}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")


async def post_init(application: Application) -> None:
    """Initialize the Claude client after the app starts."""
    global claude
    
    # Create async client with custom retry config
    claude = AsyncAntigravityClient(
        base_url=PROXY_URL,
        model="claude-sonnet-4-5-thinking",
        max_tokens=1024,  # Keep responses concise for chat
        retry_config=RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0
        ),
        conversation_max_messages=20,  # Keep last 20 messages per user
    )
    
    # Check proxy health
    if not await claude.health():
        logger.warning("⚠️ Antigravity proxy is not responding!")
    else:
        logger.info("✅ Connected to Antigravity proxy")


async def post_shutdown(application: Application) -> None:
    """Cleanup on shutdown."""
    global claude
    if claude:
        await claude.close()
        logger.info("Closed Claude client")


def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        print("❌ Error: Set TELEGRAM_BOT_TOKEN environment variable")
        return
    
    print(f"🚀 Starting Telegram bot...")
    print(f"📡 Proxy URL: {PROXY_URL}")
    
    # Create application
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Run the bot
    print("✅ Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
