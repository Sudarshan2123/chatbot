from typing import List, Dict
from src.logging import logger
from src.utils.mongo_ops import execute_query
from src.utils.session_store import SessionStore

class ConversationManager:
    """Manages conversation context for follow-up queries"""

    def __init__(self, config):
        self.config = config
        self.max_context_messages = 6  # Last 3 exchanges (user + AI)
        self.session_store = SessionStore(config)

    def get_conversation_context(self, user_id: str) -> List[Dict[str, str]]:
        """Get recent conversation history for context (Redis first, MongoDB fallback)"""
        try:
            # Try Redis first for fast access
            context = self.session_store.get_conversation_context(user_id, self.max_context_messages)
            if context:
                return context

            # Fallback to MongoDB
            history = execute_query(
                query="find",
                params={"user": user_id},
                db_name=self.config.DB_NAME,
                collection_name=self.config.HISTORY_COLLECTION_Logs,
                fetch="all"
            )

            if not history:
                return []

            # Sort by timestamp and get recent messages
            sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
            recent_history = sorted_history[:self.max_context_messages//2]  # Get last 3 exchanges

            # Format for LLM context
            context = []
            for msg in reversed(recent_history):  # Reverse to chronological order
                context.extend([
                    {"role": "user", "content": msg.get("query", "")},
                    {"role": "assistant", "content": msg.get("query_response", "")}
                ])

            return context[-self.max_context_messages:]  # Limit total messages

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return []

    def build_context_prompt(self, user_input: str, context: List[Dict[str, str]]) -> str:
        """Build prompt with conversation context"""
        if not context:
            return user_input

        context_str = "Previous conversation:\n"
        for msg in context:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_str += f"{role}: {msg['content']}\n"

        return f"{context_str}\nCurrent question: {user_input}"

    def store_conversation_turn(self, user_id: str, user_message: str, ai_response: str):
        """Store conversation turn in Redis for fast future access"""
        self.session_store.store_conversation_turn(user_id, user_message, ai_response)