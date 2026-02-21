import redis
import json
import time
from typing import List, Dict, Any, Optional
from datetime import timedelta
from src.logging import logger

class SessionStore:
    """Redis-based session storage for fast conversation context retrieval"""
    
    def __init__(self, config):
        self.config = config
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                username=config.REDIS_USERNAME,
                password=config.REDIS_PASSWORD,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to MongoDB only.")
            self.redis_client = None
    
    def store_conversation_turn(self, user_id: str, user_message: str, ai_response: str):
        """Store a conversation turn in Redis for fast access"""
        if not self.redis_client:
            return
        
        try:
            key = f"chat_context:{user_id}"
            turn = {
                "user": user_message,
                "assistant": ai_response,
                "timestamp": str(int(time.time()))
            }
            
            # Add to list (keep last 12 turns for better context)
            self.redis_client.lpush(key, json.dumps(turn))
            self.redis_client.ltrim(key, 0, 11)  # Keep only last 12
            self.redis_client.expire(key, timedelta(hours=24))  # Expire after 24h
            
            # Also store session metadata
            self._update_session_metadata(user_id)
            
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
    
    def clear_conversation_context(self, user_id: str):
        """Clear conversation context for a specific user"""
        if not self.redis_client:
            return
        
        try:
            key = f"chat_context:{user_id}"
            session_key = f"session_meta:{user_id}"
            self.redis_client.delete(key)
            self.redis_client.delete(session_key)
            logger.info(f"Cleared conversation context for user: {user_id}")
        except Exception as e:
            logger.error(f"Error clearing conversation context: {e}")
    
    def _update_session_metadata(self, user_id: str):
        """Update session metadata for tracking"""
        try:
            session_key = f"session_meta:{user_id}"
            metadata = {
                "last_activity": str(int(time.time())),
                "session_start": self.redis_client.hget(session_key, "session_start") or str(int(time.time())),
                "message_count": str(int(self.redis_client.hget(session_key, "message_count") or 0) + 1)
            }
            
            self.redis_client.hset(session_key, mapping=metadata)
            self.redis_client.expire(session_key, timedelta(hours=24))
            
        except Exception as e:
            logger.error(f"Error updating session metadata: {e}")
    
    def get_session_info(self, user_id: str) -> Dict[str, Any]:
        """Get session information for a user"""
        if not self.redis_client:
            return {}
        
        try:
            session_key = f"session_meta:{user_id}"
            chat_key = f"chat_context:{user_id}"
            
            metadata = self.redis_client.hgetall(session_key)
            context_length = self.redis_client.llen(chat_key)
            
            return {
                "session_active": bool(metadata),
                "last_activity": metadata.get("last_activity"),
                "session_start": metadata.get("session_start"),
                "message_count": int(metadata.get("message_count", 0)),
                "context_length": context_length
            }
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {}
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session user IDs"""
        if not self.redis_client:
            return []
        
        try:
            pattern = "session_meta:*"
            keys = self.redis_client.keys(pattern)
            return [key.replace("session_meta:", "") for key in keys]
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def get_conversation_context(self, user_id: str, limit: int = 8) -> List[Dict[str, str]]:
        """Get conversation context from Redis"""
        if not self.redis_client:
            return []
        
        try:
            key = f"chat_context:{user_id}"
            turns = self.redis_client.lrange(key, 0, limit//2 - 1)
            
            context = []
            for turn_json in reversed(turns):  # Reverse for chronological order
                turn = json.loads(turn_json)
                context.extend([
                    {"role": "user", "content": turn["user"]},
                    {"role": "assistant", "content": turn["assistant"]}
                ])
            
            return context[-limit:]  # Return last N messages
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return []