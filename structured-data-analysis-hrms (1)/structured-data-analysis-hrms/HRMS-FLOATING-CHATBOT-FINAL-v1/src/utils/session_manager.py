from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timedelta
from src.logging import logger
from src.utils.session_store import SessionStore

class SessionManager:
    """Advanced session management for conversation linking"""
    
    def __init__(self, config):
        self.config = config
        self.session_store = SessionStore(config)
        self.session_timeout = 3600  # 1 hour in seconds
    
    def start_session(self, user_id: str) -> bool:
        """Start a new session for user"""
        try:
            # Clear any existing context
            self.session_store.clear_conversation_context(user_id)
            
            # Initialize session metadata
            self.session_store._update_session_metadata(user_id)
            
            logger.info(f"Started new session for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return False
    
    def is_session_active(self, user_id: str) -> bool:
        """Check if user has an active session"""
        try:
            session_info = self.session_store.get_session_info(user_id)
            if not session_info.get('session_active'):
                return False
            
            # Check if session has timed out
            last_activity = session_info.get('last_activity')
            if last_activity:
                time_diff = time.time() - int(last_activity)
                return time_diff < self.session_timeout
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking session status: {e}")
            return False
    
    def extend_session(self, user_id: str) -> bool:
        """Extend session timeout"""
        try:
            if self.is_session_active(user_id):
                self.session_store._update_session_metadata(user_id)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error extending session: {e}")
            return False
    
    def end_session(self, user_id: str) -> bool:
        """End user session and clear context"""
        try:
            self.session_store.clear_conversation_context(user_id)
            logger.info(f"Ended session for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    def get_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed session statistics"""
        try:
            session_info = self.session_store.get_session_info(user_id)
            
            if session_info.get('session_start'):
                session_duration = time.time() - int(session_info['session_start'])
                session_duration_minutes = round(session_duration / 60, 2)
            else:
                session_duration_minutes = 0
            
            return {
                'user_id': user_id,
                'is_active': self.is_session_active(user_id),
                'message_count': session_info.get('message_count', 0),
                'context_length': session_info.get('context_length', 0),
                'session_duration_minutes': session_duration_minutes,
                'last_activity': datetime.fromtimestamp(
                    int(session_info['last_activity'])
                ).isoformat() if session_info.get('last_activity') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {'user_id': user_id, 'error': str(e)}
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions"""
        try:
            active_sessions = self.session_store.get_active_sessions()
            cleaned_count = 0
            
            for user_id in active_sessions:
                if not self.is_session_active(user_id):
                    self.end_session(user_id)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """Get statistics for all active sessions"""
        try:
            active_sessions = self.session_store.get_active_sessions()
            session_stats = []
            
            for user_id in active_sessions:
                if self.is_session_active(user_id):
                    stats = self.get_session_stats(user_id)
                    session_stats.append(stats)
            
            return session_stats
            
        except Exception as e:
            logger.error(f"Error getting all active sessions: {e}")
            return []
    
    def transfer_session_context(self, from_user_id: str, to_user_id: str) -> bool:
        """Transfer conversation context from one user to another"""
        try:
            # Get context from source user
            context = self.session_store.get_conversation_context(from_user_id)
            
            if not context:
                return False
            
            # Clear target user's context
            self.session_store.clear_conversation_context(to_user_id)
            
            # Transfer context by replaying conversations
            for i in range(0, len(context), 2):
                if i + 1 < len(context):
                    user_msg = context[i]['content']
                    ai_msg = context[i + 1]['content']
                    self.session_store.store_conversation_turn(to_user_id, user_msg, ai_msg)
            
            logger.info(f"Transferred session context from {from_user_id} to {to_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error transferring session context: {e}")
            return False