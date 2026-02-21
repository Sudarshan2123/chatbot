import logging
import re
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors for better classification"""
    DATABASE_ERROR = "database"
    LLM_ERROR = "llm"
    VALIDATION_ERROR = "validation"
    AUTHENTICATION_ERROR = "auth"
    NETWORK_ERROR = "network"
    PROCESSING_ERROR = "processing"
    UNKNOWN_ERROR = "unknown"

class UserFriendlyErrorHandler:
    """
    Centralized error handler that converts technical errors into user-friendly messages
    """
    
    # Error patterns and their user-friendly messages
    ERROR_PATTERNS = {
        # LLM/AI Related Errors
        r"No generation chunks were returned": {
            "message": "I'm having trouble processing your request right now. Please try again in a moment.",
            "category": ErrorCategory.LLM_ERROR
        },
        r"ValueError.*generation.*chunk": {
            "message": "I encountered an issue while generating a response. Please rephrase your question and try again.",
            "category": ErrorCategory.LLM_ERROR
        },
        r"langchain.*timeout": {
            "message": "Your request is taking longer than expected. Please try a simpler question or try again later.",
            "category": ErrorCategory.LLM_ERROR
        },
        r"vertexai.*error|vertex.*error": {
            "message": "I'm experiencing connectivity issues with the AI service. Please try again shortly.",
            "category": ErrorCategory.LLM_ERROR
        },
        
        # Database Related Errors
        r"ORA-\d+": {
            "message": "I'm having trouble accessing the database. Please try again later.",
            "category": ErrorCategory.DATABASE_ERROR
        },
        r"connection.*refused|connection.*timeout": {
            "message": "I can't connect to the database right now. Please try again in a few minutes.",
            "category": ErrorCategory.DATABASE_ERROR
        },
        r"sqlalchemy.*error": {
            "message": "There's a database connectivity issue. Please try again later.",
            "category": ErrorCategory.DATABASE_ERROR
        },
        r"No tables.*available|table.*not.*found": {
            "message": "The requested data is not currently available. Please contact support if this persists.",
            "category": ErrorCategory.DATABASE_ERROR
        },
        
        # Authentication/Authorization Errors
        r"Invalid.*token|token.*expired|unauthorized": {
            "message": "Your session has expired. Please log in again.",
            "category": ErrorCategory.AUTHENTICATION_ERROR
        },
        r"access.*denied|permission.*denied": {
            "message": "You don't have permission to access this information.",
            "category": ErrorCategory.AUTHENTICATION_ERROR
        },
        
        # Network/Connectivity Errors
        r"httpx.*error|requests.*error|connection.*error": {
            "message": "I'm having network connectivity issues. Please try again in a moment.",
            "category": ErrorCategory.NETWORK_ERROR
        },
        r"timeout.*error|read.*timeout": {
            "message": "The request timed out. Please try again with a simpler query.",
            "category": ErrorCategory.NETWORK_ERROR
        },
        
        # Validation Errors
        r"validation.*error|invalid.*input": {
            "message": "Please check your input and try again.",
            "category": ErrorCategory.VALIDATION_ERROR
        },
        r"missing.*required|field.*required": {
            "message": "Some required information is missing. Please provide all necessary details.",
            "category": ErrorCategory.VALIDATION_ERROR
        },
        
        # Processing Errors
        r"processing.*error|analysis.*error": {
            "message": "I encountered an issue while processing your request. Please try again.",
            "category": ErrorCategory.PROCESSING_ERROR
        },
        r"memory.*error|out.*of.*memory": {
            "message": "Your request requires too much processing power. Please try a simpler query.",
            "category": ErrorCategory.PROCESSING_ERROR
        }
    }
    
    @classmethod
    def handle_error(cls, error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert technical error into user-friendly response
        
        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            
        Returns:
            Dict with user-friendly message and metadata
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Log the technical error for debugging
        logger.error(f"Error in {context or 'unknown context'}: {error_type}: {error}", exc_info=True)
        
        # Find matching pattern
        for pattern, error_info in cls.ERROR_PATTERNS.items():
            if re.search(pattern, error_str, re.IGNORECASE):
                return {
                    "user_message": error_info["message"],
                    "category": error_info["category"].value,
                    "technical_error": error_type,
                    "context": context
                }
        
        # Default fallback message
        return {
            "user_message": "I'm experiencing technical difficulties. Please try again later or contact support if the issue persists.",
            "category": ErrorCategory.UNKNOWN_ERROR.value,
            "technical_error": error_type,
            "context": context
        }
    
    @classmethod
    def get_user_message(cls, error: Exception, context: Optional[str] = None) -> str:
        """
        Get just the user-friendly message from an error
        
        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            
        Returns:
            User-friendly error message string
        """
        error_info = cls.handle_error(error, context)
        return error_info["user_message"]
    
    @classmethod
    def is_critical_error(cls, error: Exception) -> bool:
        """
        Determine if an error is critical and requires immediate attention
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if error is critical, False otherwise
        """
        critical_patterns = [
            r"database.*down|database.*unavailable",
            r"out.*of.*memory",
            r"system.*error|critical.*error"
        ]
        
        error_str = str(error).lower()
        return any(re.search(pattern, error_str, re.IGNORECASE) for pattern in critical_patterns)