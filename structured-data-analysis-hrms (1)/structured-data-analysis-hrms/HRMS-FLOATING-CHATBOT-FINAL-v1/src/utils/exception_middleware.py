from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from src.utils.error_handler import UserFriendlyErrorHandler

logger = logging.getLogger(__name__)

class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """
    Global exception handler middleware that catches all unhandled exceptions
    and converts them to user-friendly responses
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException:
            # Re-raise HTTP exceptions as they're already handled
            raise
        except Exception as e:
            # Handle all other exceptions
            error_info = UserFriendlyErrorHandler.handle_error(e, f"{request.method} {request.url.path}")
            
            # Determine appropriate status code based on error category
            status_code = self._get_status_code(error_info["category"])
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": error_info["user_message"],
                    "error_type": "system_error"
                }
            )
    
    def _get_status_code(self, category: str) -> int:
        """Map error categories to appropriate HTTP status codes"""
        status_map = {
            "auth": 401,
            "validation": 400,
            "database": 503,
            "network": 503,
            "llm": 503,
            "processing": 500,
            "unknown": 500
        }
        return status_map.get(category, 500)