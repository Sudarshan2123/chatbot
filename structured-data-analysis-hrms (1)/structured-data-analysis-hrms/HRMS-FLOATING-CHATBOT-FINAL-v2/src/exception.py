import sys
from src.logging import logger
from typing import Optional

def error_message_detail(error, error_detail:sys):
    _,_,exc_tab=error_detail.exc_info()
    file_name=exc_tab.tb_frame.f_code.co_filename
    error_message="Error occuured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tab.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

class ValidationError(CustomException):
    """
    Exception raised for errors in input validation.
    
    Examples:
        >>> raise ValidationError("Invalid user input: empty string")
        >>> raise ValidationError("Token validation failed", details="Expired token")
    """
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.details = details
        error_message = f"Validation Error: {message}"
        if details:
            error_message += f" - Details: {details}"
        super().__init__(error_message, sys)
        
        # Log validation errors with WARNING level
        logger.warning(f"ValidationError: {error_message}")

class ProcessingError(CustomException):
    """
    Exception raised for errors during request processing.
    
    Examples:
        >>> raise ProcessingError("Failed to process chat message")
        >>> raise ProcessingError("Database operation failed", component="DatabaseManager")
    """
    
    def __init__(self, message: str, component: Optional[str] = None):
        self.component = component
        error_message = f"Processing Error: {message}"
        if component:
            error_message += f" in component: {component}"
        super().__init__(error_message, sys)
        
        # Log processing errors with ERROR level
        logger.error(f"ProcessingError: {error_message}")

