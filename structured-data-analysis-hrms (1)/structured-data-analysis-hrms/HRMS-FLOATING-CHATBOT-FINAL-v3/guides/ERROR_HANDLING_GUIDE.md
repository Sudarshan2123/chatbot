# Error Handling System Usage Examples

## Overview
The new error handling system automatically converts technical errors into user-friendly messages. It works at multiple levels:

1. **Global Middleware**: Catches all unhandled exceptions in FastAPI
2. **Component Level**: Individual components use the error handler
3. **Custom Exceptions**: Enhanced custom exceptions with user-friendly messages

## Usage Examples

### 1. In Components (like data_analyzer.py)
```python
from src.utils.error_handler import UserFriendlyErrorHandler

try:
    # Some operation that might fail
    result = risky_operation()
except Exception as e:
    # Convert to user-friendly message
    user_message = UserFriendlyErrorHandler.get_user_message(e, "data analysis")
    return user_message
```

### 2. Using Custom Exceptions
```python
from src.exception import ProcessingError, ValidationError

# Raise with automatic user-friendly message
raise ProcessingError("Database connection failed", component="DataAnalyzer")

# Raise with technical details only (for internal use)
raise ProcessingError("Database connection failed", user_friendly=False)
```

### 3. Quick Error Handling
```python
from src.exception import handle_error_gracefully

try:
    # Some operation
    pass
except Exception as e:
    return handle_error_gracefully(e, "chat processing")
```

## Error Categories and Messages

### LLM/AI Errors
- **Technical**: "No generation chunks were returned"
- **User-Friendly**: "I'm having trouble processing your request right now. Please try again in a moment."

### Database Errors
- **Technical**: "ORA-12541: TNS:no listener"
- **User-Friendly**: "I'm having trouble accessing the database. Please try again later."

### Authentication Errors
- **Technical**: "Invalid token signature"
- **User-Friendly**: "Your session has expired. Please log in again."

### Network Errors
- **Technical**: "httpx.ConnectTimeout"
- **User-Friendly**: "I'm having network connectivity issues. Please try again in a moment."

## Benefits

1. **Consistent Experience**: All errors show user-friendly messages
2. **Security**: No technical details exposed to users
3. **Debugging**: Technical details still logged for developers
4. **Automatic**: Works without code changes in most places
5. **Flexible**: Can be customized per component or error type

## Implementation Details

The system uses regex patterns to match common error types and provides appropriate user messages. The global middleware ensures that even unexpected errors are handled gracefully.