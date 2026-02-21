# Conversation Linking Implementation Guide

## Overview

This implementation provides intelligent conversation linking using Redis for session-based context storage. When users ask follow-up questions in the same session, the LLM can understand the connection and provide contextually relevant responses.

## Key Features

### 1. **Smart Context Detection**
- Automatically detects when user input contains references to previous conversation
- Keywords like "he", "she", "this employee", "same person", etc. trigger context linking
- Only adds context when needed to avoid prompt pollution

### 2. **Entity Extraction**
- Extracts employee codes, names, departments, and positions from conversations
- Maintains entity references across conversation turns
- Enables intelligent context building for follow-up queries

### 3. **Redis-Based Session Management**
- Fast conversation context retrieval using Redis
- Automatic session lifecycle management
- Configurable session timeouts and context limits

### 4. **Fallback to MongoDB**
- Falls back to MongoDB if Redis is unavailable
- Ensures conversation continuity even with Redis issues

## Architecture

```
User Query → Intent Classification → Context Detection → Entity Extraction → Context Building → LLM Processing
     ↓                                      ↓
Session Management ← Redis Storage ← Conversation Storage
```

## Implementation Components

### 1. **ConversationManager** (`src/pipeline/conversation_manager.py`)
- Main orchestrator for conversation context
- Handles context detection and prompt building
- Manages entity extraction and reference resolution

### 2. **SessionStore** (`src/utils/session_store.py`)
- Redis-based storage for conversation turns
- Session metadata tracking
- Fast context retrieval

### 3. **SessionManager** (`src/utils/session_manager.py`)
- Advanced session lifecycle management
- Session statistics and monitoring
- Cleanup and maintenance operations

## Configuration

Ensure your `config.yaml` includes Redis settings:

```yaml
config:
  REDIS_HOST: "localhost"
  REDIS_PORT: 6379
  REDIS_DB: 0
  REDIS_USERNAME: ""
  REDIS_PASSWORD: ""
  CACHE_TTL: 3600
```

## API Endpoints

### Chat Endpoints
- `POST /chat2` - Standard chat with conversation linking
- `POST /chat-stream` - Streaming chat with context

### Session Management
- `POST /session/start/{user_id}` - Start new session
- `POST /session/end/{user_id}` - End session
- `GET /session/stats/{user_id}` - Get session statistics
- `GET /session/active` - List all active sessions
- `POST /session/cleanup` - Clean expired sessions

## Usage Examples

### Basic Conversation Flow

```python
# 1. Start session
POST /session/start/user123

# 2. First query
POST /chat2
{
    "input": "Show me details for employee code 101373",
    "lang": "en-US"
}

# 3. Follow-up query (context-aware)
POST /chat2
{
    "input": "What is his salary?",  # "his" refers to employee 101373
    "lang": "en-US"
}

# 4. Another follow-up
POST /chat2
{
    "input": "Show me his department details",
    "lang": "en-US"
}
```

### Context Keywords That Trigger Linking

- **Pronouns**: he, she, they, him, her, them, his, hers, their
- **References**: this employee, that employee, same employee, previous employee
- **Temporal**: above, mentioned, earlier, before, same person, same user

### Entity Types Extracted

1. **Employee Codes**: Pattern matching for employee IDs
2. **Employee Names**: Names mentioned in conversations
3. **Departments**: Department references
4. **Positions**: Job titles and roles

## Testing

Run the test script to verify functionality:

```bash
python test_conversation_linking.py
```

This will test:
- Basic conversation linking
- Streaming responses with context
- Session management features
- Entity extraction and reference resolution

## Monitoring and Maintenance

### Session Statistics
```python
# Get session info
GET /session/stats/user123

Response:
{
    "user_id": "user123",
    "is_active": true,
    "message_count": 6,
    "context_length": 4,
    "session_duration_minutes": 15.5,
    "last_activity": "2024-01-15T10:30:00"
}
```

### Active Sessions Monitoring
```python
# List all active sessions
GET /session/active

Response:
{
    "status": "success",
    "active_sessions": [...],
    "count": 5
}
```

### Cleanup Operations
```python
# Clean expired sessions
POST /session/cleanup

Response:
{
    "status": "success",
    "message": "Cleaned up 3 expired sessions",
    "cleaned_count": 3
}
```

## Configuration Options

### Context Settings
- `max_context_messages`: Maximum messages to keep in context (default: 8)
- `session_timeout`: Session timeout in seconds (default: 3600)
- `context_keywords`: List of keywords that trigger context linking

### Redis Settings
- `CACHE_TTL`: Time-to-live for cached data
- Connection pool settings for optimal performance

## Best Practices

1. **Session Management**
   - Start sessions explicitly for better control
   - Clean up expired sessions regularly
   - Monitor active sessions for resource management

2. **Context Optimization**
   - Context is only added when user input contains references
   - Entity extraction focuses on HR-specific entities
   - Fallback mechanisms ensure reliability

3. **Performance**
   - Redis provides fast context retrieval
   - MongoDB fallback ensures availability
   - Configurable limits prevent memory issues

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server status
   - Verify connection settings
   - System falls back to MongoDB automatically

2. **Context Not Working**
   - Ensure user input contains reference keywords
   - Check session is active
   - Verify entity extraction patterns

3. **Session Timeout**
   - Adjust `session_timeout` setting
   - Implement session extension for long conversations
   - Monitor session activity

### Debug Information

Enable debug logging to see:
- Context detection decisions
- Entity extraction results
- Session lifecycle events
- Redis operations

## Future Enhancements

1. **Advanced Entity Linking**
   - Cross-reference validation
   - Fuzzy matching for names
   - Department hierarchy awareness

2. **Context Summarization**
   - Compress long conversations
   - Key information extraction
   - Semantic similarity matching

3. **Multi-Modal Context**
   - File upload context
   - Image reference linking
   - Document conversation history

This implementation provides a robust foundation for conversation linking that can be extended based on specific requirements.