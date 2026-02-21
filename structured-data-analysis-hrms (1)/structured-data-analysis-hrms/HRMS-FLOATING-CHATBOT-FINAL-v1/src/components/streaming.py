import asyncio
import json
from typing import AsyncGenerator, Dict, Any
from langchain_google_vertexai import ChatVertexAI
from src.logging import logger

class StreamingChatbot:
    """Handles streaming responses for chatbot interactions"""
    
    def __init__(self, model: ChatVertexAI):
        self.model = model
    
    async def generate_content_stream(
        self, 
        prompt: str, 
        context: str = "", 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming content from the model
        
        Args:
            prompt: The user's input prompt
            context: Additional context for the response
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the generated response
        """
        try:
            # Combine prompt with context if provided
            full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
            
            # Stream the response from the model
            async for chunk in self.model.astream(full_prompt):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in generate_content_stream: {e}")
            yield f"Error: {str(e)}"
    
    async def generate_sse_stream(
        self, 
        prompt: str, 
        context: str = "",
        format_as_sse: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events formatted stream
        
        Args:
            prompt: The user's input prompt
            context: Additional context for the response
            format_as_sse: Whether to format as SSE events
            
        Yields:
            str: SSE formatted chunks or plain text chunks
        """
        try:
            async for chunk in self.generate_content_stream(prompt, context):
                if format_as_sse:
                    yield f"data: {chunk}\n\n"
                else:
                    yield chunk
                    
            # Send completion signal
            if format_as_sse:
                yield "data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"Error in generate_sse_stream: {e}")
            if format_as_sse:
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield f"Error: {str(e)}"