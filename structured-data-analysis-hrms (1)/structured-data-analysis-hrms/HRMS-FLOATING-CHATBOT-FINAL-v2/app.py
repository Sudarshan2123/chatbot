"""
Main application module for the HRMS Chatbot.
Handles routing, session management, and external service integrations.
"""
import os
import re
import logging
import urllib.parse
import uvicorn

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Body
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.middleware.cors import CORSMiddleware

from src.config.configuration import ConfigurationManager
from src.entity import TextToSpeechRequest, ChatRequest2
from src.pipeline.chat import ChatbotPipeline
from src.pipeline.Clear_history import ClearHistory
from src.pipeline.database_manager import DatabaseManager
from src.pipeline.Text_To_Speach import TextToSpeach
from src.utils.error_handler import UserFriendlyErrorHandler
from src.utils.security import check_no_query_params
from src.utils.session_manager import SessionManager
from src.entity import LoginRequest

# Environment Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "HRMS Chatbot"

routes = APIRouter()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manages the startup and shutdown of the FastAPI application.
    """
    db_manager = None

    try:
        db_manager = DatabaseManager()
        if not db_manager.connect():
            raise HTTPException(status_code=400, detail="Failed to connect to database")

        logging.info("Application startup: Database Connection Pool initialized.")
    except Exception as err:
        logging.error("Startup error: Failed to initialize services: %s", err)
        raise

    yield

    try:
        if db_manager:
            logging.info("Application shutdown: Disposing database connection pool...")
            db_manager.dispose_pool()
            logging.info("Application shutdown: Database connection pool disposed.")
    except Exception as err:
        logging.error("Shutdown error: Failed to clean up resources: %s", err)

# In app.py
from src.entity import LoginRequest

@routes.post("/login")
async def login_endpoint(data: LoginRequest = Body(...)):
    try:
        # 1. Decrypt logic (Matches your JS shift-5 encryption)
        def decrypt_string(encrypted_str, shift=5):
            import base64
            # Reverse the charCode shift and base64 decode
            decoded_chars = "".join([chr(ord(c) - shift) for c in encrypted_str])
            return base64.b64decode(decoded_chars).decode('utf-8')

        emp_code = decrypt_string(data.employee_code)
        firm_id = decrypt_string(data.firm_id)

        # 2. Initialize Session
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)

        # Use emp_code as the unique user_id for the session
        success = session_manager.start_session(emp_code)

        if not success:
            raise HTTPException(status_code=500, detail="Could not initialize session")

        # 3. Success Response
        return {
            "status": "success",
            "message": "Login successful",
            "emp_code": emp_code # Returned for the UI to use in its context
        }
    except Exception as e:
        logging.error(f"Login failure: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@routes.post("/chat2")
async def chat_endpoint(request: Request, data: ChatRequest2 = Body(...)):
    """Handles standard non-streaming chatbot interactions."""
    try:
        check_no_query_params(request)
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        sanitized_input = re.sub(r'[<>{}\[\]\\|]', '', user_input)

        pipeline = ChatbotPipeline()
        user_id = data.user_id
        response = await pipeline.main_chatbot(sanitized_input, lang, user_id)
        return response
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        user_message = UserFriendlyErrorHandler.get_user_message(exc, "chat processing")
        raise HTTPException(status_code=500, detail=user_message) from exc

@routes.post("/chat-stream")
async def chat_stream_endpoint(request: Request, data: ChatRequest2 = Body(...)):
    """Streams chat response using Server-Sent Events (SSE)."""
    try:
        check_no_query_params(request)
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        sanitized_input = re.sub(r'[<>{}\[\]\\|]', '', user_input)

        async def generate_response():
            try:
                pipeline = ChatbotPipeline()
                user_id = data.user_id
                async for chunk in pipeline.generate_content_stream(sanitized_input, user_id, lang):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as err:
                logging.error("Streaming error: %s", err, exc_info=True)
                yield f"data: Error: {str(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as exc:
        logging.error("Stream endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.post("/session/start/{user_id}")
async def start_session(user_id: str):
    """Initializes a new user session."""
    try:
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)
        success = session_manager.start_session(user_id)
        return {"status": "success" if success else "error", "user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.post("/session/end/{user_id}")
async def end_session(user_id: str):
    """Terminates an existing user session."""
    try:
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)
        success = session_manager.end_session(user_id)
        return {"status": "success" if success else "error", "user_id": user_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.get("/session/stats/{user_id}")
async def get_session_stats(user_id: str):
    """Retrieves statistics for a specific user session."""
    try:
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)
        return session_manager.get_session_stats(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.get("/session/active")
async def get_active_sessions():
    """Returns a list of all currently active sessions."""
    try:
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)
        sessions = session_manager.get_all_active_sessions()
        return {"status": "success", "active_sessions": sessions, "count": len(sessions)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.post("/session/cleanup")
async def cleanup_sessions():
    """Cleans up expired sessions from the manager."""
    try:
        config = ConfigurationManager().get_base_config()
        session_manager = SessionManager(config)
        cleaned_count = session_manager.cleanup_expired_sessions()
        return {"status": "success", "cleaned_count": cleaned_count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.post("/clear_history")
async def clear_chat_history(_request: Request, session_id: str = Query(default="default_session")):
    """Clears the chat history for a specific session."""
    try:
        clear_history_pipeline = ClearHistory()
        return clear_history_pipeline.clear_history(session_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@routes.post("/text-to-speech")
async def text_to_speech_endpoint(request: Request, data: TextToSpeechRequest = Body(...)):
    """Converts provided text into speech audio."""
    try:
        check_no_query_params(request)
        tts = TextToSpeach()
        return tts.Text_to_speech_process(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def init_app() -> FastAPI:
    """Initializes and configures the FastAPI application instance."""
    app_instance = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    app_instance.include_router(routes)

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    return app_instance

main_app = init_app()

# if __name__ == "__main__":
#     PORT = int(os.getenv("PORT", "5050"))
#     HOST = os.getenv("HOST", "0.0.0.0")
#     uvicorn.run(main_app, host=HOST, port=PORT)


