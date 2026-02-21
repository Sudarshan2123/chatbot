from src.pipeline.Chat import Chatbot_Pipeline
from src.pipeline.Clear_history import ClearHistory
from fastapi.concurrency import asynccontextmanager
from src.pipeline.Text_To_Speach import TextToSpeach
from src.entity import TextToSpeechRequest, ChatRequest2
from src.pipeline.database_manager import DatabaseManager
from src.utils.security import check_no_query_params
from src.utils.error_handler import UserFriendlyErrorHandler
from fastapi import APIRouter, FastAPI, HTTPException, Query,Request,Body, Response, logger
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import os,re,logging,urllib.parse
from starlette.middleware.cors import CORSMiddleware
from src.config.configuration import ConfigurationManager


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "HRMS Chatbot"

routes = APIRouter()
global default_connection_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the FastAPI application.
    """
    mongo_client = None
    db_manager = None

    # Startup: Initialize MongoDB client and Database Connection Pool
    try:
        # Initialize singleton DatabaseManager with connection pool
        db_manager = DatabaseManager()
        if not db_manager.connect():
            raise HTTPException(status_code=400, detail="Failed to connect to database")
        else:
            logging.info("Application startup: Database Connection Pool initialized successfully.")
    except Exception as e:
        logging.error(f"Startup error: Failed to initialize services: {e}")
        raise

    yield

    # Shutdown: Clean up resources
    try:
        if db_manager:
            logging.info("Application shutdown: Disposing database connection pool...")
            db_manager.dispose_pool()
            logging.info("Application shutdown: Database connection pool disposed successfully.")
    except Exception as e:
        logging.error(f"Shutdown error: Failed to clean up resources: {e}")


# @routes.post("/login")
# async def login(request: Request, data: EncryptedLoginData = Body(...)):
#     try:
#         check_no_query_params(request)

#         userName = urllib.parse.unquote(data.userName)
#         password = data.password
#         encrypted = {
#         "username": userName,     # Example encrypted value
#         "password": password     # Example encrypted value
#     }
#         decrypted = decrypt_credentials(encrypted)
#         Login_chat = Login()
#         if userName and password:
#             result = Login_chat.login_user(decrypted['username'], decrypted['password'])
#             user_Status = result.get("user_Status")
#             if result.get("status") == "success":
#                 if user_Status=="active":
#                     access_token = result.get("access_token")
#                     response = JSONResponse(content={'status': 'success','user_Status':'active'})
#                     response.headers['Authorization'] = f"Bearer {access_token}"
#                     return response
#                 elif user_Status=="Inactive":
#                     access_token = result.get("access_token")
#                     user_Status=result.get("user_Status")
#                     response = JSONResponse(content={'status': 'success','user_Status':'Inactive'})
#                     response.headers['Authorization'] = f"Bearer {access_token}"
#                     return response
#             else:
#                 return JSONResponse(result, status_code=400)

#         else:
#             raise HTTPException(status_code=400, detail="Employee code and password required for login")
#     except ValidationError as e:
#         print(e)
#         raise HTTPException(status_code=422, detail=e.errors())


# @routes.post("/proxy-verify-otp")
# async def proxy_verify_otp(request: Request, otp_data: Dict[str, str] = Body(...)):
#     """
#     Proxies the OTP verification request to the external LDAP service.
#     """
#     try:
#         # The OTP is directly in otp_data, no need for urllib.parse.unquote unless you sent it encoded
#         otp = otp_data.get("otp")
#         if not otp:
#             raise HTTPException(status_code=400, detail="OTP is required")

#         external_url = "https://docker.mactech.net.in:5013/ldap-service/verifyLoginOtp"

#         async with httpx.AsyncClient() as client:
#             external_response = await client.post(
#                 external_url,
#                 json={"otp": otp},
#                 headers={"Content-Type": "application/json"}
#             )

#             # Check the status code from the external service
#             external_response.raise_for_status() # Raises an exception for 4xx/5xx responses

#             # If external service responds with 200, return its content
#             # You might need to adjust the structure based on what the external service returns for success/failure
#             return JSONResponse(content=external_response.json(), status_code=external_response.status_code)

#     except httpx.HTTPStatusError as e:
#         # Handle HTTP errors from the external service (e.g., 400, 401, 404 from docker.mactech.net.in)
#         logging.error(f"External OTP service responded with error: {e.response.status_code} - {e.response.text}")
#         return JSONResponse(
#             content={"detail": f"External OTP service error: {e.response.status_code} - {e.response.text}"},
#             status_code=e.response.status_code
#         )
#     except httpx.RequestError as e:
#         # Handle network errors, DNS issues, etc., when calling the external service
#         logging.error(f"Failed to connect to external OTP service: {e}")
#         raise HTTPException(status_code=503, detail=f"Cannot connect to external OTP service: {e}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred in proxy-verify-otp: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error during OTP proxy.")




@routes.post("/chat2")
async def chat(request: Request, data: ChatRequest2 = Body(...)):
    try:
        check_no_query_params(request)
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        sanitized_input = re.sub(r'[<>{}\[\]\\|]', '', user_input)
        chat=Chatbot_Pipeline()
        user_id = "101373"  # Use consistent user_id
        response = await chat.main_chatbot(sanitized_input, lang, user_id)
        return response
    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        user_message = UserFriendlyErrorHandler.get_user_message(e, "chat processing")
        raise HTTPException(status_code=500, detail=user_message)

@routes.post("/chat-stream")
async def chat_stream(request: Request, data: ChatRequest2 = Body(...)):
    """
    Streams chat response using the generate_content_stream function
    """
    try:
        check_no_query_params(request)
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        # print(f"DEBUG: Raw user_input before sanitization: '{user_input}'")
        sanitized_input = re.sub(r'[<>{}\[\]\\|]', '', user_input)
        # print(f"DEBUG: Sanitized input after regex: '{sanitized_input}'")

        async def generate_response():
            try:
                chat = Chatbot_Pipeline()
                user_id = "101373"  # Use consistent user_id
                async for chunk in chat.generate_content_stream(sanitized_input, lang, user_id):
                    yield f"data: {chunk}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logging.error(f"Streaming error: {e}", exc_info=True)
                yield f"data: Error: {str(e)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    except Exception as e:
        logging.error(f"Stream endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @routes.post("/chat_history/{user_id}",response_model=ChatHistoryResponse)
# async def get_chat_history(
#     user_id: str, # Changed from session_id to user_id
#     request : Request,
#     limit: Annotated[int, Query(ge=1, le=100)] = 10,
#     offset: Annotated[int, Query(ge=0)] = 0,
#     # mongo_client: MongoClient = Depends(history_page_instance.get_mongo_client),
# ) -> ChatHistoryResponse:
#     authorization_header = request.headers.get("Authorization")
#     if authorization_header and authorization_header.startswith("Bearer "):
#         access_token = authorization_header[len("Bearer "):].strip()
#     else:
#         raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
#     try:
#         config_obj=ConfigurationManager()
#         config = config_obj.get_base_config()
#         chat=Token(config)
#         tok_data=chat.validate_access_token(access_token)
#         if not tok_data:
#             raise HTTPException(status_code=401, detail="Invalid or expired access token")

#         new_access_token=chat.create_update_token(tok_data)
#         # Use singleton DatabaseManager (already initialized at startup)
#         db_manager = DatabaseManager()
#         # 1. Fetch history from MongoDB using the HistoryPage instance:
#         # history = history_page_instance.get_chat_history_from_mongodb(
#         #     user_id=user_id,
#         #     mongo_client=mongo_client,
#         #     limit=limit,
#         #     offset=offset,
#         # )

#         # 3. Convert the MongoDB results (dictionaries) into a list of ChatMessage objects:
#         chat_messages = []
#         if history:
#             for message_doc in history:
#                 try:
#                     query_content = message_doc.get("query")
#                     response_content = message_doc.get("query_response")
#                     timestamp_str = message_doc.get("timestamp")

#                     created_at = HistoryPage.parse_timestamp_string(timestamp_str)

#                     # Create ChatMessage for the user's query
#                     if query_content:
#                         user_chat_message = ChatMessage(
#                             role="user",
#                             content=query_content,
#                             created_at=created_at,
#                         )
#                         chat_messages.append(user_chat_message)

#                     # Create ChatMessage for the AI's response
#                     if response_content:
#                         ai_chat_message = ChatMessage(
#                             role="Ai",
#                             content=response_content,
#                             created_at=created_at,
#                         )
#                         chat_messages.append(ai_chat_message)

#                 except Exception as e:
#                     logging.error(f"Error processing message document {message_doc.get('_id')}: {e}", exc_info=True)
#                     # Continue processing other documents even if one fail
#         else:
#             logging.info(f"No chat history found for user: {user_id}. Returning empty list.")

        # 4. Construct the final response using the ChatHistoryResponse model:
        # chat_response = ChatHistoryResponse(
        #     user_id=user_id, messages=chat_messages
        # )
        # return Response(
        #         content=chat_response.model_dump_json(),
        #         headers={"Authorization": f"Bearer {new_access_token}"},
        #         media_type="application/json"
        #     )

    # except HTTPException as e:
    #     raise e  # Re-raise HTTPExceptions (404, etc.)
    # except Exception as e:
    #     # 5. Handle any other exceptions (e.g., MongoDB connection errors, etc.):
    #     logging.error(f"Error in get_chat_history endpoint: {e}", exc_info=True)
    #     raise HTTPException(
    #         status_code=500, detail="Failed to retrieve chat history due to an internal server error."
    #     )



@routes.post("/session/start/{user_id}")
async def start_session(user_id: str):
    try:
        config_obj = ConfigurationManager()
        config = config_obj.get_base_config()
        from src.utils.session_manager import SessionManager
        session_manager = SessionManager(config)
        success = session_manager.start_session(user_id)
        return {"status": "success" if success else "error", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routes.post("/session/end/{user_id}")
async def end_session(user_id: str):
    try:
        config_obj = ConfigurationManager()
        config = config_obj.get_base_config()
        from src.utils.session_manager import SessionManager
        session_manager = SessionManager(config)
        success = session_manager.end_session(user_id)
        return {"status": "success" if success else "error", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routes.get("/session/stats/{user_id}")
async def get_session_stats(user_id: str):
    try:
        config_obj = ConfigurationManager()
        config = config_obj.get_base_config()
        from src.utils.session_manager import SessionManager
        session_manager = SessionManager(config)
        stats = session_manager.get_session_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routes.get("/session/active")
async def get_active_sessions():
    try:
        config_obj = ConfigurationManager()
        config = config_obj.get_base_config()
        from src.utils.session_manager import SessionManager
        session_manager = SessionManager(config)
        sessions = session_manager.get_all_active_sessions()
        return {"status": "success", "active_sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routes.post("/session/cleanup")
async def cleanup_sessions():
    try:
        config_obj = ConfigurationManager()
        config = config_obj.get_base_config()
        from src.utils.session_manager import SessionManager
        session_manager = SessionManager(config)
        cleaned_count = session_manager.cleanup_expired_sessions()
        return {"status": "success", "message": f"Cleaned up {cleaned_count} expired sessions", "cleaned_count": cleaned_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routes.post("/clear_history")
async def clear_chat_history(request: Request, session_id: str = Query(default="default_session")):
    try:
        print(f"Session ID: {session_id}")
        clear_history = ClearHistory()
        response = clear_history.clear_history(session_id)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        return response

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())





@routes.post("/text-to-speech")
async def text_to_speech(request: Request, data: TextToSpeechRequest = Body(...)):
    try:
        check_no_query_params(request)
        # authorization_header = request.headers.get("Authorization")
        # if authorization_header and authorization_header.startswith("Bearer "):
        #     access_token = authorization_header[len("Bearer "):].strip()
        # else:
        #     raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
        text_to_speech = TextToSpeach()
        response=text_to_speech.Text_to_speech_process(data)
        return response
        # access_token = urllib.parse.unquote(data.access_token)


    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())



# Add the custom middleware to the FastAPI app

# def init_app() -> FastAPI:
#     app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
#     app.include_router(routes)

#     # Add global exception handler first
#     # app.add_middleware(GlobalExceptionMiddleware)

#     # app.add_middleware(SecurityHeadersMiddleware)
#     # app.add_middleware(RestrictSwaggerMiddleware)
#     # app.add_middleware(BlockReDocMiddleware)
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],  # Adjust this as necessary for your deployment
#         allow_credentials=True,
#         allow_methods=["POST"],
#         allow_headers=["Authorization"],
#         # expose_headers=["Authorization"],
#     )
#     return app

def init_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    app.include_router(routes)

    # Updated CORS middleware for streaming
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],  # Changed from ["POST"] to ["*"]
        allow_headers=["*"],  # Changed from ["Authorization"] to ["*"]
        expose_headers=["*"],  # Added this
    )
    return app


app = init_app()
if app is None:
    raise TypeError("app not instantiated")

# if __name__ == "__main__":
#     PORT = int(os.getenv("PORT", default=5050))
#     HOST = os.getenv("HOST", default="0.0.0.0")
#     app = init_app()
#     if app is None:
#         raise TypeError("app not instantiated")
#     uvicorn.run(app, host=HOST, port=PORT)
