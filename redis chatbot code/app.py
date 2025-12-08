import asyncio
from typing import Annotated, Any, Dict
import httpx
from pymongo import MongoClient
from src.components.token import Token
from src.pipeline.Chat import Chatbot_Pipeline
from src.pipeline.Clear_history import ClearHistory
from fastapi.concurrency import asynccontextmanager
from src.pipeline.Login import Login
from src.pipeline.Text_To_Speach import TextToSpeach
from src.entity import ChatRequest,ChatHistoryClear,TranslationRequest,TextToSpeechRequest,EncryptedLoginData,OTP_generator_request,ChatRequest2
from src.pipeline.database_manager import DatabaseManager
from src.pipeline.history import ChatHistoryResponse, ChatMessage, Config, HistoryPage
from src.utils.security import SecurityHeadersMiddleware,RestrictSwaggerMiddleware,BlockReDocMiddleware,check_no_query_params
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query,Request,Body, Response, logger
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import os,re,logging,urllib.parse,uvicorn
from starlette.middleware.cors import CORSMiddleware
from src.config.configuration import ConfigurationManager
from src.utils.common import decrypt_credentials
# Import CSRF protection
from starlette_csrf import CSRFMiddleware


# logging.getLogger().setLevel(logging.DEBUG)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
routes = APIRouter()
config = Config()
global default_connection_data
history_page_instance = HistoryPage(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the FastAPI application, specifically handling the MongoDB client.
    """
    mongo_client = None

    # Startup: Initialize MongoDB client
    try:
        logging.info("Application startup: Initializing MongoDB client And Database Connection...")
        mongo_client = history_page_instance.get_mongo_client()
        logging.info("Application startup: MongoDB client initialized successfully.")
        db_manager = DatabaseManager()
        if not db_manager.connect():
            raise HTTPException(status_code=400, detail="Failed to connect to database")
        else:
            # analyzer = DataAnalyzer()
            logging.info("Advanced DataAnalyzer initialized successfully")
            logging.info("Application startup: Database Connection client initialized successfully.")
    except Exception as e:
        logging.error(f"Startup error: Failed to initialize MongoDB client: {e}")
        raise

    yield  

    if mongo_client: 
        try:
            logging.info("Application shutdown: Closing MongoDB client connection...")
            history_page_instance.close_mongo_client()
            logging.info("Application shutdown: MongoDB client connection closed successfully.")
        except Exception as e:
            logging.error(f"Shutdown error: Failed to close MongoDB client connection: {e}")
   

@routes.get("/csrf-token")
async def get_csrf_token(request: Request):
    """Get CSRF token for authenticated requests"""
    csrf_token = request.headers.get("X-CSRFToken") or request.cookies.get("csrftoken")
    return {"csrf_token": csrf_token}

@routes.post("/login")
async def login(request: Request, data: EncryptedLoginData = Body(...)):
    try:
        check_no_query_params(request)
        
        userName = urllib.parse.unquote(data.userName)
        password = data.password
        encrypted = {
        "username": userName,     # Example encrypted value
        "password": password     # Example encrypted value
    }
        decrypted = decrypt_credentials(encrypted)
        Login_chat = Login()
        if userName and password:
            result = Login_chat.login_user(decrypted['username'], decrypted['password'])
            user_Status = result.get("user_Status")
            if result.get("status") == "success":
                if user_Status=="active":
                    access_token = result.get("access_token")
                    response = JSONResponse(content={'status': 'success','user_Status':'active'})
                    response.headers['Authorization'] = f"Bearer {access_token}"
                    return response
                elif user_Status=="Inactive":
                    access_token = result.get("access_token")
                    user_Status=result.get("user_Status")
                    response = JSONResponse(content={'status': 'success','user_Status':'Inactive'})
                    response.headers['Authorization'] = f"Bearer {access_token}"
                    return response
            else:
                return JSONResponse(result, status_code=400)
            
        else:
            raise HTTPException(status_code=400, detail="Employee code and password required for login")
    except ValidationError as e:
        print(e)
        raise HTTPException(status_code=422, detail=e.errors())
    

@routes.post("/proxy-verify-otp")
async def proxy_verify_otp(request: Request, otp_data: Dict[str, str] = Body(...)):
    """
    Proxies the OTP verification request to the external LDAP service.
    """
    try:
        # The OTP is directly in otp_data, no need for urllib.parse.unquote unless you sent it encoded
        otp = otp_data.get("otp")
        if not otp:
            raise HTTPException(status_code=400, detail="OTP is required")

        external_url = "https://docker.mactech.net.in:5013/ldap-service/verifyLoginOtp"

        async with httpx.AsyncClient() as client:
            external_response = await client.post(
                external_url,
                json={"otp": otp},
                headers={"Content-Type": "application/json"}
            )

            # Check the status code from the external service
            external_response.raise_for_status() # Raises an exception for 4xx/5xx responses

            # If external service responds with 200, return its content
            # You might need to adjust the structure based on what the external service returns for success/failure
            return JSONResponse(content=external_response.json(), status_code=external_response.status_code)

    except httpx.HTTPStatusError as e:
        # Handle HTTP errors from the external service (e.g., 400, 401, 404 from docker.mactech.net.in)
        logging.error(f"External OTP service responded with error: {e.response.status_code} - {e.response.text}")
        return JSONResponse(
            content={"detail": f"External OTP service error: {e.response.status_code} - {e.response.text}"},
            status_code=e.response.status_code
        )
    except httpx.RequestError as e:
        # Handle network errors, DNS issues, etc., when calling the external service
        logging.error(f"Failed to connect to external OTP service: {e}")
        raise HTTPException(status_code=503, detail=f"Cannot connect to external OTP service: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in proxy-verify-otp: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during OTP proxy.")
    


    
@routes.post("/chat2")
async def chat(request: Request, data: ChatRequest2 = Body(...)):
    try:
        check_no_query_params(request) 
        # access_token = request.headers.get("Authorization")
        authorization_header = request.headers.get("Authorization")
        if authorization_header and authorization_header.startswith("Bearer "):
            access_token = authorization_header[len("Bearer "):].strip()
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        print(f"user input recived from the user on chat api :{user_input}")
        sanitized_input = re.sub(r'[<>{}[\]\\|]', '', user_input)
        chat=Chatbot_Pipeline()
        response = await chat.main_chatbot(access_token, sanitized_input,lang)
        return response
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())



@routes.post("/chat_history/{user_id}",response_model=ChatHistoryResponse)
async def get_chat_history(
    user_id: str, # Changed from session_id to user_id
    request : Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
    mongo_client: MongoClient = Depends(history_page_instance.get_mongo_client),
) -> ChatHistoryResponse:
    authorization_header = request.headers.get("Authorization")
    if authorization_header and authorization_header.startswith("Bearer "):
        access_token = authorization_header[len("Bearer "):].strip()
    else:
        raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
    try:
        config_obj=ConfigurationManager()
        config = config_obj.get_base_config()
        chat=Token(config)
        tok_data=chat.validate_access_token(access_token)
        if not tok_data:
            raise HTTPException(status_code=401, detail="Invalid or expired access token")

        new_access_token=chat.create_update_token(tok_data)
        db_manager = DatabaseManager()
        if not db_manager.connect():
            raise HTTPException(status_code=400, detail="Failed to connect to database")
        else:
            # analyzer = DataAnalyzer()
            logging.info("Advanced DataAnalyzer initialized successfully")
            logging.info("Application startup: Database Connection client initialized successfully.")
        # 1. Fetch history from MongoDB using the HistoryPage instance:
        history = history_page_instance.get_chat_history_from_mongodb(
            user_id=user_id, 
            mongo_client=mongo_client, 
            limit=limit,             
            offset=offset,           
        )

        # 3. Convert the MongoDB results (dictionaries) into a list of ChatMessage objects:
        chat_messages = []
        if history: 
            for message_doc in history:
                try:
                    query_content = message_doc.get("query")
                    response_content = message_doc.get("query_response")
                    timestamp_str = message_doc.get("timestamp")

                    created_at = HistoryPage.parse_timestamp_string(timestamp_str)

                    # Create ChatMessage for the user's query
                    if query_content:
                        user_chat_message = ChatMessage(
                            role="user",
                            content=query_content,
                            created_at=created_at,
                        )
                        chat_messages.append(user_chat_message)

                    # Create ChatMessage for the AI's response
                    if response_content:
                        ai_chat_message = ChatMessage(
                            role="Ai", 
                            content=response_content,
                            created_at=created_at, 
                        )
                        chat_messages.append(ai_chat_message)

                except Exception as e:
                    logging.error(f"Error processing message document {message_doc.get('_id')}: {e}", exc_info=True)
                    # Continue processing other documents even if one fail
        else:
            logging.info(f"No chat history found for user: {user_id}. Returning empty list.")
            
        # 4. Construct the final response using the ChatHistoryResponse model:
        chat_response = ChatHistoryResponse(
            user_id=user_id, messages=chat_messages
        )
        return Response(
                content=chat_response.model_dump_json(),
                headers={"Authorization": f"Bearer {new_access_token}"},
                media_type="application/json"
            )

    except HTTPException as e:
        raise e  # Re-raise HTTPExceptions (404, etc.)
    except Exception as e:
        # 5. Handle any other exceptions (e.g., MongoDB connection errors, etc.):
        logging.error(f"Error in get_chat_history endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chat history due to an internal server error."
        )



@routes.post("/clear_history")
async def clear_chat_history(request: Request, data: ChatHistoryClear = Body(...)):
    try:
        check_no_query_params(request)
        authorization_header = request.headers.get("Authorization")
        if authorization_header and authorization_header.startswith("Bearer "):
            access_token = authorization_header[len("Bearer "):].strip()
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header") 
        access_token = urllib.parse.unquote(data.access_token)
        clear_history = ClearHistory()
        response=clear_history.clear_history(access_token)
        return response

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())    





@routes.post("/text-to-speech")
async def text_to_speech(request: Request, data: TextToSpeechRequest = Body(...)):
    try:
        check_no_query_params(request) 
        authorization_header = request.headers.get("Authorization")
        if authorization_header and authorization_header.startswith("Bearer "):
            access_token = authorization_header[len("Bearer "):].strip()
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
        text_to_speech = TextToSpeach()
        response=text_to_speech.Text_to_speech_process(data,access_token)
        return response
        # access_token = urllib.parse.unquote(data.access_token)
        
                          
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())    



# Add the custom middleware to the FastAPI app

def init_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None,lifespan=lifespan)
    csrf_secret = os.getenv("CSRF_SECRET", "dJ3z5MQmuTkjCfzFeTT-8Ttxkjj8HuQLj_i63_g-rT4")
    # CSRF middleware temporarily disabled - enable for production
    # app.add_middleware(
    #     CSRFMiddleware,
    #     secret=csrf_secret,
    #     cookie_name="csrftoken",
    #     header_name="X-CSRFToken",
    #     cookie_secure=True,
    #     cookie_samesite="strict",
    #     exempt_urls=[re.compile(r"/docs"), re.compile(r"/openapi\.json"), re.compile(r"/login")]
    # )
    app.include_router(routes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RestrictSwaggerMiddleware)
    app.add_middleware(BlockReDocMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this as necessary for your deployment
        allow_credentials=True,
        allow_methods=["POST"],
        allow_headers=["Authorization","X-CSRFToken"],
        expose_headers=["Authorization"],
    )
    return app
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", default=5050))
    HOST = os.getenv("HOST", default="0.0.0.0")
    app = init_app()
    if app is None:
        raise TypeError("app not instantiated")
    uvicorn.run(app, host=HOST, port=PORT)
