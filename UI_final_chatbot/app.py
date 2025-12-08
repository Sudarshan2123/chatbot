from fastapi import FastAPI, WebSocket, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import secrets
import time
from typing import Dict
import re
class LoginRequest(BaseModel):
    username: str
    password: str

csrf_tokens: Dict[str, float] = {}
CSRF_TOKEN_EXPIRY = 3600 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CSRFMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exempt_paths=None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or []
    
    async def dispatch(self, request: Request, call_next):
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return await call_next(request)
        
        for pattern in self.exempt_paths:
            if re.match(pattern, request.url.path):
                return await call_next(request)
        
        csrf_token = request.headers.get("X-CSRFToken")
        if not csrf_token and hasattr(request, "form"):
            form = await request.form()
            csrf_token = form.get("csrf_token")
        
        if not csrf_token or not validate_csrf_token(csrf_token):
            return JSONResponse(
                status_code=403, 
                content={"detail": "CSRF token missing or invalid"}
            )
        
        return await call_next(request)

app.add_middleware(
    CSRFMiddleware,
    exempt_paths=[
        r"/csrf-token$",
        r"/static/.*"
    ]
)

def generate_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    csrf_tokens[token] = time.time()
    return token

def validate_csrf_token(token: str) -> bool:
    if token not in csrf_tokens:
        return False
    if time.time() - csrf_tokens[token] > CSRF_TOKEN_EXPIRY:
        del csrf_tokens[token]
        return False
    return True

@app.get("/csrf-token")
async def get_csrf_token():
    return {"csrf_token": generate_csrf_token()}

@app.post("/login")
async def login(request: LoginRequest):
    return {"message": "Login successful"}

@app.get("/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    return {"user_id": user_id, "history": []}

@app.post("/chat2")
async def chat2(request: Request):
    return {"response": "Chat response"}

@app.delete("/clear_history")
async def clear_history():
    return {"message": "History cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
