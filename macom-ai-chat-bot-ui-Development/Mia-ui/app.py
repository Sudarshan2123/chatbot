import os
import secrets
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- MIDDLEWARE DEFINITIONS ---

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Generate nonce for CSP
        nonce = secrets.token_urlsafe(16)
        request.state.nonce = nonce 
        
        response = await call_next(request)
        
        # Security Headers
        response.headers['Server'] = 'Frontend'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'no-referrer'
        
        # Updated CSP: Removed 'strict-dynamic' temporarily to ensure local scripts load easily
        # Added 'nonce-{nonce}' to script-src
        response.headers['Content-Security-Policy'] = (
            f"default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}'; "
            f"style-src 'self' 'unsafe-inline'; "
            f"img-src 'self' data:; "
            f"connect-src 'self' http://localhost:5050 https://mia-bot-446976656513.us-central1.run.app; "
            f"base-uri 'self';"
        )
        return response

class RestrictSwaggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Block /docs and /openapi.json
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json"):
            return JSONResponse({"detail": "Access restricted"}, status_code=403)
        return await call_next(request)

class BlockReDocMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/redoc":
            return JSONResponse({"detail": "Access forbidden"}, status_code=403)
        return await call_next(request)

# --- APP INITIALIZATION ---

app = FastAPI(docs_url=None, redoc_url=None)

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(BASE_DIR, "static")
templates_path = os.path.join(BASE_DIR, "templates")

# --- MIDDLEWARE REGISTRATION ---
# Note: Middleware executes in reverse order of addition

# 1. Standard CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Custom Security (Uncomment these to enable protection)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BlockReDocMiddleware)
app.add_middleware(RestrictSwaggerMiddleware)

# --- STATIC FILES & TEMPLATES ---

if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory=templates_path)

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Safe way to get nonce: fallback to empty string if middleware is disabled
    nonce = getattr(request.state, "nonce", "")
    return templates.TemplateResponse("base.html", {"request": request, "nonce": nonce})

if __name__ == "__main__":
    # Using 127.0.0.1 is often more stable for local CSP testing than 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=5001)