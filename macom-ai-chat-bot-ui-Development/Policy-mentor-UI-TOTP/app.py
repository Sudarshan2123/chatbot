from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import secrets
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        nonce = secrets.token_urlsafe(16)
        request.state.nonce = nonce  #
        response = await call_next(request)
        response.headers['Server'] = 'Frontend'  
        response.headers['X-Powered-By'] = ''
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Expect-CT'] = 'max-age=86400, enforce'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Clear-Site-Data'] = '"cache", "cookies", "storage"'
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers['Permissions-Policy'] = 'geolocation=(), camera=()'
        # response.headers['Content-Security-Policy'] = (
                                            
        #                                             "default-src 'self'; "
        #                                             "script-src 'strict-dynamic' 'sha256-IN4qS4M96pvyMSLGx6+19gpeeI3Maz5Ak8hWoh53paU='; "
        #                                             # "script-src-elem 'self';"
        #                                             "style-src 'self' 'unsafe-inline'; "  # semicolon separating directives
        #                                             "style-src-elem 'self' ;"  # Separate for external stylesheets
        #                                             "connect-src 'self' http://localhost:5050;"
        #                                             "base-uri 'self';"
        #                                             #"connect-src 'self' https://vapt-mia-app-3-476055803082.us-central1.run.app;"
        #                                         )
        response.headers['Content-Security-Policy'] = (
                                                "default-src 'self'; "
                                                "script-src 'strict-dynamic' 'sha256-u7xQcZ5006nQiKGPjNRm+65hVtjt01gq+air62f5s3Q=' 'sha256-Ady8Dn6VdiUv6Javn9piXFbW8mUSlM0taAAOwZWC53s=' ; "
                                                "script-src-elem http://localhost:5000 'sha256-u7xQcZ5006nQiKGPjNRm+65hVtjt01gq+air62f5s3Q=' 'sha256-Ady8Dn6VdiUv6Javn9piXFbW8mUSlM0taAAOwZWC53s=' ;"
                                                "style-src 'self' ;"
                                                "media-src 'self' data:;"
                                                "connect-src 'self' https://chat-backend-app-446976656513.us-central1.run.app  ; "
                                                "base-uri 'self';"
                                            )
        response.headers["NEL"] = '{"report_to": "default", "max_age": 31536000, "include_subdomains": true}'
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        return response
    #http://localhost:5050
    #'unsafe-inline'
    #'unsafe-hashes'
class RestrictSwaggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/docs"):
            return JSONResponse({"detail": "Access to Swagger UI is restricted"}, status_code=403)
        response = await call_next(request)
        return response

class BlockReDocMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/redoc":
            return JSONResponse(status_code=403, content={"detail": "Access forbidden"})
        return await call_next(request)


app = FastAPI(docs_url=None, redoc_url=None)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
import jwt
from datetime import datetime, timedelta



def create_jwt_token(payload, secret_key, key_id="key1"):
    # Set expiration time
    expiration = datetime.utcnow() + timedelta(hours=24)
    
    # Add expiration claim to payload
    payload['exp'] = int(expiration.timestamp())
    
    # Create headers with kid
    headers = {
        "kid": key_id
    }
    
    # Create the token
    token = jwt.encode(payload, secret_key, algorithm='HS256', headers=headers)
    
    print(f"Generated token: {token}")
    print(f"Secret key used: {secret_key}")
    
    # Decode and print the token details
    decoded = jwt.decode(token, options={"verify_signature": False})
    print(f"Decoded payload: {decoded}")
    
    return token
# Example usage

    
# Generate token


# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(BlockReDocMiddleware)
# app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RestrictSwaggerMiddleware)
templates = Jinja2Templates(directory="static")

@app.get("/Macom-RBI-BOT", response_class=HTMLResponse)
async def index(request: Request):  # Missing 'request' parameter
    
    return templates.TemplateResponse("index.html", {"request": request}) 

@app.get("/redirect_chat", response_class=HTMLResponse)
async def redirect(request: Request):  # Missing 'request' parameter
    
    return templates.TemplateResponse("redirect_loader.html", {"request": request}) 


@app.get("/ldap_authenticate", response_class=RedirectResponse)
async def redirect_to_hosted_ui():
    # payload="Rbi-bot"
    payload = {
        'site': "Rbi-bot"

    }
    secret_key="9y$B&E)H@McQfTjWnZr4u7x!A%D*G-Ka"
    token = create_jwt_token(payload, secret_key)
    return f"https://mac.mactech.net.in/LdapWebPage/Login.aspx?token={token}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)