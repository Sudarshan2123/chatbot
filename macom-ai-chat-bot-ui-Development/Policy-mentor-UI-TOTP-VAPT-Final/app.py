from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
import jwt
from datetime import datetime, timedelta
from ui_security import SecurityHeadersMiddleware, RestrictSwaggerMiddleware, BlockReDocMiddleware, BruteForceMiddleware


# Simple brute force protection middleware


def create_jwt_token(payload, secret_key, key_id="key1"):
    expiration = datetime.utcnow() + timedelta(hours=24)
    payload['exp'] = int(expiration.timestamp())
    headers = {
        "kid": key_id
    }
    token = jwt.encode(payload, secret_key, algorithm='HS256', headers=headers)   
    print(f"Generated token: {token}")
    print(f"Secret key used: {secret_key}")
    
    decoded = jwt.decode(token, options={"verify_signature": False})
    print(f"Decoded payload: {decoded}")
    
    return token


app = FastAPI(docs_url=None, redoc_url=None)

# Add the brute force middleware first


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RestrictSwaggerMiddleware)
app.add_middleware(BlockReDocMiddleware)
#app.add_middleware(BruteForceMiddleware)

templates = Jinja2Templates(directory="static")

@app.get("/Macom-policy-mentor", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "nonce": request.state.nonce
    })

@app.get("/redirect_chat", response_class=HTMLResponse)
async def redirect(request: Request):
    return templates.TemplateResponse("redirect_loader.html", {
        "request": request,
        "nonce": request.state.nonce
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "nonce": request.state.nonce
    })


@app.get("/ldap_authenticate", response_class=RedirectResponse)
async def redirect_to_hosted_ui():
    # payload="Rbi-bot"
    payload = {
        'site': "policy-mentor"
    }
    secret_key="9y$B&E)H@McQfTjWnZr4u7x!A%D*G-Ka"
    token = create_jwt_token(payload, secret_key)
    return f"https://mac.mactech.net.in/LdapWebPage/Login.aspx?token={token}"


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5001)
