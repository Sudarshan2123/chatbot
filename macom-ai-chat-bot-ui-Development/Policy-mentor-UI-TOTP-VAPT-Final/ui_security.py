from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import FastAPI, Request
import time
import secrets
from fastapi.responses import HTMLResponse
from typing import Dict, Tuple
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Generate a new nonce for each request
        nonce = secrets.token_urlsafe(32)
        
        # Store nonce in request state for template access
        request.state.nonce = nonce
        
        response = await call_next(request)
        response.headers['Server'] = 'Frontend'  
        response.headers['X-Powered-By'] = ''
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Expect-CT'] = 'max-age=86400, enforce'
        response.headers['X-XSS-Protection'] = '1; mode=block'
#        response.headers['Clear-Site-Data'] = '"cache", "cookies", "storage"'
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers['Permissions-Policy'] = 'geolocation=(), camera=()'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            f"script-src 'strict-dynamic' 'nonce-{nonce}' ; "
            "style-src 'self' ;"
            "connect-src 'self' https://policy-mentor-446976656513.us-central1.run.app ; "
            "base-uri 'self';"
        )
        # response.headers['Content-Security-Policy'] = (
                                            
        #                                             "default-src 'self'; "
        #                                             "script-src 'strict-dynamic' 'sha256-u7xQcZ5006nQiKGPjNRm+65hVtjt01gq+air62f5s3Q=' 'sha256-Ady8Dn6VdiUv6Javn9piXFbW8mUSlM0taAAOwZWC53s=';" 
        #                                             "script-src-elem https://bot.mactech.net.in 'sha256-u7xQcZ5006nQiKGPjNRm+65hVtjt01gq+air62f5s3Q=' 'sha256-Ady8Dn6VdiUv6Javn9piXFbW8mUSlM0taAAOwZWC53s=' ;"
        #                                             "style-src 'self'; "  # semicolon separating directives
        #                                             "base-uri 'self';"
        #                                             # "connect-src 'self' http://localhost:5050;"
        #                                             "connect-src 'self' https://retrieval-app-v-4-10-476055803082.us-central1.run.app;"
        #                                         )
        response.headers["NEL"] = '{"report_to": "default", "max_age": 31536000, "include_subdomains": true}'
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        return response

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


class BruteForceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_attempts: int = 5, lockout_time: int = 300, rate_limit_requests: int = 30, rate_limit_duration: int = 60):
        super().__init__(app)
        # Brute force protection settings
        self.max_attempts = max_attempts
        self.lockout_time = lockout_time  # in seconds
        self.failed_attempts: Dict[str, Tuple[int, float]] = {}  # IP -> (attempts, last_attempt_time)
        self.locked_ips: Dict[str, float] = {}  # IP -> lock_expiry_time
        
        # Rate limiting settings
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_duration = rate_limit_duration
        self.request_records: Dict[str, list] = {}  # IP -> [timestamp1, timestamp2, ...]
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean up expired locks
        self.locked_ips = {ip: expiry for ip, expiry in self.locked_ips.items() if expiry > current_time}
        
        # Check if IP is locked out
        if client_ip in self.locked_ips:
            remaining = int(self.locked_ips[client_ip] - current_time)
            return HTMLResponse(
                content=f"<html><body><h1>Too many failed attempts</h1>"
                f"<p>Your IP has been temporarily blocked due to too many failed attempts. "
                f"Please try again in {remaining} seconds.</p></body></html>",
                status_code=429,
                headers={"Retry-After": str(remaining)}
            )
        
        # Implement rate limiting for all endpoints
        if client_ip not in self.request_records:
            self.request_records[client_ip] = []
            
        # Remove timestamps older than the window
        self.request_records[client_ip] = [
            ts for ts in self.request_records[client_ip] 
            if current_time - ts < self.rate_limit_duration
        ]
        
        # Check if rate limit is exceeded
        if len(self.request_records[client_ip]) >= self.rate_limit_requests:
            return HTMLResponse(
                content="<html><body><h1>Rate limit exceeded</h1>"
                "<p>Too many requests in a short period. Please try again later.</p></body></html>",
                status_code=429,
                headers={"Retry-After": str(self.rate_limit_duration)}
            )
        
        # Record this request
        self.request_records[client_ip].append(current_time)
        
        # Process the request
        response = await call_next(request)
        
        # For ldap_authenticate endpoint, track failed attempts
        path = request.url.path
        if path == "/ldap_authenticate":
            # A 302 redirect is a successful authentication in this case
            if response.status_code == 307 or response.status_code == 302:
                # Successful authentication - reset failed attempts
                if client_ip in self.failed_attempts:
                    del self.failed_attempts[client_ip]
            else:
                # Failed authentication - increment counter
                attempts, _ = self.failed_attempts.get(client_ip, (0, current_time))
                attempts += 1
                self.failed_attempts[client_ip] = (attempts, current_time)
                
                # Lock if max attempts exceeded
                if attempts >= self.max_attempts:
                    self.locked_ips[client_ip] = current_time + self.lockout_time
                    if client_ip in self.failed_attempts:
                        del self.failed_attempts[client_ip]
            
        return response