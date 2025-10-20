"""
FastAPI application v2 - Interview Practice System.

LangGraph-based interview system with in-memory session management.
Uses gpt-4.1-nano for AI-powered interview conversations.
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Import routers
from src.routers.sessions_v2 import router as sessions_router


# ==================== Rate Limiting ====================

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit configuration
RATE_LIMIT_SESSIONS = "10/hour"  # Max 10 session creations per hour
RATE_LIMIT_MESSAGES = "60/hour"   # Max 60 messages per hour


# ==================== FastAPI App ====================

app = FastAPI(
    title="Interview Practice API v2",
    description="AI-powered interview preparation system using LangGraph and OpenAI gpt-4.1-nano",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter state to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ==================== CORS Configuration ====================

# Get allowed origins from environment (fallback to localhost for development)
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Routers ====================

app.include_router(sessions_router)


# ==================== Health Check ====================

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Check if the API is running"
)
async def health_check():
    """
    Simple health check endpoint.
    
    Returns API status and version.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "service": "Interview Practice API v2"
    }


@app.get(
    "/",
    tags=["Root"],
    summary="API root",
    description="API information and documentation links"
)
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Interview Practice API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    Validates environment configuration on startup.
    """
    # Validate OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables!")
        print("   The application will fail when creating sessions.")
    else:
        print("✅ OpenAI API key loaded")
    
    print(f"🚀 Interview Practice API v2 started")
    print(f"   Allowed origins: {allow_origins}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    """
    print("👋 Interview Practice API v2 shutting down")


# ==================== Run Configuration ====================

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "app_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
