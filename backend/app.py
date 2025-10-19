"""
FastAPI application entrypoint (orchestration only).

Responsibilities:
- Load environment variables
- Instantiate FastAPI app
- Configure middleware (e.g., CORS)
- Register routers from modular packages

All business logic and endpoints live in dedicated modules under `src/`.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables BEFORE importing routers
load_dotenv()

# Routers
from src.routers import health, interview, sessions


# Create FastAPI app
app = FastAPI(
    title="Interview Practice API",
    description="AI-powered interview preparation backend",
    version="0.1.0",
)


# CORS configuration (configurable by env)
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(health.router)
app.include_router(interview.router)
app.include_router(sessions.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)