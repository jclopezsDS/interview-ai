from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Interview Practice API is running"}

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "version": "0.1.0"
    }
