"""
FastAPI router v2 for interview session management.

Endpoints for creating, managing, and interacting with interview sessions.
Uses LangGraph-based interview system with in-memory storage.
"""
from typing import List
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from src.domain.models_v2 import (
    CreateSessionRequest,
    SendMessageRequest,
    SessionResponse,
    MessageResponse,
    Message,
    ErrorResponse
)
from src.services.session_manager_v2 import get_session_manager


# ==================== Router Setup ====================

router = APIRouter(
    prefix="/api/sessions",
    tags=["Interview Sessions v2"]
)


# ==================== Endpoints ====================

@router.post(
    "",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new interview session",
    description="Initialize a new interview session with job description and candidate background. Returns greeting and first question."
)
async def create_session(request: CreateSessionRequest) -> MessageResponse:
    """
    Create a new interview session.
    
    The session is initialized with:
    - Job description and candidate background (injected in system prompt)
    - Interview type and difficulty level
    - Compiled LangGraph with MemorySaver
    
    Returns the AI's greeting and first question.
    """
    try:
        manager = get_session_manager()
        
        session_id = manager.create_session(
            job_description=request.job_description,
            user_background=request.user_background,
            interview_type=request.interview_type,
            difficulty=request.difficulty
        )
        
        # Get initial state (greeting + first question)
        session_data = manager.get_session(session_id)
        
        # Extract AI's initial message (greeting + question)
        ai_messages = [msg for msg in session_data["messages"] if msg["role"] == "assistant"]
        ai_message = "\n\n".join([msg["content"] for msg in ai_messages])
        
        return MessageResponse(
            session_id=session_id,
            ai_message=ai_message,
            question_count=session_data["question_count"],
            is_complete=session_data["is_complete"],
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.post(
    "/{session_id}/message",
    response_model=MessageResponse,
    summary="Send message to session",
    description="Send user's answer/response to the interview session. Returns AI's evaluation, feedback, or next question."
)
async def send_message(session_id: str, request: SendMessageRequest) -> MessageResponse:
    """
    Send a message to an active interview session.
    
    The AI will:
    - Evaluate the answer if it's a response to a question
    - Provide feedback and ask if clarification is needed
    - Generate next question if user says "next"
    - Provide clarification + next question if user asks
    - Close interview after 6 questions
    """
    try:
        manager = get_session_manager()
        
        result = manager.send_message(session_id, request.message)
        
        return MessageResponse(
            session_id=result["session_id"],
            ai_message=result["ai_message"],
            question_count=result["question_count"],
            is_complete=result["is_complete"],
            timestamp=datetime.utcnow()
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
    description="Retrieve complete session information including conversation history and current state."
)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get session information and conversation history.
    
    Returns:
    - Session metadata (type, difficulty, etc.)
    - Complete conversation history
    - Current progress (question count, completion status)
    """
    manager = get_session_manager()
    
    session_data = manager.get_session(session_id)
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Convert message dicts to Message objects
    messages = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.utcnow()  # Note: actual timestamps not stored in MVP
        )
        for msg in session_data["messages"]
    ]
    
    return SessionResponse(
        session_id=session_data["session_id"],
        interview_type=session_data["interview_type"],
        difficulty=session_data["difficulty"],
        question_count=session_data["question_count"],
        is_active=session_data["is_active"],
        is_complete=session_data["is_complete"],
        messages=messages,
        created_at=session_data["created_at"]
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete session",
    description="Delete an interview session from memory. Session data will be permanently lost."
)
async def delete_session(session_id: str):
    """
    Delete a session from memory.
    
    This is permanent - all conversation history will be lost.
    Use when interview is complete or needs to be abandoned.
    """
    manager = get_session_manager()
    
    deleted = manager.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # 204 No Content - no response body needed
    return None
