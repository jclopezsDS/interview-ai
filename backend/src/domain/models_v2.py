"""
Pydantic schemas v2 for Interview Practice API.

Defines request/response models for the LangGraph-based interview system.
"""
from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Request Schemas ====================

class CreateSessionRequest(BaseModel):
    """Request to create a new interview session."""
    
    job_description: str = Field(
        ...,
        description="Job description for the interview",
        min_length=10,
        max_length=5000,
        examples=["Senior Python Developer with 5+ years of experience in FastAPI and LangChain"]
    )
    
    user_background: str = Field(
        ...,
        description="Candidate's background and experience",
        min_length=10,
        max_length=5000,
        examples=["3 years as Backend Developer, familiar with Python and REST APIs"]
    )
    
    interview_type: Literal["Technical", "Behavioral", "Case Study"] = Field(
        ...,
        description="Type of interview to conduct"
    )
    
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        ...,
        description="Difficulty level of the interview"
    )


class SendMessageRequest(BaseModel):
    """Request to send a user message in an active session."""
    
    message: str = Field(
        ...,
        description="User's message/answer",
        min_length=1,
        max_length=5000,
        examples=["I would use a dictionary to solve this problem because..."]
    )


# ==================== Response Schemas ====================

class Message(BaseModel):
    """Individual message in the conversation."""
    
    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Message sender role"
    )
    
    content: str = Field(
        ...,
        description="Message content"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was created"
    )


class SessionResponse(BaseModel):
    """Response containing session information and conversation history."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier (UUID)"
    )
    
    interview_type: str = Field(
        ...,
        description="Type of interview"
    )
    
    difficulty: str = Field(
        ...,
        description="Difficulty level"
    )
    
    question_count: int = Field(
        default=0,
        description="Number of questions asked so far"
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether the session is still active"
    )
    
    is_complete: bool = Field(
        default=False,
        description="Whether the interview is complete (after 6 questions)"
    )
    
    messages: List[Message] = Field(
        default_factory=list,
        description="Conversation history"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session was created"
    )


class MessageResponse(BaseModel):
    """Response after sending a message."""
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    
    ai_message: str = Field(
        ...,
        description="AI's response message"
    )
    
    question_count: int = Field(
        ...,
        description="Current question count"
    )
    
    is_complete: bool = Field(
        default=False,
        description="Whether the interview is complete (after 6 questions)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Session ID if applicable"
    )
