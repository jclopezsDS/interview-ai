from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class InterviewType(str, Enum):
    technical = "technical"
    behavioral = "behavioral"
    case_study = "case_study"


class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class ParseContextRequest(BaseModel):
    jobDescription: str
    candidateBackground: str
    companyInfo: Optional[str] = None
    roleDetails: Optional[str] = None


class ParsedContext(BaseModel):
    jobTitle: Optional[str] = None
    companyName: Optional[str] = None
    requiredSkills: Optional[List[str]] = None
    yearsOfExperience: Optional[int] = None
    suggestedInterviewType: Optional[str] = None
    suggestedDifficulty: Optional[str] = None


class InterviewRequest(BaseModel):
    interview_type: InterviewType
    difficulty: DifficultyLevel
    focus_area: Optional[str] = None
    job_description: Optional[str] = None


class InterviewResponse(BaseModel):
    question: str
    context: str
    difficulty: DifficultyLevel
    follow_up_hints: List[str] = []


class CreateSessionRequest(BaseModel):
    jobDescription: str
    candidateBackground: str
    companyInfo: Optional[str] = None
    roleDetails: Optional[str] = None
    interviewType: InterviewType
    difficulty: DifficultyLevel


class CreateSessionResponse(BaseModel):
    sessionId: str


class PostMessageRequest(BaseModel):
    sessionId: str
    content: str


class ChatMessage(BaseModel):
    id: str
    sessionId: str
    role: str  # "ai" | "user"
    content: str
    timestamp: datetime
