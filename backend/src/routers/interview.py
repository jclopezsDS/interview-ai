from typing import List, Optional
from fastapi import APIRouter
from src.domain.schemas import (
    InterviewRequest,
    InterviewResponse,
    ParseContextRequest,
    ParsedContext,
    InterviewType,
    DifficultyLevel,
)
from src.services.question_service import generate_question as svc_generate_question


router = APIRouter()


@router.post("/generate-question", response_model=InterviewResponse)
async def generate_question(request: InterviewRequest):
    payload = svc_generate_question(
        interview_type=request.interview_type,
        difficulty=request.difficulty,
        job_description=request.job_description,
        candidate_background=None,
    )
    return InterviewResponse(
        question=payload["question"],
        context=payload["context"],
        difficulty=request.difficulty,
        follow_up_hints=payload["follow_up_hints"],
    )


@router.post("/api/parse-context", response_model=ParsedContext)
async def parse_context(req: ParseContextRequest):
    jd = (req.jobDescription or "").lower()
    bg = (req.candidateBackground or "").lower()

    job_title = None
    for token in ["engineer", "developer", "designer", "manager", "analyst"]:
        if token in jd:
            job_title = token.title()
            break

    years = None
    for y in [10, 7, 5, 3, 2, 1]:
        if f"{y}+" in jd or f"{y} years" in jd or f"{y} años" in jd:
            years = y
            break

    skills_base = [
        "python",
        "javascript",
        "react",
        "fastapi",
        "sql",
        "tailwind",
        "typescript",
    ]
    skills = [s.title() for s in skills_base if s in jd or s in bg]

    suggested_type = "technical"
    if any(k in jd for k in ["behavioral", "soft skills", "comportamiento"]):
        suggested_type = "behavioral"
    if any(k in jd for k in ["case study", "business", "estudio de caso"]):
        suggested_type = "case_study"

    suggested_difficulty = "intermediate"
    if any(k in jd for k in ["senior", "lead", "advanced"]):
        suggested_difficulty = "advanced"
    if any(k in jd for k in ["junior", "entry", "beginner"]):
        suggested_difficulty = "beginner"

    return ParsedContext(
        jobTitle=job_title or "",
        companyName=None,
        requiredSkills=skills or ["Communication", "Problem Solving"],
        yearsOfExperience=years or 3,
        suggestedInterviewType=suggested_type,
        suggestedDifficulty=suggested_difficulty,
    )
