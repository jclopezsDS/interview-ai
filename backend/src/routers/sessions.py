from typing import List
from fastapi import APIRouter, HTTPException, Depends
import os
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

from src.services.storage import get_storage, StorageService
from src.domain.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    PostMessageRequest,
    ChatMessage,
)
from src.services.question_service import generate_question as svc_generate_question
from src.services.chat_service import next_ai_reply as svc_next_reply
from src.graph.interview_graph import InterviewGraph
from src.core.state_transitions import (
    initial_phase,
    next_phase_on_event,
    InterviewPhase,
)

router = APIRouter()
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
_graph = InterviewGraph() if USE_LLM else None

@router.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    storage: StorageService = Depends(get_storage),
):
    # Initialize session with base config
    base_config = req.model_dump()
    base_config.update({
        "phase": initial_phase().value,
        "turnCount": 0,
    })
    session_id = storage.create_session(base_config)

    if USE_LLM and _graph is not None:
        result = _graph.start(base_config)
        opening_text = result.get("ai_message", "Let's begin.")
        opening_msg = ChatMessage(
            id=str(uuid4()),
            sessionId=session_id,
            role="ai",
            content=opening_text,
            timestamp=datetime.utcnow(),
        )
        storage.append_message(session_id, opening_msg.model_dump())
        # Advance phase based on graph decision
        storage.update_session(session_id, {"config": {"phase": result.get("next_phase")}})
    else:
        payload = svc_generate_question(
            interview_type=req.interviewType,
            difficulty=req.difficulty,
            job_description=req.jobDescription,
            candidate_background=req.candidateBackground,
        )
        opening = (
            f"Let's begin a {req.interviewType.replace('_', ' ')} interview at {req.difficulty} level.\n"
            f"First question: {payload['question']}"
        )
        opening_msg = ChatMessage(
            id=str(uuid4()),
            sessionId=session_id,
            role="ai",
            content=opening,
            timestamp=datetime.utcnow(),
        )
        storage.append_message(session_id, opening_msg.model_dump())

    # Advance phase after seeding opening question (fallback path only)
    if not USE_LLM:
        new_phase = next_phase_on_event(InterviewPhase(base_config["phase"]), "start", turns=0).value
        storage.update_session(session_id, {"config": {"phase": new_phase}})

    return CreateSessionResponse(sessionId=session_id)


@router.get("/api/messages/{session_id}", response_model=List[ChatMessage])
async def get_messages(session_id: str, storage: StorageService = Depends(get_storage)):
    raw = storage.get_messages(session_id)
    return [ChatMessage(**m) for m in raw]


@router.post("/api/messages")
async def post_message(
    req: PostMessageRequest,
    storage: StorageService = Depends(get_storage),
):
    if not storage.session_exists(req.sessionId):
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = ChatMessage(
        id=str(uuid4()),
        sessionId=req.sessionId,
        role="user",
        content=req.content,
        timestamp=datetime.utcnow(),
    )
    storage.append_message(req.sessionId, user_msg.model_dump())

    session = storage.get_session(req.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    cfg = session.get("config", {})
    current_phase = cfg.get("phase", initial_phase().value)
    turns = int(cfg.get("turnCount", 0)) + 1
    storage.update_session(req.sessionId, {"config": {"turnCount": turns}})

    if USE_LLM and _graph is not None:
        result = _graph.on_answer(cfg, req.content, turns=turns, current_phase=current_phase)
        ai_text = result.get("ai_message", "")
        ai_msg = ChatMessage(
            id=str(uuid4()),
            sessionId=req.sessionId,
            role="ai",
            content=ai_text,
            timestamp=datetime.utcnow(),
        )
        storage.append_message(req.sessionId, ai_msg.model_dump())
        storage.update_session(req.sessionId, {"config": {"phase": result.get("next_phase")}})
        next_phase = result.get("next_phase")
    else:
        reply_content = svc_next_reply(
            interview_type=cfg.get("interviewType"),
            difficulty=cfg.get("difficulty"),
            last_user_message=req.content,
        )
        ai_msg = ChatMessage(
            id=str(uuid4()),
            sessionId=req.sessionId,
            role="ai",
            content=reply_content,
            timestamp=datetime.utcnow(),
        )
        storage.append_message(req.sessionId, ai_msg.model_dump())

        # Determine and update next phase
        next_phase = next_phase_on_event(InterviewPhase(current_phase), "answer", turns=turns).value
        storage.update_session(req.sessionId, {"config": {"phase": next_phase}})

    # If conversation concluded, append closing and mark status
    if next_phase == InterviewPhase.CONCLUSION.value:
        closing = ChatMessage(
            id=str(uuid4()),
            sessionId=req.sessionId,
            role="ai",
            content="Thanks for practicing today. This concludes the session.",
            timestamp=datetime.utcnow(),
        )
        storage.append_message(req.sessionId, closing.model_dump())
        storage.update_session(req.sessionId, {"status": "completed"})

    return {"ok": True}
