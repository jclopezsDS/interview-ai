"""
Minimal interview graph orchestrator for the LLM-first MVP.

- Uses state transitions from `src.core.state_transitions`.
- Uses prompts from `src.core.prompt_templates_app` and LLM from `src.core.llm_client_app`.
- Validates outputs with `src.core.validators` light helpers.

Note: This is a thin orchestrator, not a full LangGraph dependency. It can be
replaced by a LangGraph graph later with the same node responsibilities.
"""
from __future__ import annotations

from typing import Dict, Any

from src.core.state_transitions import (
    InterviewPhase,
    initial_phase,
    next_phase_on_event,
)
from src.core.prompt_templates_app import (
    get_system_prompt,
    build_question_prompt,
    build_evaluate_prompt,
)
from src.core.llm_client_app import get_llm_client_app
from src.core.validators import (
    validate_question_output,
    validate_evaluate_output,
    normalize_whitespace,
    clamp_text,
)



class InterviewGraph:
    def __init__(self) -> None:
        self.llm = get_llm_client_app()

    def _ask_llm(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.llm.generate_text(system_prompt, user_prompt)
        return resp.get("text", "").strip()

    def start(self, session_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Run the start of the interview: GREETING -> QUESTIONING (one question)."""
        sys = get_system_prompt(session_cfg.get("interviewType"))
        up = build_question_prompt(
            session_cfg.get("interviewType"),
            session_cfg.get("difficulty"),
            session_cfg.get("jobDescription"),
            session_cfg.get("candidateBackground"),
        )
        text = self._ask_llm(sys, up)
        # Validate and normalize question
        qres = validate_question_output(text)
        if qres.get("ok"):
            question = qres.get("question", "").strip()
        else:
            # Fallback: normalize and clamp
            question = clamp_text(normalize_whitespace(text or ""), 350)
        return {
            "ai_message": question,
            "next_phase": next_phase_on_event(InterviewPhase.GREETING, "start", turns=0).value,
        }

    def on_answer(self, session_cfg: Dict[str, Any], last_user_message: str, turns: int, current_phase: str) -> Dict[str, Any]:
        """Handle a user answer: produce feedback/repregunta; compute next phase."""
        sys = get_system_prompt(session_cfg.get("interviewType"))
        # Optionally we could pass the last question as context; for MVP accept None
        up = build_evaluate_prompt(
            session_cfg.get("interviewType"),
            session_cfg.get("difficulty"),
            last_user_message,
            previous_question_context=None,
        )
        text = self._ask_llm(sys, up)
        # Validate evaluation output aiming for Feedback/Follow-up
        eres = validate_evaluate_output(text)
        if eres.get("ok"):
            ai_text = f"Feedback: {eres.get('feedback','').strip()}\nFollow-up: {eres.get('follow_up','').strip()}"
        else:
            # Fallback to a clamped normalized text
            ai_text = clamp_text(normalize_whitespace(text or ""), 800)
        next_p = next_phase_on_event(InterviewPhase(current_phase), "answer", turns=turns).value
        return {"ai_message": ai_text, "next_phase": next_p}
