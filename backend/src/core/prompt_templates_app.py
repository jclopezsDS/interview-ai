"""
Minimal, app-facing prompt builders for the LLM-first MVP.

Notes:
- The experimental templates and demo calls live in `prompt_templates.py` (kept for notebooks).
- This module provides lightweight, stable prompts for `QuestionNode` and `EvaluateNode`.
- No external dependencies; pure string builders and simple switching by interviewType/difficulty.
"""

from typing import Dict


_SYSTEM_PERSONAS: Dict[str, str] = {
    "technical": (
        "You are an experienced Senior Software Engineer interviewing a candidate. "
        "Assess practical skills, clear reasoning, and trade-off awareness with a concise, professional tone."
    ),
    "behavioral": (
        "You are an experienced HR Manager. Assess soft skills using STAR. Be empathetic, concise, and practical."
    ),
    "case_study": (
        "You are a Senior Business Analyst. Evaluate problem-solving and structured thinking. Keep instructions crisp."
    ),
    "general": (
        "You are a professional interviewer. Keep prompts clear, helpful, and focused on the role context."
    ),
}

_DIFFICULTY_HINTS: Dict[str, str] = {
    "easy": "Keep the question basic and approachable.",
    "intermediate": "Aim for moderate depth and realistic complexity.",
    "hard": "Increase depth, edge cases, and rigor while staying clear.",
}


def get_system_prompt(interview_type: str) -> str:
    """Return a concise system persona for the interview type.

    Falls back to "general" if unknown.
    """
    return _SYSTEM_PERSONAS.get(interview_type or "", _SYSTEM_PERSONAS["general"])  # type: ignore[index]


def build_question_prompt(
    interview_type: str,
    difficulty: str,
    job_description: str | None,
    candidate_background: str | None,
) -> str:
    """Build a minimal, deterministic prompt for generating the next interview question.

    Output expectation (model side): Return exactly one question line. Avoid preambles and lists.
    """
    persona = get_system_prompt(interview_type)
    diff_hint = _DIFFICULTY_HINTS.get(difficulty or "", _DIFFICULTY_HINTS["intermediate"])  # type: ignore[index]

    jd = (job_description or "").strip()
    cb = (candidate_background or "").strip()

    context_block = ""
    if jd:
        context_block += f"Job Description (short):\n{jd[:600]}\n\n"
    if cb:
        context_block += f"Candidate Background (short):\n{cb[:600]}\n\n"

    prompt = (
        f"[SYSTEM]\n{persona}\n\n"
        f"[INSTRUCTIONS]\n"
        f"- Interview type: {interview_type or 'general'}\n"
        f"- Difficulty: {difficulty or 'intermediate'} — {diff_hint}\n"
        f"- Produce exactly ONE clear interview question.\n"
        f"- No preamble, no numbering, no extra commentary.\n"
        f"- Keep it within 220 characters if possible.\n\n"
        f"[CONTEXT]\n{context_block if context_block else 'No extra context provided.'}\n"
        f"[OUTPUT]\nReturn only the question line."
    )
    return prompt


def build_evaluate_prompt(
    interview_type: str,
    difficulty: str,
    last_user_message: str,
    previous_question_context: str | None,
) -> str:
    """Build a prompt to produce brief feedback and a single follow-up question.

    Output expectation (model side):
    - Feedback: one or two concise sentences.
    - Follow-up: exactly one targeted question.
    """
    persona = get_system_prompt(interview_type)
    diff_hint = _DIFFICULTY_HINTS.get(difficulty or "", _DIFFICULTY_HINTS["intermediate"])  # type: ignore[index]

    uq = (previous_question_context or "").strip()
    ua = (last_user_message or "").strip()

    prompt = (
        f"[SYSTEM]\n{persona}\n\n"
        f"[INSTRUCTIONS]\n"
        f"- Interview type: {interview_type or 'general'}\n"
        f"- Difficulty: {difficulty or 'intermediate'} — {diff_hint}\n"
        f"- Provide two parts only: Feedback and Follow-up.\n"
        f"- Feedback: 1-2 sentences, specific and actionable.\n"
        f"- Follow-up: exactly ONE concise question to deepen assessment.\n"
        f"- Avoid lists, numbering, or verbose preambles.\n\n"
        f"[PREVIOUS QUESTION]\n{uq if uq else 'N/A'}\n\n"
        f"[CANDIDATE ANSWER]\n{ua[:1200]}\n\n"
        f"[OUTPUT FORMAT]\n"
        f"Feedback: <one or two sentences>\n"
        f"Follow-up: <one question>"
    )
    return prompt
