from __future__ import annotations
from src.domain.schemas import InterviewType, DifficultyLevel


def next_ai_reply(
    interview_type: InterviewType,
    difficulty: DifficultyLevel,
    last_user_message: str,
) -> str:
    base_prompt = "Thanks for your answer. "

    if interview_type == InterviewType.technical:
        follow = "Could you detail the specific technologies, trade-offs, and performance considerations?"
    elif interview_type == InterviewType.behavioral:
        follow = "What actions did you take, what alternatives did you consider, and what measurable impact did you achieve?"
    else:
        follow = "Walk me through your assumptions, the data you'd collect, and how you'd validate the chosen approach."

    if difficulty == DifficultyLevel.beginner:
        tone = "Let's keep it concise and practical. "
    elif difficulty == DifficultyLevel.intermediate:
        tone = "Please include concrete examples and reasoning. "
    else:
        tone = "Focus on trade-offs, risks, and scaling implications. "

    return f"{base_prompt}{tone}{follow}"
