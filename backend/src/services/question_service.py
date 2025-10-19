from __future__ import annotations
from typing import List, Tuple
from src.domain.schemas import InterviewType, DifficultyLevel


def _technical_templates(level: DifficultyLevel) -> Tuple[str, str, List[str]]:
    if level == DifficultyLevel.beginner:
        return (
            "Explain the difference between var, let, and const in JavaScript.",
            "Assesses basics of scoping and reassignment.",
            ["Give a short code example.", "When would you choose each?"],
        )
    if level == DifficultyLevel.intermediate:
        return (
            "How would you design a debounced search input in React?",
            "Evaluates state management and performance considerations.",
            ["What trade-offs exist?", "How to test it?"],
        )
    return (
        "Design a scalable system to serve real-time notifications to millions of users.",
        "Assesses distributed systems, messaging, and consistency.",
        ["What components are needed?", "How to ensure reliability?"],
    )


def _behavioral_templates(level: DifficultyLevel) -> Tuple[str, str, List[str]]:
    if level == DifficultyLevel.beginner:
        return (
            "Tell me about a time you received constructive feedback.",
            "Assesses self-awareness and growth mindset.",
            ["What did you change afterward?", "What was the impact?"],
        )
    if level == DifficultyLevel.intermediate:
        return (
            "Describe a project where you had to balance quality and deadlines.",
            "Evaluates prioritization and communication.",
            ["How did you communicate risks?", "What was the outcome?"],
        )
    return (
        "Tell me about a time you influenced a decision without direct authority.",
        "Assesses leadership and stakeholder management.",
        ["What conflicting interests existed?", "What did you learn?"],
    )


def _case_study_templates(level: DifficultyLevel) -> Tuple[str, str, List[str]]:
    if level == DifficultyLevel.beginner:
        return (
            "Estimate daily active users for a simple to-do app and key metrics.",
            "Assesses product thinking and metric selection.",
            ["Which assumptions matter most?", "How to validate?"],
        )
    if level == DifficultyLevel.intermediate:
        return (
            "Increase conversion for a checkout flow with a 60% drop-off.",
            "Evaluates funnel analysis and experimentation.",
            ["What data would you collect?", "What experiment would you run?"],
        )
    return (
        "Launch a new subscription tier; outline pricing, packaging, and risks.",
        "Assesses strategy, segmentation, and trade-offs.",
        ["How to measure success?", "What are failure modes?"],
    )


def generate_question(
    interview_type: InterviewType,
    difficulty: DifficultyLevel,
    job_description: str | None,
    candidate_background: str | None,
):
    if interview_type == InterviewType.technical:
        q, ctx, hints = _technical_templates(difficulty)
    elif interview_type == InterviewType.behavioral:
        q, ctx, hints = _behavioral_templates(difficulty)
    else:
        q, ctx, hints = _case_study_templates(difficulty)

    # Optionally, bias question towards JD/background keywords in MVP (light touch)
    jd_hint = (job_description or "").strip()
    if jd_hint:
        ctx = f"{ctx} Context: role relates to '{jd_hint[:80]}'."

    return {
      "question": q,
      "context": ctx,
      "follow_up_hints": hints,
    }
