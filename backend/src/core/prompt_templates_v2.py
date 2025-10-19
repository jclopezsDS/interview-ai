"""
Prompt templates v2 for LangGraph interview nodes.

Dynamic system and user prompts for each node in the interview graph.
Includes context injection (job description + user background).
"""


def build_system_prompt(
    interview_type: str,
    difficulty: str,
    job_description: str,
    user_background: str
) -> str:
    """
    Build dynamic system prompt with context injection.
    
    This prompt is set once at graph initialization and provides
    persistent context for all LLM calls.
    """
    return f"""You are an expert AI interviewer conducting a {interview_type} interview at {difficulty} level.

JOB DESCRIPTION:
{job_description}

CANDIDATE BACKGROUND:
{user_background}

YOUR ROLE:
- Ask relevant, insightful questions based on the job requirements and candidate's background
- Evaluate answers thoroughly and provide constructive feedback
- Adapt difficulty to match the {difficulty} level
- Be professional, encouraging, and fair
- Keep questions focused and clear

INTERVIEW FLOW:
1. Start with a warm greeting
2. Ask exactly 6 questions total
3. After each answer: evaluate, give feedback, offer clarification if needed
4. End with a summary and improvement points

Remember: You're helping the candidate improve, not just testing them."""


# ==================== Node-specific Prompts ====================

def get_greeting_prompt() -> str:
    """Prompt for greeting node."""
    return """Generate a warm, professional greeting to start the interview.

Include:
- Welcome message
- Brief explanation of the interview format (6 questions with feedback)
- Encouragement to answer thoughtfully
- Mention they can ask for clarification after each question

Keep it concise (3-4 sentences)."""


def get_question_prompt(question_number: int) -> str:
    """Prompt for question generation node."""
    return f"""Generate interview question #{question_number} of 6.

Requirements:
- Relevant to the job description and candidate's background
- Appropriate for the specified difficulty level
- Clear and specific
- Allows for detailed response
- Different from previous questions

Format: Just the question, no preamble."""


def get_evaluation_prompt(user_answer: str, current_question: str) -> str:
    """Prompt for answer evaluation node."""
    return f"""Evaluate the candidate's answer to this question:

QUESTION: {current_question}

ANSWER: {user_answer}

Provide a thorough evaluation considering:
- Accuracy and correctness
- Depth of understanding
- Communication clarity
- Relevance to the question

Be specific and constructive. Format as a brief paragraph."""


def get_feedback_prompt(evaluation: str, is_last_question: bool = False) -> str:
    """Prompt for feedback node."""
    closing_question = (
        "That was the final question. Type 'done' or 'finish' when you're ready to see your overall summary." 
        if is_last_question 
        else "Would you like me to clarify anything, or shall we move to the next question?"
    )
    
    return f"""Based on this evaluation:

{evaluation}

Provide feedback to the candidate including:
1. What they did well (specific strengths)
2. What could be improved (concrete suggestions)
3. At the end, {'inform them: "' + closing_question + '"' if is_last_question else 'ask: "' + closing_question + '"'}

Keep feedback constructive and encouraging. Format as 2-3 short paragraphs."""


def get_clarification_prompt(user_request: str, current_question: str) -> str:
    """Prompt for clarification node."""
    return f"""The candidate asked for clarification about this question:

QUESTION: {current_question}

CLARIFICATION REQUEST: {user_request}

Provide:
1. Clear, helpful clarification (2-3 sentences)
2. Then immediately transition to the next question by saying something like "Now, let's move on to the next question:" followed by the new question

Be concise but thorough in your clarification."""


def get_closing_prompt(question_count: int) -> str:
    """Prompt for closing node."""
    return f"""The interview is complete. The candidate answered {question_count} questions.

Generate a closing message including:
1. Thank you for participating
2. Brief overall performance summary (2-3 sentences)
3. Top 2-3 specific points to improve
4. Encouraging final statement

Keep it professional, constructive, and motivating. Total: 4-5 sentences."""


# ==================== Helper Functions ====================

def format_conversation_context(messages: list) -> str:
    """
    Format recent conversation history for context.
    
    Used when nodes need to reference previous exchanges.
    """
    if not messages:
        return "No previous conversation."
    
    context = []
    for msg in messages[-6:]:  # Last 3 exchanges (user + assistant pairs)
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        context.append(f"{role.upper()}: {content}")
    
    return "\n\n".join(context)
