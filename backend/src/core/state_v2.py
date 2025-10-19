"""
InterviewState definition for LangGraph v2.

Defines the state schema that flows through the interview graph nodes.
Uses LangGraph's MessagesState as base for conversation history.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class InterviewState(MessagesState):
    """
    State schema for the interview graph.
    
    Inherits from MessagesState to automatically handle conversation history
    with Simple Buffer Memory (all messages passed in context).
    
    Custom fields track interview progress and metadata.
    """
    
    # Session identification
    session_id: str
    
    # Context (injected in system prompt, stored for reference)
    job_description: str
    user_background: str
    
    # Interview configuration
    interview_type: str  # "Technical" | "Behavioral" | "Case Study"
    difficulty: str      # "Beginner" | "Intermediate" | "Advanced"
    
    # Progress tracking
    question_count: int  # Current question number (max 6)
    
    # Conversation state
    current_question: str         # Last question asked
    awaiting_clarification: bool  # True if waiting for clarification response
    evaluation: str               # Temporary storage for evaluation between nodes
    
    # Control flag
    is_complete: bool  # True after closing node


# Type alias for reducer annotations (if needed for custom reducers)
def add_messages(left: list, right: list) -> list:
    """
    Default message reducer (already handled by MessagesState).
    
    Appends new messages to the message list.
    """
    return left + right
