"""
OpenAI LLM client v2 for gpt-4.1-nano.

Lightweight wrapper for LangGraph integration with OpenAI's ChatOpenAI.
Configured specifically for the interview practice system.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ==================== LLM Configuration ====================

def get_llm_client(temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize OpenAI LLM client for gpt-4.1-nano.
    
    Args:
        temperature: Controls randomness (0.0-1.0)
                    0.7 = balanced creativity and consistency
    
    Returns:
        Configured ChatOpenAI instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=temperature,
        api_key=api_key,
        max_tokens=1000,  # Reasonable limit for interview responses
        streaming=False   # For MVP, no streaming needed
    )


# ==================== Message Helpers ====================

def create_system_message(content: str) -> SystemMessage:
    """Create a system message for LangGraph."""
    return SystemMessage(content=content)


def create_human_message(content: str) -> HumanMessage:
    """Create a human message for LangGraph."""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """Create an AI message for LangGraph."""
    return AIMessage(content=content)


# ==================== LLM Invocation ====================

async def invoke_llm_async(
    llm: ChatOpenAI,
    system_prompt: str,
    user_message: str
) -> str:
    """
    Async invocation of LLM with system and user messages.
    
    Args:
        llm: ChatOpenAI instance
        system_prompt: System instructions
        user_message: User's message/prompt
    
    Returns:
        LLM response as string
    """
    messages = [
        create_system_message(system_prompt),
        create_human_message(user_message)
    ]
    
    response = await llm.ainvoke(messages)
    return response.content


def invoke_llm_sync(
    llm: ChatOpenAI,
    system_prompt: str,
    user_message: str
) -> str:
    """
    Synchronous invocation of LLM with system and user messages.
    
    Args:
        llm: ChatOpenAI instance
        system_prompt: System instructions
        user_message: User's message/prompt
    
    Returns:
        LLM response as string
    """
    messages = [
        create_system_message(system_prompt),
        create_human_message(user_message)
    ]
    
    response = llm.invoke(messages)
    return response.content
