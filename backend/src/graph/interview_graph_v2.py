"""
LangGraph interview graph v2.

Defines the complete interview flow with nodes and conditional edges.
Implements 6-question interview with feedback and optional clarification.
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from src.core.state_v2 import InterviewState
from src.core.llm_client_v2 import get_llm_client
from src.core.prompt_templates_v2 import (
    build_system_prompt,
    get_greeting_prompt,
    get_question_prompt,
    get_evaluation_prompt,
    get_feedback_prompt,
    get_clarification_prompt,
    get_closing_prompt,
)


# ==================== Graph Nodes ====================

def greeting_node(state: InterviewState) -> dict:
    """Generate initial greeting and set up interview."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    greeting = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_greeting_prompt()}
    ]).content
    
    return {
        "messages": [AIMessage(content=greeting)],
        "question_count": 0,
        "awaiting_clarification": False,
        "is_complete": False
    }


def question_generation_node(state: InterviewState) -> dict:
    """Generate next interview question."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    question_number = state["question_count"] + 1
    question = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_question_prompt(question_number)}
    ]).content
    
    return {
        "messages": [AIMessage(content=question)],
        "question_count": question_number,
        "current_question": question,
        "awaiting_clarification": False
    }


def answer_evaluation_node(state: InterviewState) -> dict:
    """Evaluate user's answer to current question."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    # Get last user message (HumanMessage object)
    user_answer = state["messages"][-1].content
    
    evaluation = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_evaluation_prompt(user_answer, state["current_question"])}
    ]).content
    
    print(f"[EVALUATION NODE] Generated evaluation (length={len(evaluation)}): {evaluation[:200]}...")
    
    # Store evaluation in state for feedback node (as internal note, not shown to user)
    return {
        "evaluation": evaluation  # Temporary field for next node
    }


def feedback_node(state: InterviewState) -> dict:
    """Provide feedback and ask if clarification is needed."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    # Check if this was the last question
    is_last_question = state["question_count"] >= 6
    
    evaluation = state.get("evaluation", "")
    print(f"[FEEDBACK NODE] Received evaluation (length={len(evaluation)}): {evaluation[:200] if evaluation else 'EMPTY!'}...")
    
    feedback = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_feedback_prompt(evaluation, is_last_question)}
    ]).content
    
    return {
        "messages": [AIMessage(content=feedback)],
        "awaiting_clarification": not is_last_question  # Don't wait if it's the last question
    }


def clarification_node(state: InterviewState) -> dict:
    """Provide clarification and include next question."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    # Get user's clarification request (HumanMessage object)
    user_request = state["messages"][-1].content
    
    # Generate clarification + next question in one response
    clarification_with_question = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_clarification_prompt(user_request, state["current_question"])}
    ]).content
    
    question_number = state["question_count"] + 1
    
    # Extract question from response (after markers like "next question:" or "Now,")
    # If extraction fails, use whole response as context
    question_text = clarification_with_question
    for marker in ["next question:", "Now,", "Question:"]:
        if marker.lower() in clarification_with_question.lower():
            parts = clarification_with_question.lower().split(marker.lower())
            if len(parts) > 1:
                question_text = parts[-1].strip()
                break
    
    return {
        "messages": [AIMessage(content=clarification_with_question)],
        "question_count": question_number,
        "current_question": question_text,
        "awaiting_clarification": False
    }


def closing_node(state: InterviewState) -> dict:
    """Provide closing summary and thank you message."""
    llm = get_llm_client()
    
    system_prompt = build_system_prompt(
        state["interview_type"],
        state["difficulty"],
        state["job_description"],
        state["user_background"]
    )
    
    # Check if user asked for clarification before closing
    last_message = state["messages"][-1].content.lower()
    clarification_keywords = ["clarify", "clarification", "explain", "what do you mean", "elaborate", "don't understand", "unclear"]
    asked_for_clarification = any(keyword in last_message for keyword in clarification_keywords)
    
    if asked_for_clarification:
        print(f"[CLOSING NODE] Detected clarification request before closing")
        # Provide brief clarification + closing
        clarification_request = state["messages"][-1].content
        closing_prompt = f"""The candidate asked for clarification: "{clarification_request}"

Provide a brief, helpful clarification (2-3 sentences), then transition to the closing summary.

Generate a closing message including:
1. Brief clarification response
2. Thank you for participating
3. Brief overall performance summary (2-3 sentences)
4. Top 2-3 specific points to improve
5. Encouraging final statement

Keep it professional, constructive, and motivating. Total: 6-8 sentences."""
    else:
        print(f"[CLOSING NODE] Normal closing (no clarification)")
        closing_prompt = get_closing_prompt(state["question_count"])
    
    closing = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": closing_prompt}
    ]).content
    
    return {
        "messages": [AIMessage(content=closing)],
        "is_complete": True
    }


# ==================== Routing Logic ====================

def route_entry(state: InterviewState) -> Literal["greeting", "answer_evaluation", "clarification", "question_generation", "closing"]:
    """
    Main entry router - determines which node to execute based on state.
    
    Logic:
    - First invocation (no messages) → greeting
    - User answered question (not awaiting clarification) → answer_evaluation
    - User responded to feedback:
        - Wants clarification → clarification
        - Wants next question → question_generation
        - 6 questions done → closing
    """
    print(f"[ROUTING] question_count={state.get('question_count', 0)}, awaiting_clarification={state.get('awaiting_clarification', False)}, is_complete={state.get('is_complete', False)}")
    
    # First invocation
    if not state.get("messages") or len(state["messages"]) == 0:
        print("[ROUTING] → greeting (first invocation)")
        return "greeting"
    
    # Check if interview is complete
    if state.get("is_complete"):
        print("[ROUTING] → closing (is_complete=True)")
        return "closing"
    
    last_message = state["messages"][-1].content.lower()
    
    # Check for clarification request at any time (even after question 6)
    clarification_keywords = ["clarify", "clarification", "explain", "what do you mean", "elaborate", "don't understand", "unclear"]
    wants_clarification = any(keyword in last_message for keyword in clarification_keywords)
    
    # Special case: After question 6, clarification goes to closing_with_clarification
    if state["question_count"] >= 6 and wants_clarification:
        print("[ROUTING] → closing (clarification request after question 6)")
        return "closing"
    
    # Check if 6 questions done and user wants to finish
    if state["question_count"] >= 6 and not state.get("awaiting_clarification"):
        finish_keywords = ["done", "finish", "complete", "end"]
        wants_to_finish = any(keyword in last_message for keyword in finish_keywords)
        
        if wants_to_finish:
            print("[ROUTING] → closing (user wants to finish after 6 questions)")
            return "closing"
    
    # Check if we're awaiting clarification decision
    if state.get("awaiting_clarification"):
        print(f"[ROUTING] awaiting_clarification=True, last_message={last_message[:100]}...")
        
        # Check if user wants clarification (already computed above)
        # clarification_keywords and wants_clarification already set
        
        if wants_clarification:
            print("[ROUTING] → clarification (user wants clarification)")
            return "clarification"
        
        # Check if 6 questions done
        if state["question_count"] >= 6:
            print("[ROUTING] → closing (6 questions completed)")
            return "closing"
        
        # User wants next question
        print("[ROUTING] → question_generation (user wants next)")
        return "question_generation"
    
    # User answered a question, needs evaluation
    print("[ROUTING] → answer_evaluation (user answered question)")
    return "answer_evaluation"


# ==================== Build Graph ====================

def create_interview_graph() -> StateGraph:
    """
    Build the complete interview graph with all nodes and edges.
    
    Flow (each invoke processes user input and generates AI response):
    
    Invoke 1 (no user message): 
      START → greeting → question_generation → END
      Returns: Greeting + First question
    
    Invoke 2 (user answers question):
      answer_evaluation → feedback → END
      Returns: Evaluation + Feedback + "Clarify or next?"
    
    Invoke 3a (user says "next"):
      question_generation → END
      Returns: Next question
    
    Invoke 3b (user says "clarify"):
      clarification → END
      Returns: Clarification + Next question
    
    Invoke N (after 6 questions):
      closing → END
      Returns: Summary + Goodbye
    """
    # Initialize graph
    graph = StateGraph(InterviewState)
    
    # Add nodes
    graph.add_node("greeting", greeting_node)
    graph.add_node("question_generation", question_generation_node)
    graph.add_node("answer_evaluation", answer_evaluation_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("closing", closing_node)
    
    # Entry routing from START
    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "greeting": "greeting",
            "answer_evaluation": "answer_evaluation",
            "clarification": "clarification",
            "question_generation": "question_generation",
            "closing": "closing"
        }
    )
    
    # After greeting, generate first question
    graph.add_edge("greeting", "question_generation")
    
    # All nodes return to END (wait for next user input / next invoke)
    graph.add_edge("question_generation", END)
    graph.add_edge("answer_evaluation", "feedback")
    graph.add_edge("feedback", END)
    graph.add_edge("clarification", END)
    graph.add_edge("closing", END)
    
    return graph


def compile_interview_graph() -> StateGraph:
    """
    Compile the graph with MemorySaver checkpointer.
    
    Returns:
        Compiled graph ready for invocation
    """
    graph = create_interview_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
