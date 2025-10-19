"""
Session manager v2 for in-memory session storage.

Manages interview sessions with compiled LangGraph graphs and MemorySaver.
Each session has a unique ID and maintains its own conversation state.
"""
import uuid
from typing import Dict, Optional, Any
from datetime import datetime

from langchain_core.messages import HumanMessage

from src.graph.interview_graph_v2 import compile_interview_graph
from src.core.state_v2 import InterviewState


# ==================== In-Memory Storage ====================

class SessionManager:
    """
    Manages interview sessions in memory.
    
    Each session contains:
    - Compiled LangGraph graph
    - Session metadata (type, difficulty, etc.)
    - Conversation state (handled by graph's MemorySaver)
    """
    
    def __init__(self):
        # {session_id: {"graph": compiled_graph, "metadata": dict, "created_at": datetime}}
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(
        self,
        job_description: str,
        user_background: str,
        interview_type: str,
        difficulty: str
    ) -> str:
        """
        Create a new interview session.
        
        Args:
            job_description: Job description for context
            user_background: Candidate's background
            interview_type: Type of interview
            difficulty: Difficulty level
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        # Compile graph for this session
        graph = compile_interview_graph()
        
        # Store session
        self._sessions[session_id] = {
            "graph": graph,
            "metadata": {
                "session_id": session_id,
                "job_description": job_description,
                "user_background": user_background,
                "interview_type": interview_type,
                "difficulty": difficulty,
                "created_at": datetime.utcnow(),
                "is_active": True
            },
            "thread_id": session_id  # Use session_id as thread_id for checkpointing
        }
        
        # Initialize session with first invocation (greeting + first question)
        self._invoke_graph(session_id, None)
        
        return session_id
    
    def send_message(self, session_id: str, user_message: str) -> Dict:
        """
        Send user message to a session and get AI response.
        
        Args:
            session_id: Session identifier
            user_message: User's message/answer
        
        Returns:
            Response dict with AI message and session state
        
        Raises:
            ValueError: If session not found or inactive
            RuntimeError: If graph invocation fails
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._sessions[session_id]
        
        if not session["metadata"]["is_active"]:
            raise ValueError(f"Session {session_id} is no longer active")
        
        # Check if interview is already complete
        graph = session["graph"]
        thread_id = session["thread_id"]
        current_state = graph.get_state({"configurable": {"thread_id": thread_id}})
        
        if current_state.values and current_state.values.get("is_complete"):
            raise ValueError(f"Session {session_id} is already complete. Start a new interview.")
        
        # Invoke graph with user message
        try:
            result = self._invoke_graph(session_id, user_message)
            return result
        except Exception as e:
            print(f"[ERROR] Graph invocation failed for session {session_id}: {e}")
            raise RuntimeError(f"Failed to process message: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session information and conversation history.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data dict or None if not found
        """
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        graph = session["graph"]
        thread_id = session["thread_id"]
        
        # Get current state from graph
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        
        # Debug
        print(f"[DEBUG get_session] state.values type: {type(state.values)}")
        print(f"[DEBUG get_session] state.values keys: {state.values.keys() if hasattr(state.values, 'keys') else 'not a dict'}")
        print(f"[DEBUG get_session] messages in state: {len(state.values.get('messages', []))}")
        
        return {
            **session["metadata"],
            "messages": [
                {
                    "role": "assistant" if (hasattr(msg, 'type') and msg.type == "ai") or msg.__class__.__name__ == "AIMessage" 
                           else ("user" if (hasattr(msg, 'type') and msg.type == "human") or msg.__class__.__name__ == "HumanMessage"
                           else msg.type if hasattr(msg, 'type') else "unknown"),
                    "content": msg.content,
                }
                for msg in state.values.get("messages", [])
            ],
            "question_count": state.values.get("question_count", 0),
            "is_complete": state.values.get("is_complete", False)
        }
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from memory.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            self._sessions[session_id]["metadata"]["is_active"] = False
            del self._sessions[session_id]
            return True
        return False
    
    def _invoke_graph(self, session_id: str, user_message: Optional[str]) -> Dict:
        """
        Internal method to invoke the graph with optional user message.
        
        Args:
            session_id: Session identifier
            user_message: User message (None for initial invocation)
        
        Returns:
            Result dict with AI response
        """
        session = self._sessions[session_id]
        graph = session["graph"]
        metadata = session["metadata"]
        thread_id = session["thread_id"]
        
        # Get current state
        current_state = graph.get_state({"configurable": {"thread_id": thread_id}})
        
        # Build input state
        if current_state.values:
            # Session already started, add user message
            input_state = {
                "messages": [HumanMessage(content=user_message)] if user_message else []
            }
        else:
            # First invocation - initialize state
            input_state = {
                "session_id": session_id,
                "job_description": metadata["job_description"],
                "user_background": metadata["user_background"],
                "interview_type": metadata["interview_type"],
                "difficulty": metadata["difficulty"],
                "question_count": 0,
                "current_question": "",
                "awaiting_clarification": False,
                "evaluation": "",
                "is_complete": False,
                "messages": []
            }
        
        # Invoke graph
        result = graph.invoke(
            input_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Debug: print result structure
        print(f"[DEBUG] Graph result keys: {result.keys() if hasattr(result, 'keys') else 'not a dict'}")
        print(f"[DEBUG] Messages in result: {len(result.get('messages', []))} messages")
        
        # Extract AI response (last message)
        ai_message = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            ai_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            print(f"[DEBUG] Extracted AI message length: {len(ai_message)}")
        
        return {
            "session_id": session_id,
            "ai_message": ai_message,
            "question_count": result.get("question_count", 0),
            "is_complete": result.get("is_complete", False)
        }


# ==================== Global Instance ====================

# Singleton instance for the application
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return _session_manager
