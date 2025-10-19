"""
Conversation state management for interview sessions.

This module handles the creation, persistence, and management of conversation
states during interview practice sessions.
"""

import json
import gzip
import pickle
import random
import sqlite3
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional
import statistics
import time
import uuid
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import asyncio
import concurrent.futures
from functools import lru_cache
import weakref
import csv
import zipfile
from io import StringIO, BytesIO


@dataclass
class ConversationState:
    """Represents the state of an interview conversation session."""
    session_id: str
    current_phase: str
    candidate_profile: Dict[str, Any]
    interview_history: List[Dict[str, Any]]
    context_data: Dict[str, Any]
    created_at: str
    updated_at: str


def setup_conversation_engine(storage_path: str = "conversation_data") -> Dict[str, Any]:
    """Initialize conversation state manager with memory persistence and context tracking.
    
    Args:
        storage_path: Directory path for storing conversation states
        
    Returns:
        Dict[str, Any]: Conversation engine configuration and management functions
    """
    print(f"[CONV] Initializing conversation engine with storage: {storage_path}")
    
    storage_dir = Path(storage_path)
    storage_dir.mkdir(exist_ok=True)
    
    # Trace collection system
    trace_buffer = deque(maxlen=1000)
    trace_lock = Lock()
    
    def initialize_database(db_path: str = "traces.db") -> None:
        """Initialize SQLite database with optimized schema."""
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    trace_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON traces(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_operation ON traces(operation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON traces(start_time)")
        
        print("[STORAGE] Database schema initialized with optimized indexes")
    
    def store_trace(trace_data: Dict[str, Any], db_path: str = "traces.db") -> bool:
        """Store trace data with compression."""
        try:
            compressed_data = gzip.compress(pickle.dumps(trace_data))
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO traces 
                    (trace_id, session_id, operation, start_time, end_time, duration_ms, status, trace_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_data["trace_id"],
                    trace_data["session_id"],
                    trace_data["operation"],
                    trace_data["start_time"],
                    trace_data.get("end_time"),
                    trace_data.get("duration_ms"),
                    trace_data["status"],
                    compressed_data
                ))
            
            print(f"[STORAGE] Stored trace: {trace_data['trace_id']}")
            return True
            
        except Exception as e:
            print(f"[STORAGE] Error storing trace: {e}")
            return False
    
    def retrieve_trace(trace_id: str, db_path: str = "traces.db") -> Optional[Dict[str, Any]]:
        """Retrieve specific trace by ID."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT trace_data FROM traces WHERE trace_id = ?", 
                    (trace_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    decompressed_data = gzip.decompress(row[0])
                    trace_data = pickle.loads(decompressed_data)
                    print(f"[STORAGE] Retrieved trace: {trace_id}")
                    return trace_data
                
        except Exception as e:
            print(f"[STORAGE] Error retrieving trace: {e}")
        
        return None
    
    def get_session_traces(session_id: str, limit: int = 1000, db_path: str = "traces.db") -> List[Dict[str, Any]]:
        """Get all traces for a session with limit."""
        traces = []
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT trace_data FROM traces 
                    WHERE session_id = ? 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (session_id, limit))
                
                for row in cursor.fetchall():
                    decompressed_data = gzip.decompress(row[0])
                    trace_data = pickle.loads(decompressed_data)
                    traces.append(trace_data)
                    
        except Exception as e:
            print(f"[STORAGE] Error retrieving session traces: {e}")
        
        print(f"[STORAGE] Retrieved {len(traces)} traces for session: {session_id}")
        return traces
    
    def query_traces_by_operation(operation: str, limit: int = 500, db_path: str = "traces.db") -> List[Dict[str, Any]]:
        """Query traces by operation type."""
        traces = []
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT trace_data FROM traces 
                    WHERE operation = ? 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (operation, limit))
                
                for row in cursor.fetchall():
                    decompressed_data = gzip.decompress(row[0])
                    trace_data = pickle.loads(decompressed_data)
                    traces.append(trace_data)
                    
        except Exception as e:
            print(f"[STORAGE] Error querying traces by operation: {e}")
        
        print(f"[STORAGE] Retrieved {len(traces)} traces for operation: {operation}")
        return traces
    
    def analyze_performance_patterns(time_window_hours: int = 24, db_path: str = "traces.db") -> Dict[str, Any]:
        """Analyze performance patterns from stored traces."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT operation, AVG(duration_ms), COUNT(*), MIN(duration_ms), MAX(duration_ms)
                    FROM traces 
                    WHERE start_time > ? AND duration_ms IS NOT NULL
                    GROUP BY operation
                """, (cutoff_time,))
                
                patterns = {}
                for row in cursor.fetchall():
                    operation, avg_duration, count, min_duration, max_duration = row
                    patterns[operation] = {
                        "avg_duration_ms": round(avg_duration, 2),
                        "total_operations": count,
                        "min_duration_ms": min_duration,
                        "max_duration_ms": max_duration
                    }
                
        except Exception as e:
            print(f"[STORAGE] Error analyzing patterns: {e}")
            return {}
        
        print(f"[PATTERN] Analyzed {len(patterns)} operation patterns over {time_window_hours}h")
        return patterns
    
    def cleanup_old_traces(days_to_keep: int = 30, db_path: str = "traces.db") -> int:
        """Clean up traces older than specified days."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM traces WHERE start_time < ?", 
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                
        except Exception as e:
            print(f"[STORAGE] Error during cleanup: {e}")
            return 0
        
        print(f"[STORAGE] Cleaned up {deleted_count} traces older than {days_to_keep} days")
        return deleted_count
    
    def get_storage_stats(db_path: str = "traces.db") -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM traces")
                total_traces = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM traces")
                unique_sessions = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT operation) FROM traces")
                unique_operations = cursor.fetchone()[0]
                
                db_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
                
        except Exception as e:
            print(f"[STORAGE] Error getting stats: {e}")
            return {}
        
        stats = {
            "total_traces": total_traces,
            "unique_sessions": unique_sessions,
            "unique_operations": unique_operations,
            "database_size_mb": round(db_size / (1024 * 1024), 2)
        }
        
        print(f"[STORAGE] Stats: {total_traces} traces, {unique_sessions} sessions")
        return stats
    
    def start_trace(session_id: str, operation: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new trace for an operation."""
        trace_id = f"{session_id}_{operation}_{int(time.time() * 1000)}"
        
        trace_entry = {
            "trace_id": trace_id,
            "session_id": session_id,
            "operation": operation,
            "start_time": time.time(),
            "end_time": None,
            "duration_ms": None,
            "status": "started",
            "metadata": metadata or {},
            "decision_points": [],
            "performance_metrics": {},
            "error_info": None
        }
        
        with trace_lock:
            trace_buffer.append(trace_entry)
        
        print(f"[TRACE] Started: {operation} [{trace_id}]")
        return trace_id
    
    def add_decision_point(trace_id: str, decision: str, reasoning: str, alternatives: List[str] = None) -> None:
        """Add decision point to trace."""
        decision_entry = {
            "timestamp": time.time(),
            "decision": decision,
            "reasoning": reasoning,
            "alternatives": alternatives or [],
            "confidence": 1.0
        }
        
        with trace_lock:
            for trace in reversed(trace_buffer):
                if trace["trace_id"] == trace_id:
                    trace["decision_points"].append(decision_entry)
                    break
        
        print(f"[TRACE] Decision recorded: {decision}")
    
    def add_performance_metric(trace_id: str, metric_name: str, value: float, unit: str = "") -> None:
        """Add performance metric to trace."""
        with trace_lock:
            for trace in reversed(trace_buffer):
                if trace["trace_id"] == trace_id:
                    trace["performance_metrics"][metric_name] = {
                        "value": value,
                        "unit": unit,
                        "timestamp": time.time()
                    }
                    break
        
        print(f"[TRACE] Metric recorded: {metric_name} = {value}{unit}")
    
    def end_trace(trace_id: str, status: str = "completed", error_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """End trace and calculate final metrics."""
        end_time = time.time()
        
        with trace_lock:
            for trace in reversed(trace_buffer):
                if trace["trace_id"] == trace_id:
                    trace["end_time"] = end_time
                    trace["duration_ms"] = round((end_time - trace["start_time"]) * 1000, 2)
                    trace["status"] = status
                    trace["error_info"] = error_info
                    
                    print(f"[TRACE] Completed: {trace['operation']} ({trace['duration_ms']}ms)")
                    return trace.copy()
        
        print(f"[TRACE] Trace not found: {trace_id}")
        return {}
    
    def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific trace by ID."""
        with trace_lock:
            for trace in reversed(trace_buffer):
                if trace["trace_id"] == trace_id:
                    return trace.copy()
        return None
    
    def get_session_traces_buffer(session_id: str) -> List[Dict[str, Any]]:
        """Get all traces for a session from buffer."""
        with trace_lock:
            session_traces = [
                trace.copy() for trace in trace_buffer 
                if trace["session_id"] == session_id
            ]
        
        print(f"[TRACE] Retrieved {len(session_traces)} traces for session: {session_id}")
        return session_traces
    
    def get_performance_summary(session_id: str) -> Dict[str, Any]:
        """Generate performance summary from traces."""
        session_traces = get_session_traces_buffer(session_id)
        
        if not session_traces:
            return {"status": "no_traces"}
        
        total_operations = len(session_traces)
        completed_traces = [t for t in session_traces if t["status"] == "completed"]
        
        avg_duration = (
            sum(t["duration_ms"] for t in completed_traces) / len(completed_traces)
            if completed_traces else 0
        )
        
        operation_counts = {}
        for trace in session_traces:
            op = trace["operation"]
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        summary = {
            "session_id": session_id,
            "total_operations": total_operations,
            "completed_operations": len(completed_traces),
            "average_duration_ms": round(avg_duration, 2),
            "operation_breakdown": operation_counts,
            "total_decision_points": sum(len(t["decision_points"]) for t in session_traces),
            "error_count": len([t for t in session_traces if t["status"] == "error"])
        }
        
        print(f"[TRACE] Performance summary: {total_operations} operations, {round(avg_duration, 1)}ms avg")
        return summary
    
    def clear_traces(session_id: str = None) -> int:
        """Clear traces for session or all traces."""
        with trace_lock:
            if session_id:
                initial_count = len(trace_buffer)
                filtered_traces = deque(
                    (trace for trace in trace_buffer if trace["session_id"] != session_id),
                    maxlen=1000
                )
                trace_buffer.clear()
                trace_buffer.extend(filtered_traces)
                cleared_count = initial_count - len(trace_buffer)
            else:
                cleared_count = len(trace_buffer)
                trace_buffer.clear()
        
        print(f"[TRACE] Cleared {cleared_count} traces")
        return cleared_count
    
    def calculate_engagement_score(interaction_data: List[Dict[str, Any]]) -> float:
        """Calculate engagement score based on interaction patterns."""
        if not interaction_data:
            return 0.0
        
        response_times = [i.get("response_time", 30) for i in interaction_data]
        question_count = len(interaction_data)
        
        avg_response_time = statistics.mean(response_times)
        time_score = max(0, min(100, 100 - (avg_response_time - 15) * 2))
        
        interaction_score = min(100, question_count * 10)
        
        follow_ups = sum(1 for i in interaction_data if i.get("follow_up_questions", 0) > 0)
        curiosity_score = (follow_ups / question_count) * 100 if question_count > 0 else 0
        
        engagement = (time_score * 0.4 + interaction_score * 0.3 + curiosity_score * 0.3)
        
        print(f"[ANALYTICS] Engagement score calculated: {engagement:.1f}")
        return round(engagement, 1)
    
    def assess_conversation_quality(state: ConversationState) -> Dict[str, float]:
        """Assess overall conversation quality metrics."""
        if not state.interview_history:
            return {"overall_quality": 0.0}
        
        total_interactions = len(state.interview_history)
        successful_interactions = sum(
            1 for i in state.interview_history 
            if i.get("status") == "completed"
        )
        
        completion_rate = (successful_interactions / total_interactions) * 100
        
        response_quality_scores = [
            i.get("quality_score", 50) for i in state.interview_history 
            if "quality_score" in i
        ]
        avg_response_quality = statistics.mean(response_quality_scores) if response_quality_scores else 50
        
        phase_transitions = len(state.context_data.get("transitions", []))
        flow_score = min(100, phase_transitions * 25)
        
        overall_quality = (completion_rate * 0.4 + avg_response_quality * 0.4 + flow_score * 0.2)
        
        quality_metrics = {
            "overall_quality": round(overall_quality, 1),
            "completion_rate": round(completion_rate, 1),
            "response_quality": round(avg_response_quality, 1),
            "flow_score": round(flow_score, 1)
        }
        
        print(f"[ANALYTICS] Quality assessment: {overall_quality:.1f} overall")
        return quality_metrics
    
    def analyze_effectiveness(state: ConversationState) -> Dict[str, Any]:
        """Analyze conversation effectiveness for learning outcomes."""
        memory = state.context_data.get("memory", {})
        metrics = memory.get("performance_metrics", {})
        
        total_questions = metrics.get("total_questions", 0)
        correct_answers = metrics.get("correct_answers", 0)
        
        if total_questions == 0:
            return {"effectiveness_score": 0.0, "learning_progress": "insufficient_data"}
        
        accuracy_rate = (correct_answers / total_questions) * 100
        
        difficulty_progression = memory.get("difficulty_progression", [])
        improvement_trend = 0.0
        
        if len(difficulty_progression) >= 3:
            recent_performance = difficulty_progression[-3:]
            recent_success_rate = sum(1 for p in recent_performance if p.get("success", False)) / 3
            
            early_performance = difficulty_progression[:3]
            early_success_rate = sum(1 for p in early_performance if p.get("success", False)) / 3
            
            improvement_trend = (recent_success_rate - early_success_rate) * 100
        
        topic_coverage = len(memory.get("topic_coverage", {}))
        coverage_score = min(100, topic_coverage * 20)
        
        effectiveness_score = (accuracy_rate * 0.5 + coverage_score * 0.3 + max(0, improvement_trend) * 0.2)
        
        learning_progress = "excellent" if effectiveness_score > 80 else \
                           "good" if effectiveness_score > 60 else \
                           "needs_improvement" if effectiveness_score > 40 else "poor"
        
        effectiveness_analysis = {
            "effectiveness_score": round(effectiveness_score, 1),
            "accuracy_rate": round(accuracy_rate, 1),
            "improvement_trend": round(improvement_trend, 1),
            "topics_covered": topic_coverage,
            "learning_progress": learning_progress
        }
        
        print(f"[ANALYTICS] Effectiveness analysis: {effectiveness_score:.1f} score")
        return effectiveness_analysis
    
    def generate_real_time_insights(state: ConversationState) -> List[str]:
        """Generate actionable insights from current conversation state."""
        insights = []
        
        quality_metrics = assess_conversation_quality(state)
        effectiveness = analyze_effectiveness(state)
        engagement = calculate_engagement_score(state.interview_history)
        
        if quality_metrics["completion_rate"] < 70:
            insights.append("Consider simplifying questions to improve completion rate")
        
        if engagement < 50:
            insights.append("Candidate engagement is low - try more interactive questions")
        
        if effectiveness["accuracy_rate"] < 40:
            insights.append("Reduce question difficulty to build confidence")
        elif effectiveness["accuracy_rate"] > 90:
            insights.append("Consider increasing question difficulty for better assessment")
        
        if effectiveness["topics_covered"] < 3:
            insights.append("Expand topic coverage to better assess candidate skills")
        
        if len(state.interview_history) > 10 and state.current_phase == "questioning":
            insights.append("Consider transitioning to evaluation phase")
        
        print(f"[ANALYTICS] Generated {len(insights)} real-time insights")
        return insights
    
    def get_analytics_dashboard(state: ConversationState) -> Dict[str, Any]:
        """Generate comprehensive analytics dashboard."""
        engagement_score = calculate_engagement_score(state.interview_history)
        quality_metrics = assess_conversation_quality(state)
        effectiveness = analyze_effectiveness(state)
        insights = generate_real_time_insights(state)
        
        dashboard = {
            "session_id": state.session_id,
            "current_phase": state.current_phase,
            "total_interactions": len(state.interview_history),
            "engagement_score": engagement_score,
            "quality_metrics": quality_metrics,
            "effectiveness": effectiveness,
            "real_time_insights": insights,
            "recommendation": "continue" if engagement_score > 60 else "adjust_strategy",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"[ANALYTICS] Dashboard generated: {len(state.interview_history)} interactions analyzed")
        return dashboard
    
    def track_conversation_trend(session_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track conversation trends over time."""
        if not session_traces:
            return {"trend": "no_data"}
        
        operation_durations = {}
        for trace in session_traces:
            op = trace["operation"]
            duration = trace.get("duration_ms", 0)
            if op not in operation_durations:
                operation_durations[op] = []
            operation_durations[op].append(duration)
        
        trends = {}
        for operation, durations in operation_durations.items():
            if len(durations) >= 3:
                recent_avg = statistics.mean(durations[-3:])
                early_avg = statistics.mean(durations[:3])
                trend_direction = "improving" if recent_avg < early_avg else "declining"
                trends[operation] = {
                    "direction": trend_direction,
                    "change_percent": round(((recent_avg - early_avg) / early_avg) * 100, 1)
                }
        
        print(f"[ANALYTICS] Tracked trends for {len(trends)} operations")
        return {"trends": trends, "operations_analyzed": len(operation_durations)}
    
    def create_conversation_state(candidate_profile: Dict[str, Any]) -> ConversationState:
        """Create new conversation state for a candidate."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        state = ConversationState(
            session_id=session_id,
            current_phase="greeting",
            candidate_profile=candidate_profile,
            interview_history=[],
            context_data={},
            created_at=timestamp,
            updated_at=timestamp
        )
        
        print(f"[CONV] Created new conversation state: {session_id}")
        return state
    
    def save_conversation_state(state: ConversationState) -> None:
        """Persist conversation state to storage."""
        state.updated_at = datetime.now().isoformat()
        file_path = storage_dir / f"{state.session_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(asdict(state), f, indent=2)
        
        print(f"[CONV] Saved conversation state: {state.session_id}")
    
    def load_conversation_state(session_id: str) -> Optional[ConversationState]:
        """Load conversation state from storage."""
        file_path = storage_dir / f"{session_id}.json"
        
        if not file_path.exists():
            print(f"[CONV] Conversation state not found: {session_id}")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"[CONV] Loaded conversation state: {session_id}")
        return ConversationState(**data)
    
    def update_conversation_context(state: ConversationState, key: str, value: Any) -> None:
        """Update conversation context data."""
        state.context_data[key] = value
        state.updated_at = datetime.now().isoformat()
        print(f"[CONV] Updated context: {key}")
    
    def initialize_memory(candidate_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize memory context for new conversation."""
        memory_context = {
            "candidate_profile": candidate_profile,
            "conversation_summary": "",
            "key_insights": [],
            "response_patterns": {},
            "performance_metrics": {
                "total_questions": 0,
                "correct_answers": 0,
                "response_time_avg": 0.0,
                "engagement_score": 0.0
            },
            "topic_coverage": {},
            "difficulty_progression": [],
            "interviewer_notes": []
        }
        print(f"[CONV] Memory initialized for candidate: {candidate_profile.get('name', 'Unknown')}")
        return memory_context
    
    def update_memory(state: ConversationState, interaction: Dict[str, Any]) -> None:
        """Update memory with new interaction data."""
        if "memory" not in state.context_data:
            state.context_data["memory"] = initialize_memory(state.candidate_profile)
        
        memory = state.context_data["memory"]
        
        if interaction.get("type") == "question_response":
            memory["performance_metrics"]["total_questions"] += 1
            
            if interaction.get("correct", False):
                memory["performance_metrics"]["correct_answers"] += 1
            
            response_time = interaction.get("response_time", 0)
            current_avg = memory["performance_metrics"]["response_time_avg"]
            total_questions = memory["performance_metrics"]["total_questions"]
            memory["performance_metrics"]["response_time_avg"] = (
                (current_avg * (total_questions - 1) + response_time) / total_questions
            )
            
            topic = interaction.get("topic", "general")
            memory["topic_coverage"][topic] = memory["topic_coverage"].get(topic, 0) + 1
            
            difficulty = interaction.get("difficulty", "medium")
            memory["difficulty_progression"].append({
                "question_num": total_questions,
                "difficulty": difficulty,
                "success": interaction.get("correct", False)
            })
        
        state.updated_at = datetime.now().isoformat()
        print(f"[CONV] Memory updated with {interaction.get('type', 'interaction')}")
    
    def get_memory_summary(state: ConversationState) -> Dict[str, Any]:
        """Generate summary of conversation memory."""
        if "memory" not in state.context_data:
            return {"status": "no_memory", "summary": "No memory data available"}
        
        memory = state.context_data["memory"]
        metrics = memory["performance_metrics"]
        
        accuracy = (metrics["correct_answers"] / metrics["total_questions"] 
                   if metrics["total_questions"] > 0 else 0)
        
        summary = {
            "candidate_name": state.candidate_profile.get("name", "Unknown"),
            "total_interactions": len(state.interview_history),
            "accuracy_rate": round(accuracy * 100, 1),
            "avg_response_time": round(metrics["response_time_avg"], 2),
            "topics_covered": len(memory["topic_coverage"]),
            "current_phase": state.current_phase,
            "engagement_level": memory["performance_metrics"]["engagement_score"]
        }
        
        print(f"[CONV] Generated memory summary: {summary['total_interactions']} interactions")
        return summary
    
    def add_interviewer_note(state: ConversationState, note: str, category: str = "general") -> None:
        """Add interviewer observation to memory."""
        if "memory" not in state.context_data:
            state.context_data["memory"] = initialize_memory(state.candidate_profile)
        
        note_entry = {
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "category": category,
            "phase": state.current_phase
        }
        
        state.context_data["memory"]["interviewer_notes"].append(note_entry)
        print(f"[CONV] Added interviewer note: {category}")
    
    def extract_insights(state: ConversationState) -> List[str]:
        """Extract key insights from conversation memory."""
        if "memory" not in state.context_data:
            return []
        
        memory = state.context_data["memory"]
        insights = []
        
        metrics = memory["performance_metrics"]
        if metrics["total_questions"] > 0:
            accuracy = metrics["correct_answers"] / metrics["total_questions"]
            if accuracy > 0.8:
                insights.append("Strong technical performance")
            elif accuracy < 0.5:
                insights.append("Needs improvement in technical areas")
        
        if memory["performance_metrics"]["response_time_avg"] > 60:
            insights.append("Takes time to think through problems")
        elif memory["performance_metrics"]["response_time_avg"] < 15:
            insights.append("Quick response times, confident answers")
        
        topic_coverage = memory["topic_coverage"]
        if len(topic_coverage) > 3:
            insights.append("Demonstrates broad knowledge across topics")
        
        print(f"[CONV] Extracted {len(insights)} insights from memory")
        return insights
    
    def calculate_performance_score(recent_responses: List[Dict[str, Any]]) -> float:
        """Calculate current performance score from recent responses."""
        if not recent_responses:
            return 50.0
        
        window_size = min(5, len(recent_responses))
        recent_window = recent_responses[-window_size:]
        
        accuracy_scores = []
        response_time_scores = []
        quality_scores = []
        
        for response in recent_window:
            accuracy = 100 if response.get("correct", False) else 0
            accuracy_scores.append(accuracy)
            
            response_time = response.get("response_time", 60)
            time_score = max(0, 100 - (response_time - 30) * 2)
            response_time_scores.append(time_score)
            
            quality = response.get("quality_score", 50)
            quality_scores.append(quality)
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_time_score = sum(response_time_scores) / len(response_time_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        performance_score = (avg_accuracy * 0.5 + avg_quality * 0.3 + avg_time_score * 0.2)
        
        print(f"[DIFFICULTY] Performance score calculated: {performance_score:.1f}")
        return performance_score
    
    def determine_optimal_difficulty(performance_score: float, current_difficulty: str, difficulty_levels: Dict[str, Any]) -> str:
        """Determine optimal difficulty level based on performance."""
        current_level = list(difficulty_levels.keys()).index(current_difficulty) if current_difficulty in difficulty_levels else 1
        
        target_level = current_level
        
        if performance_score >= 85:
            target_level = min(len(difficulty_levels) - 1, current_level + 1)
        elif performance_score >= 70:
            target_level = current_level
        elif performance_score >= 50:
            target_level = max(0, current_level)
        else:
            target_level = max(0, current_level - 1)
        
        optimal_difficulty = list(difficulty_levels.keys())[target_level]
        
        print(f"[DIFFICULTY] Optimal difficulty determined: {optimal_difficulty} (score: {performance_score:.1f})")
        return optimal_difficulty
    
    def adapt_question_parameters(difficulty: str, difficulty_levels: Dict[str, Any], topic_complexity: float = 1.0) -> Dict[str, Any]:
        """Adapt question parameters based on difficulty level."""
        base_config = difficulty_levels[difficulty]
        
        adapted_params = {
            "difficulty_level": difficulty,
            "complexity_multiplier": base_config["complexity"] * topic_complexity,
            "expected_response_time": base_config["time_limit"],
            "hint_availability": difficulty in ["beginner", "intermediate"],
            "follow_up_depth": base_config["complexity"],
            "technical_depth": min(5, base_config["complexity"] + 1)
        }
        
        if difficulty == "beginner":
            adapted_params.update({
                "provide_examples": True,
                "allow_multiple_attempts": True,
                "guided_prompts": True
            })
        elif difficulty == "expert":
            adapted_params.update({
                "require_trade_offs": True,
                "expect_edge_cases": True,
                "system_constraints": True
            })
        
        print(f"[DIFFICULTY] Parameters adapted for {difficulty} level")
        return adapted_params
    
    def get_adaptation_feedback(
        previous_difficulty: str, 
        new_difficulty: str, 
        performance_score: float,
        difficulty_levels: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate feedback about difficulty adaptation."""
        difficulty_change = 0
        current_idx = list(difficulty_levels.keys()).index(new_difficulty)
        previous_idx = list(difficulty_levels.keys()).index(previous_difficulty) if previous_difficulty in difficulty_levels else current_idx
        
        difficulty_change = current_idx - previous_idx
        
        feedback = {
            "difficulty_change": difficulty_change,
            "new_level": new_difficulty,
            "previous_level": previous_difficulty,
            "performance_score": performance_score,
            "adaptation_reason": "",
            "candidate_message": ""
        }
        
        if difficulty_change > 0:
            feedback["adaptation_reason"] = "Strong performance - increasing challenge"
            feedback["candidate_message"] = "Great job! Let's try some more challenging questions."
        elif difficulty_change < 0:
            feedback["adaptation_reason"] = "Adjusting to better match current skill level"
            feedback["candidate_message"] = "Let's focus on building confidence with these questions."
        else:
            feedback["adaptation_reason"] = "Maintaining current difficulty level"
            feedback["candidate_message"] = "You're progressing well at this level."
        
        print(f"[DIFFICULTY] Adaptation feedback: {feedback['adaptation_reason']}")
        return feedback
    
    def validate_difficulty_transition(
        current_difficulty: str, 
        proposed_difficulty: str, 
        session_length: int,
        difficulty_levels: Dict[str, Any]
    ) -> bool:
        """Validate if difficulty transition is appropriate."""
        if session_length < 3:
            return proposed_difficulty in ["beginner", "intermediate"]
        
        current_idx = list(difficulty_levels.keys()).index(current_difficulty) if current_difficulty in difficulty_levels else 1
        proposed_idx = list(difficulty_levels.keys()).index(proposed_difficulty)
        
        max_jump = 1 if session_length < 10 else 2
        transition_valid = abs(proposed_idx - current_idx) <= max_jump
        
        print(f"[DIFFICULTY] Transition validation: {current_difficulty} -> {proposed_difficulty}: {transition_valid}")
        return transition_valid
    
    def analyze_response_quality(response: str, expected_keywords: List[str] = None) -> float:
        """Analyze response quality to determine next questioning strategy."""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        response_lower = response.lower()
        word_count = len(response.split())
        
        completeness_score = min(1.0, word_count / 50)
        
        keyword_score = 0.0
        if expected_keywords:
            found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
            keyword_score = found_keywords / len(expected_keywords)
        
        depth_indicators = ["because", "therefore", "however", "although", "furthermore", "moreover"]
        depth_score = min(1.0, sum(1 for indicator in depth_indicators if indicator in response_lower) / 3)
        
        quality_score = (completeness_score * 0.4 + keyword_score * 0.4 + depth_score * 0.2)
        
        print(f"[QUESTION] Response quality analyzed: {quality_score:.2f}")
        return quality_score
    
    def select_next_strategy(current_performance: Dict[str, Any], conversation_context: Dict[str, Any], question_strategies: Dict[str, Any]) -> str:
        """Select optimal questioning strategy based on performance and context."""
        accuracy_rate = current_performance.get("accuracy_rate", 50) / 100
        response_quality = current_performance.get("avg_response_quality", 50) / 100
        engagement_score = current_performance.get("engagement_score", 50) / 100
        
        strategy_scores = {}
        
        for strategy, config in question_strategies.items():
            base_score = config["weight"]
            
            if strategy == "clarification" and response_quality < config["trigger_threshold"]:
                base_score *= 2.0
            elif strategy == "deep_dive" and accuracy_rate > config["trigger_threshold"]:
                base_score *= 1.5
            elif strategy == "lateral_thinking" and engagement_score < config["trigger_threshold"]:
                base_score *= 1.8
            elif strategy == "scenario_based" and accuracy_rate > config["trigger_threshold"]:
                base_score *= 1.3
            elif strategy == "follow_up" and response_quality > config["trigger_threshold"]:
                base_score *= 1.4
            
            strategy_scores[strategy] = base_score
        
        selected_strategy = max(strategy_scores, key=strategy_scores.get)
        print(f"[QUESTION] Selected strategy: {selected_strategy} (score: {strategy_scores[selected_strategy]:.2f})")
        return selected_strategy
    
    def generate_adaptive_question(
        topic: str, 
        difficulty: str, 
        strategy: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate question based on adaptive strategy."""
        question_templates = {
            "clarification": [
                f"Can you clarify what you meant by your approach to {topic}?",
                f"Could you elaborate on the {topic} concept you mentioned?",
                f"Can you provide more details about your {topic} experience?"
            ],
            "deep_dive": [
                f"What are the potential challenges with {topic} in a production environment?",
                f"How would you optimize {topic} for scalability?",
                f"What are the trade-offs to consider when implementing {topic}?"
            ],
            "lateral_thinking": [
                f"If you couldn't use {topic}, what alternative approach would you take?",
                f"How would you explain {topic} to a non-technical stakeholder?",
                f"What creative solutions have you seen for {topic} challenges?"
            ],
            "scenario_based": [
                f"Imagine you're debugging a {topic} issue in production. Walk me through your process.",
                f"You have 2 hours to implement {topic} for a critical deadline. What's your approach?",
                f"A junior developer asks you about {topic}. How do you mentor them?"
            ],
            "follow_up": [
                f"Building on your {topic} answer, what would you do next?",
                f"That's interesting about {topic}. Can you give me a specific example?",
                f"How does your {topic} approach compare to industry best practices?"
            ]
        }
        
        available_questions = question_templates.get(strategy, question_templates["clarification"])
        selected_question = random.choice(available_questions)
        
        question_data = {
            "question": selected_question,
            "topic": topic,
            "difficulty": difficulty,
            "strategy": strategy,
            "expected_response_time": 60 if strategy == "deep_dive" else 30,
            "evaluation_criteria": {
                "completeness": 0.4,
                "accuracy": 0.3,
                "depth": 0.3
            }
        }
        
        print(f"[QUESTION] Generated {strategy} question for {topic}")
        return question_data
    
    def adapt_strategy_weights(feedback_data: List[Dict[str, Any]], question_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy weights based on success feedback."""
        strategy_performance = {}
        
        for feedback in feedback_data:
            strategy = feedback.get("strategy", "unknown")
            success_score = feedback.get("success_score", 0.5)
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(success_score)
        
        for strategy, scores in strategy_performance.items():
            if strategy in question_strategies:
                avg_performance = sum(scores) / len(scores)
                
                if avg_performance > 0.7:
                    question_strategies[strategy]["weight"] *= 1.1
                elif avg_performance < 0.4:
                    question_strategies[strategy]["weight"] *= 0.9
                
                question_strategies[strategy]["weight"] = max(0.1, min(2.0, question_strategies[strategy]["weight"]))
        
        print(f"[QUESTION] Adapted weights for {len(strategy_performance)} strategies")
        return question_strategies
    
    def assess_topic_expertise(topic: str, responses: List[Dict[str, Any]]) -> float:
        """Assess candidate expertise in specific topic based on responses."""
        topic_responses = [r for r in responses if r.get("topic") == topic]
        
        if not topic_responses:
            return 0.0
        
        accuracy_scores = [100 if r.get("correct", False) else 0 for r in topic_responses]
        quality_scores = [r.get("quality_score", 50) for r in topic_responses]
        depth_scores = [r.get("depth_score", 50) for r in topic_responses]
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_depth = sum(depth_scores) / len(depth_scores)
        
        expertise_score = (avg_accuracy * 0.4 + avg_quality * 0.3 + avg_depth * 0.3)
        
        print(f"[TOPIC] Expertise assessed for {topic}: {expertise_score:.1f}")
        return expertise_score
    
    def identify_knowledge_gaps(
        candidate_profile: Dict[str, Any], 
        assessed_topics: Dict[str, float]
    ) -> List[str]:
        """Identify knowledge gaps based on role requirements and current performance."""
        target_role = candidate_profile.get("target_role", "software_engineer")
        experience_level = candidate_profile.get("experience_level", "mid")
        
        role_requirements = {
            "software_engineer": ["algorithms", "data_structures", "system_design", "problem_solving"],
            "data_scientist": ["machine_learning", "algorithms", "databases", "problem_solving"],
            "devops_engineer": ["system_design", "networking", "security", "devops"],
            "frontend_developer": ["web_development", "algorithms", "problem_solving"],
            "backend_developer": ["system_design", "databases", "algorithms", "security"]
        }
        
        expected_thresholds = {
            "junior": 60.0,
            "mid": 75.0,
            "senior": 85.0,
            "lead": 90.0
        }
        
        required_topics = role_requirements.get(target_role, role_requirements["software_engineer"])
        threshold = expected_thresholds.get(experience_level, 75.0)
        
        knowledge_gaps = []
        for topic in required_topics:
            current_score = assessed_topics.get(topic, 0.0)
            if current_score < threshold:
                knowledge_gaps.append(topic)
        
        print(f"[TOPIC] Identified {len(knowledge_gaps)} knowledge gaps for {target_role}")
        return knowledge_gaps
    
    def calculate_topic_priority(
        topic: str, 
        current_expertise: float, 
        interview_objectives: Dict[str, Any],
        topic_taxonomy: Dict[str, Any]
    ) -> float:
        """Calculate priority score for topic exploration."""
        topic_category = None
        topic_config = None
        
        for category, topics in topic_taxonomy.items():
            if topic in topics:
                topic_category = category
                topic_config = topics[topic]
                break
        
        if not topic_config:
            return 0.0
        
        base_priority = topic_config["weight"]
        
        expertise_factor = max(0.1, 1.0 - (current_expertise / 100))
        
        objective_weight = interview_objectives.get("topic_weights", {}).get(topic, 1.0)
        
        time_remaining = interview_objectives.get("time_remaining", 30)
        time_factor = min(1.0, time_remaining / 45)
        
        coverage_bonus = 0.0
        covered_topics = interview_objectives.get("covered_topics", [])
        if topic not in covered_topics:
            coverage_bonus = 0.3
        
        priority_score = (
            base_priority * expertise_factor * objective_weight * time_factor + coverage_bonus
        )
        
        print(f"[TOPIC] Priority calculated for {topic}: {priority_score:.2f}")
        return priority_score
    
    def validate_topic_prerequisites(topic: str, covered_topics: List[str], topic_taxonomy: Dict[str, Any]) -> bool:
        """Validate if topic prerequisites have been covered."""
        topic_config = None
        
        for category_topics in topic_taxonomy.values():
            if topic in category_topics:
                topic_config = category_topics[topic]
                break
        
        if not topic_config:
            return True
        
        prerequisites = topic_config.get("prerequisites", [])
        prerequisites_met = all(prereq in covered_topics for prereq in prerequisites)
        
        print(f"[TOPIC] Prerequisites validation for {topic}: {prerequisites_met}")
        return prerequisites_met
    
    def find_related_topics(topic1: str, topic2: str, topic_taxonomy: Dict[str, Any]) -> List[str]:
        """Find topics that bridge between two topics."""
        related_topics = []
        
        topic1_category = None
        topic2_category = None
        
        for category, topics in topic_taxonomy.items():
            if topic1 in topics:
                topic1_category = category
            if topic2 in topics:
                topic2_category = category
        
        if topic1_category == topic2_category:
            return []
        
        bridging_topics = ["problem_solving", "algorithms", "communication"]
        for bridge_topic in bridging_topics:
            if bridge_topic != topic1 and bridge_topic != topic2:
                related_topics.append(bridge_topic)
        
        return related_topics[:2]
    
    def assess_conversation_flow(interactions: List[Dict[str, Any]]) -> float:
        """Assess smoothness and logic of conversation flow."""
        if len(interactions) < 2:
            return 50.0
        
        transition_scores = []
        
        for i in range(1, len(interactions)):
            prev_interaction = interactions[i-1]
            curr_interaction = interactions[i]
            
            topic_continuity = 100 if prev_interaction.get("topic") == curr_interaction.get("topic") else 70
            
            prev_difficulty = prev_interaction.get("difficulty", "medium")
            curr_difficulty = curr_interaction.get("difficulty", "medium")
            difficulty_levels = ["easy", "medium", "hard", "expert"]
            
            prev_idx = difficulty_levels.index(prev_difficulty) if prev_difficulty in difficulty_levels else 1
            curr_idx = difficulty_levels.index(curr_difficulty) if curr_difficulty in difficulty_levels else 1
            
            difficulty_progression = 100 if abs(curr_idx - prev_idx) <= 1 else 60
            
            response_time_prev = prev_interaction.get("response_time", 30)
            response_time_curr = curr_interaction.get("response_time", 30)
            time_consistency = max(50, 100 - abs(response_time_curr - response_time_prev) * 2)
            
            transition_score = (topic_continuity * 0.4 + difficulty_progression * 0.3 + time_consistency * 0.3)
            transition_scores.append(transition_score)
        
        flow_score = sum(transition_scores) / len(transition_scores)
        
        print(f"[FEEDBACK] Conversation flow assessed: {flow_score:.1f}")
        return flow_score
    
    def evaluate_question_quality(question_data: Dict[str, Any], response_data: Dict[str, Any]) -> float:
        """Evaluate effectiveness of individual questions."""
        question = question_data.get("question", "")
        strategy = question_data.get("strategy", "unknown")
        
        clarity_score = min(100, len(question.split()) * 5) if len(question.split()) <= 20 else 80
        
        response_quality = response_data.get("quality_score", 50)
        engagement_indicators = response_data.get("follow_up_questions", 0)
        engagement_score = min(100, 60 + engagement_indicators * 20)
        
        strategy_effectiveness = {
            "clarification": 85,
            "deep_dive": 90,
            "lateral_thinking": 80,
            "scenario_based": 95,
            "follow_up": 75
        }
        
        strategy_score = strategy_effectiveness.get(strategy, 70)
        
        overall_quality = (clarity_score * 0.3 + response_quality * 0.4 + engagement_score * 0.2 + strategy_score * 0.1)
        
        print(f"[FEEDBACK] Question quality evaluated: {overall_quality:.1f}")
        return overall_quality
    
    def measure_candidate_engagement(interaction_history: List[Dict[str, Any]]) -> float:
        """Measure candidate engagement throughout conversation."""
        if not interaction_history:
            return 50.0
        
        response_lengths = [len(i.get("response", "").split()) for i in interaction_history]
        avg_response_length = sum(response_lengths) / len(response_lengths)
        length_score = min(100, avg_response_length * 2)
        
        response_times = [i.get("response_time", 60) for i in interaction_history]
        avg_response_time = sum(response_times) / len(response_times)
        time_score = max(0, 100 - (avg_response_time - 30) * 1.5)
        
        follow_up_count = sum(i.get("follow_up_questions", 0) for i in interaction_history)
        curiosity_score = min(100, follow_up_count * 15)
        
        completion_rate = sum(1 for i in interaction_history if i.get("status") == "completed") / len(interaction_history) * 100
        
        engagement_score = (length_score * 0.3 + time_score * 0.25 + curiosity_score * 0.25 + completion_rate * 0.2)
        
        print(f"[FEEDBACK] Candidate engagement measured: {engagement_score:.1f}")
        return engagement_score
    
    def track_learning_progress(memory_data: Dict[str, Any]) -> float:
        """Track candidate learning progress over time."""
        performance_metrics = memory_data.get("performance_metrics", {})
        difficulty_progression = memory_data.get("difficulty_progression", [])
        
        if not difficulty_progression:
            return 50.0
        
        recent_performance = difficulty_progression[-3:] if len(difficulty_progression) >= 3 else difficulty_progression
        early_performance = difficulty_progression[:3] if len(difficulty_progression) >= 6 else difficulty_progression
        
        recent_success_rate = sum(1 for p in recent_performance if p.get("success", False)) / len(recent_performance) * 100
        early_success_rate = sum(1 for p in early_performance if p.get("success", False)) / len(early_performance) * 100
        
        improvement_trend = recent_success_rate - early_success_rate
        trend_score = max(0, 50 + improvement_trend * 2)
        
        accuracy_rate = (performance_metrics.get("correct_answers", 0) / 
                        max(1, performance_metrics.get("total_questions", 1))) * 100
        
        topic_coverage = len(memory_data.get("topic_coverage", {}))
        coverage_score = min(100, topic_coverage * 20)
        
        progress_score = (trend_score * 0.4 + accuracy_rate * 0.4 + coverage_score * 0.2)
        
        print(f"[FEEDBACK] Learning progress tracked: {progress_score:.1f}")
        return progress_score
    
    def calculate_overall_effectiveness(metric_scores: Dict[str, float], feedback_metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall effectiveness score."""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            if metric in feedback_metrics:
                weight = feedback_metrics[metric]["weight"]
                total_score += score * weight
                total_weight += weight
        
        overall_effectiveness = total_score / total_weight if total_weight > 0 else 50.0
        
        print(f"[FEEDBACK] Overall effectiveness calculated: {overall_effectiveness:.1f}")
        return overall_effectiveness
    
    engine_config = {
        "storage_path": str(storage_dir),
        "create_state": create_conversation_state,
        "save_state": save_conversation_state,
        "load_state": load_conversation_state,
        "update_context": update_conversation_context,
        "initialize_memory": initialize_memory,
        "update_memory": update_memory,
        "get_memory_summary": get_memory_summary,
        "add_interviewer_note": add_interviewer_note,
        "extract_insights": extract_insights,
        "start_trace": start_trace,
        "add_decision": add_decision_point,
        "add_metric": add_performance_metric,
        "end_trace": end_trace,
        "get_trace": get_trace,
        "get_session_traces": get_session_traces_buffer,
        "get_performance_summary": get_performance_summary,
        "clear_traces": clear_traces,
        "calculate_engagement": calculate_engagement_score,
        "assess_quality": assess_conversation_quality,
        "analyze_effectiveness": analyze_effectiveness,
        "generate_insights": generate_real_time_insights,
        "get_dashboard": get_analytics_dashboard,
        "track_trends": track_conversation_trend,
        "assess_conversation_flow": assess_conversation_flow,
        "measure_candidate_engagement": measure_candidate_engagement,
        "track_learning_progress": track_learning_progress,
        "calculate_overall_effectiveness": calculate_overall_effectiveness,
        "evaluate_question_quality": evaluate_question_quality,
        "initialize_database": initialize_database,
        "store_trace": store_trace,
        "retrieve_trace": retrieve_trace,
        "get_session_traces_db": get_session_traces,
        "query_by_operation": query_traces_by_operation,
        "analyze_patterns": analyze_performance_patterns,
        "cleanup_old_traces": cleanup_old_traces,
        "get_storage_stats": get_storage_stats,
        "analyze_response": analyze_response_quality,
        "select_strategy": select_next_strategy,
        "generate_question": generate_adaptive_question,
        "adapt_weights": adapt_strategy_weights,
        "calculate_performance": calculate_performance_score,
        "determine_difficulty": determine_optimal_difficulty,
        "adapt_parameters": adapt_question_parameters,
        "get_feedback": get_adaptation_feedback,
        "validate_transition": validate_difficulty_transition,
        "assess_topic_expertise": assess_topic_expertise,
        "identify_knowledge_gaps": identify_knowledge_gaps,
        "calculate_topic_priority": calculate_topic_priority,
        "validate_topic_prerequisites": validate_topic_prerequisites,
        "find_related_topics": find_related_topics,
        "buffer_size": len(trace_buffer),
        "total_conversations": len(list(storage_dir.glob("*.json")))
    }
    
    print(f"[CONV] Conversation engine initialized successfully")
    print(f"[CONV] Existing conversations: {engine_config['total_conversations']}")
    
    return engine_config