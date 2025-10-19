"""
JSON schemas for interview questions and responses.

This module defines the JSON schemas used for structured outputs
in the interview practice application.
"""

from typing import Dict, Any, List


def create_question_schema(
    interview_types: List[str] = None,
    difficulty_levels: List[str] = None
) -> Dict[str, Any]:
    """Define JSON schema for interview questions with metadata.
    
    Args:
        interview_types: Supported interview types
        difficulty_levels: Question difficulty levels
        
    Returns:
        Dict[str, Any]: Complete JSON schema for question generation
    """
    print(f"[JSON] Creating question schema")
    
    if interview_types is None:
        interview_types = ["technical", "behavioral", "system_design", "coding"]
    
    if difficulty_levels is None:
        difficulty_levels = ["junior", "mid", "senior", "lead"]
    
    schema = {
        "type": "object",
        "required": ["question", "type", "difficulty", "duration_minutes"],
        "properties": {
            "question": {
                "type": "string",
                "description": "The interview question text",
                "minLength": 10,
                "maxLength": 500
            },
            "type": {
                "type": "string",
                "enum": interview_types,
                "description": "Interview question category"
            },
            "difficulty": {
                "type": "string", 
                "enum": difficulty_levels,
                "description": "Question difficulty level"
            },
            "duration_minutes": {
                "type": "integer",
                "minimum": 2,
                "maximum": 30,
                "description": "Expected answer duration"
            },
            "follow_ups": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Follow-up questions"
            },
            "evaluation_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key evaluation criteria"
            }
        }
    }
    
    print("Question Schema Structure:")
    print("┌─ Required Fields:")
    for field in schema["required"]:
        field_type = schema["properties"][field]["type"]
        print(f"│  ✓ {field} ({field_type})")
    
    print("├─ Optional Fields:")
    optional_fields = [f for f in schema["properties"] if f not in schema["required"]]
    for field in optional_fields:
        field_type = schema["properties"][field]["type"]
        print(f"│  • {field} ({field_type})")
    
    print("├─ Interview Types:")
    for i, itype in enumerate(interview_types, 1):
        print(f"│  {i}. {itype}")
    
    print("└─ Difficulty Levels:")
    for i, level in enumerate(difficulty_levels, 1):
        print(f"   {i}. {level}")
    
    print(f"[SCHEMA] Question schema v1.0 created successfully")
    return {
        "schema": schema,
        "types": interview_types,
        "levels": difficulty_levels,
        "version": "1.0"
    }


def create_response_schema(
    rating_scales: List[str] = None,
    feedback_categories: List[str] = None
) -> Dict[str, Any]:
    """Design schema for candidate responses and feedback.
    
    Args:
        rating_scales: Available rating scales
        feedback_categories: Feedback categorization options
        
    Returns:
        Dict[str, Any]: Complete JSON schema for response evaluation
    """
    print(f"[JSON] Creating response schema")
    
    if rating_scales is None:
        rating_scales = ["excellent", "good", "fair", "poor"]
    
    if feedback_categories is None:
        feedback_categories = ["technical_accuracy", "communication", "problem_solving", "depth"]
    
    schema = {
        "type": "object",
        "required": ["response_text", "overall_rating", "feedback"],
        "properties": {
            "response_text": {
                "type": "string",
                "description": "Candidate's response text",
                "minLength": 5,
                "maxLength": 2000
            },
            "overall_rating": {
                "type": "string",
                "enum": rating_scales,
                "description": "Overall response rating"
            },
            "feedback": {
                "type": "object",
                "required": ["summary", "strengths", "improvements"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief evaluation summary",
                        "maxLength": 300
                    },
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Response strengths"
                    },
                    "improvements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Areas for improvement"
                    }
                }
            },
            "category_scores": {
                "type": "object",
                "properties": {cat: {"type": "string", "enum": rating_scales} for cat in feedback_categories},
                "description": "Detailed category scoring"
            },
            "duration_seconds": {
                "type": "integer",
                "minimum": 30,
                "maximum": 1800,
                "description": "Response duration"
            }
        }
    }
    
    # Visual schema monitoring
    print("Response Schema Structure:")
    print("┌─ Required Fields:")
    for field in schema["required"]:
        field_type = schema["properties"][field]["type"]
        print(f"│  ✓ {field} ({field_type})")
    
    print("├─ Feedback Structure:")
    feedback_props = schema["properties"]["feedback"]["properties"]
    for field, props in feedback_props.items():
        print(f"│  • {field} ({props['type']})")
    
    print("├─ Rating Scales:")
    for i, rating in enumerate(rating_scales, 1):
        print(f"│  {i}. {rating}")
    
    print("└─ Evaluation Categories:")
    for i, category in enumerate(feedback_categories, 1):
        print(f"   {i}. {category}")
    
    print(f"[SCHEMA] Response schema v1.0 created successfully")
    return {
        "schema": schema,
        "rating_scales": rating_scales,
        "categories": feedback_categories,
        "version": "1.0"
    }