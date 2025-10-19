"""
LLM integration for structured output generation.

This module provides functions for generating structured outputs
using LLM function calling in the interview practice application.
"""

import json
from typing import Dict, Any
from openai import OpenAI
from jsonschema import validate, ValidationError


def implement_structured_generation(
    client: OpenAI,
    prompt: str,
    schema: Dict[str, Any],
    model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
    """Generate questions using OpenAI function calling.
    
    Args:
        client: OpenAI client instance
        prompt: Generation prompt
        schema: JSON schema for output validation
        model: Model to use for generation
        
    Returns:
        Dict[str, Any]: Structured output with metadata
    """
    print(f"[JSON] Starting structured generation with {model}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            functions=[{
                "name": "generate_structured_output",
                "description": "Generate structured interview content",
                "parameters": schema
            }],
            function_call={"name": "generate_structured_output"},
            temperature=0.7
        )
        
        function_call = response.choices[0].message.function_call
        structured_data = json.loads(function_call.arguments)
        
        result = {
            "success": True,
            "data": structured_data,
            "metadata": {
                "model": model,
                "tokens_used": response.usage.total_tokens,
                "function_called": function_call.name
            }
        }
        
        print(f"[SUCCESS] Generated {len(structured_data)} fields, {response.usage.total_tokens} tokens")
        return result
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {str(e)}")
        return {"success": False, "error": "JSON decode error: {}".format(str(e))}
    
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        return {"success": False, "error": "Generation error: {}".format(str(e))}


def generate_structured_question(
    client,
    prompt: str,
    question_schema: Dict[str, Any],
    question_type: str = "technical",
    difficulty: str = "mid"
    ) -> Dict[str, Any]:
    """Generate structured question using OpenAI function calling."""
    
    print(f"[JSON] Generating structured question with {question_type} type and {difficulty} difficulty")
    
    function_schema = {
        "name": "generate_interview_question",
        "description": "Generate a structured interview question",
        "parameters": question_schema
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Generate a {difficulty} level {question_type} interview question."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_schema],
            function_call={"name": "generate_interview_question"}
        )
        
        question_data = json.loads(response.choices[0].message.function_call.arguments)
        
        print(f"[SUCCESS] Generated question with {len(question_data)} fields, {response.usage.total_tokens} tokens")
        return {
            "success": True,
            "question": question_data,
            "metadata": {
                "tokens_used": response.usage.total_tokens,
                "model": "gpt-4o-mini"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def process_candidate_response(
    client,
    question: str,
    candidate_response: str,
    response_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Process and evaluate candidate response."""
    
    print(f"[JSON] Processing candidate response for question: {question[:50]}...")
    
    evaluation_prompt = f"""Question: {question}
    Candidate Response: {candidate_response}

    Provide structured feedback and evaluation."""
    
    function_schema = {
        "name": "evaluate_response",
        "description": "Evaluate candidate response with structured feedback",
        "parameters": response_schema
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            functions=[function_schema],
            function_call={"name": "evaluate_response"}
        )
        
        evaluation_data = json.loads(response.choices[0].message.function_call.arguments)
        
        print(f"[SUCCESS] Processed response with {len(evaluation_data)} fields, {response.usage.total_tokens} tokens")
        return {
            "success": True,
            "evaluation": evaluation_data,
            "metadata": {
                "tokens_used": response.usage.total_tokens,
                "model": "gpt-4o-mini"
            }
        }
    except Exception as e:
        print(f"[ERROR] Response processing failed: {str(e)}")
        return {"success": False, "error": str(e)}


def validate_json_structure(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate JSON data against schemas."""
    print(f"[JSON] Validating JSON structure")
    try:
        validate(instance=data, schema=schema)
        print(f"[SUCCESS] JSON structure validation passed")
        return {"valid": True}
    except ValidationError as e:
        print(f"[ERROR] JSON structure validation failed: {str(e)}")
        return {"valid": False, "error": str(e)}