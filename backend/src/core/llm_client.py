"""
Unified LLM client interface for multiple providers.

This module provides a unified interface for interacting with different
LLM providers including OpenAI, Gemini, Claude, etc.
"""

import openai
from openai import OpenAI
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
import google.generativeai as genai


class LLMProvider(Enum):
    OPENAI = 'openai'
    GOOGLE = 'google'


@dataclass
class LLMResponse:
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    response_time: float
    cost_estimate: float


def get_gemini_api_key(env_filename: str = "../.env") -> str:
    load_dotenv(dotenv_path=Path(env_filename))
    
    if not Path(env_filename).resolve().exists():
        print(f"[WARNING] .env file not found at '{env_filename}'")
        print("[INFO] Trying to load from current or parent directory default")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Environment variable 'GEMINI_API_KEY' (or 'GOOGLE_API_KEY') "
            "not found in .env file. "
            "Make sure .env contains 'GOOGLE_API_KEY=your_key_here'."
        )
    return api_key


def setup_openai_client(env_path: str = ".env") -> OpenAI:
    """
    Initialize OpenAI client with environment variables and error handling.

    Args:
        env_path (str): Path to the environment file. Defaults to ".env"

    Returns:
        openai.OpenAI: Configured OpenAI client instance

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment
        ConnectionError: If API connection test fails
    """
    print("[SETUP] Initializing OpenAI client configuration")

    load_dotenv(env_path)
    print(f"[CONFIG] Loading environment variables from: {env_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("[CONFIG] API key successfully retrieved from environment")

    client = openai.OpenAI(api_key=api_key)
    print("[SETUP] OpenAI client initialized successfully")

    return client


def setup_gemini_client(env_path: str = "../.env") -> genai.GenerativeModel:
    """Initialize Google Gemini client with environment variables.
    
    Args:
        env_path (str): Path to the environment file. Defaults to "../.env"
        
    Returns:
        genai.GenerativeModel: Configured Gemini client instance
        
    Raises:
        ValueError: If GEMINI_API_KEY is not found in environment
    """
    print("[SETUP] Initializing Google Gemini client configuration")
    
    try:
        gemini_api_key = get_gemini_api_key(env_filename=env_path)
        genai.configure(api_key=gemini_api_key)
        print("[SUCCESS] Gemini API key loaded and configured successfully")
        print(f"[INFO] First 5 characters of API key: {gemini_api_key[:5]}*****")
        
        # Return a GenerativeModel instance
        client = genai.GenerativeModel('gemini-2.5-flash')
        print("[SETUP] Google Gemini client initialized successfully")
        return client
        
    except Exception as e:
        print(f"[ERROR] Failed to load or configure API key: {e}")
        raise ValueError(f"Gemini API configuration failed: {str(e)}")


def setup_multi_llm_clients(env_path: str = '../.env') -> Dict[LLMProvider, Any]:
    """Setup clients for multiple LLM providers.
    
    Args:
        env_path (str): Path to the environment file. Defaults to "../.env"
        
    Returns:
        Dict[LLMProvider, Any]: Dictionary mapping providers to their clients
    """
    print(f'[INIT] Loading environment variables from: {env_path}')
    
    clients = {}
    
    # Setup OpenAI client
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            clients[LLMProvider.OPENAI] = OpenAI(api_key=openai_key)
            print('[SUCCESS] OpenAI client initialized')
        else:
            print('[WARNING] OpenAI API key not found')
    except Exception as e:
        print(f'[ERROR] OpenAI client setup failed: {e}')
    
    # Setup Google Gemini client
    try:
        gemini_client = setup_gemini_client(env_path)
        clients[LLMProvider.GOOGLE] = gemini_client
        print('[SUCCESS] Google Gemini client initialized')
    except Exception as e:
        print(f'[WARNING] Google Gemini client setup failed: {e}')
    
    print(f'[COMPLETED] Multi-LLM setup finished with {len(clients)} providers')
    return clients


def implement_provider_fallback(
    clients: Dict[LLMProvider, Any],
    primary_provider: LLMProvider = LLMProvider.OPENAI,
    max_retries: int = 2,
    timeout_seconds: float = 10.0
    ) -> Dict[str, Any]:
    
    print(f"[FALLBACK] Initializing fallback system with primary: {primary_provider.value}")
    
    available_providers = list(clients.keys())
    fallback_order = [primary_provider] + [p for p in available_providers if p != primary_provider]
    
    print(f"[CONFIG] Fallback order: {[p.value for p in fallback_order]}")
    
    def make_request_with_fallback(prompt: str, model_params: Dict[str, Any] = None) -> LLMResponse:
        if model_params is None:
            model_params = {"temperature": 0.7, "max_tokens": 500}
        
        print(f"[REQUEST] Starting request with fallback for prompt length: {len(prompt)}")
        
        for i, provider in enumerate(fallback_order):
            if provider not in clients:
                print(f"[SKIP] Provider {provider.value} not available")
                continue
            
            try:
                start_time = time.time()
                print(f"[ATTEMPT] Provider {provider.value} (attempt {i+1})")
                
                if provider == LLMProvider.OPENAI:
                    response = clients[provider].chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=model_params.get("temperature", 0.7),
                        max_tokens=model_params.get("max_tokens", 500),
                        timeout=timeout_seconds
                    )
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    cost = tokens * 0.0001
                    model_name = "gpt-4.1-nano"
                
                elif provider == LLMProvider.GOOGLE:
                    response = clients[provider].generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=model_params.get("temperature", 0.7),
                            max_output_tokens=model_params.get("max_tokens", 500)
                        )
                    )
                    content = response.text
                    tokens = len(content.split()) * 1.3
                    cost = tokens * 0.00005
                    model_name = "gemini-2.5-flash"
                
                response_time = time.time() - start_time
                
                llm_response = LLMResponse(
                    content=content,
                    provider=provider,
                    model=model_name,
                    tokens_used=int(tokens),
                    response_time=response_time,
                    cost_estimate=cost
                )
                
                print(f"[SUCCESS] {provider.value} responded in {response_time:.2f}s")
                return llm_response
                
            except Exception as e:
                print(f"[ERROR] {provider.value} failed: {str(e)[:100]}")
                if i == len(fallback_order) - 1:
                    print("[FAILURE] All providers failed")
                    raise Exception(f"All LLM providers failed. Last error: {str(e)}")
                continue
    
    return {
        "fallback_order": fallback_order,
        "available_providers": available_providers,
        "make_request": make_request_with_fallback,
        "primary_provider": primary_provider,
        "max_retries": max_retries,
        "timeout_seconds": timeout_seconds,
        "clients": clients
    }


def create_judge_prompts() -> Dict[str, str]:
    print("[JUDGE] Creating evaluation prompts for LLM-as-judge system")
    
    prompts = {
        "question_quality": """You are an expert technical interviewer evaluating interview questions.

    Rate the following interview question on a scale of 1-10 based on these criteria:
    1. Clarity and specificity
    2. Appropriate difficulty level
    3. Relevance to technical skills
    4. Potential for meaningful discussion

    Question to evaluate: {question}

    Provide your response in this exact JSON format:
    {{
        "overall_score": <1-10>,
        "clarity_score": <1-10>,
        "difficulty_score": <1-10>,
        "relevance_score": <1-10>,
        "reasoning": "<brief explanation>",
        "improvements": "<suggested improvements or 'none'>"
    }}""",

            "response_assessment": """You are an expert interviewer evaluating a candidate's response to a technical question.

    Original Question: {question}
    Candidate Response: {response}

    Rate the response on a scale of 1-10 based on:
    1. Technical accuracy
    2. Completeness of answer
    3. Communication clarity
    4. Problem-solving approach

    Provide your response in this exact JSON format:
    {{
        "overall_score": <1-10>,
        "technical_accuracy": <1-10>,
        "completeness": <1-10>,
        "clarity": <1-10>,
        "feedback": "<constructive feedback>",
        "strengths": "<key strengths identified>",
        "areas_for_improvement": "<areas to improve>"
    }}""",

            "consensus_evaluation": """You are evaluating two different interview questions to determine which is better.

    Question A: {question_a}
    Question B: {question_b}

    Context: {context}

    Determine which question is more suitable based on:
    1. Alignment with job requirements
    2. Interview effectiveness
    3. Candidate assessment value
    4. Implementation clarity

    Provide your response in this exact JSON format:
    {{
        "winner": "<A or B>",
        "confidence": <1-10>,
        "reasoning": "<detailed comparison>",
        "question_a_score": <1-10>,
        "question_b_score": <1-10>
    }}"""
    }
    
    print(f"[COMPLETED] Created {len(prompts)} judge prompt templates")
    return prompts


def implement_cross_validation(
    fallback_system: Dict[str, Any],
    judge_prompts: Dict[str, str],
    content_to_evaluate: str,
    evaluation_type: str = "question_quality",
    context: Dict[str, str] = None
    ) -> Dict[str, Any]:
    
    print(f"[CROSS-VALIDATION] Starting {evaluation_type} evaluation")
    
    if evaluation_type not in judge_prompts:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    prompt_template = judge_prompts[evaluation_type]
    
    if evaluation_type == "question_quality":
        evaluation_prompt = prompt_template.format(question=content_to_evaluate)
    elif evaluation_type == "response_assessment" and context:
        evaluation_prompt = prompt_template.format(
            question=context.get("question", ""),
            response=content_to_evaluate
        )
    elif evaluation_type == "consensus_evaluation" and context:
        evaluation_prompt = prompt_template.format(
            question_a=context.get("question_a", ""),
            question_b=context.get("question_b", ""),
            context=content_to_evaluate
        )
    else:
        evaluation_prompt = prompt_template.format(content=content_to_evaluate)
    
    try:
        judge_response = fallback_system["make_request"](
            evaluation_prompt,
            {"temperature": 0.3, "max_tokens": 800}
        )
        
        evaluation_result = json.loads(judge_response.content)
        
        result = {
            "evaluation_type": evaluation_type,
            "judge_provider": judge_response.provider.value,
            "judge_model": judge_response.model,
            "evaluation_result": evaluation_result,
            "response_time": judge_response.response_time,
            "cost": judge_response.cost_estimate
        }
        
        print(f"[SUCCESS] Cross-validation completed by {judge_response.provider.value}")
        print(f"[SCORE] Overall score: {evaluation_result.get('overall_score', 'N/A')}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse judge response as JSON: {e}")
        return {
            "evaluation_type": evaluation_type,
            "error": "JSON parsing failed",
            "raw_response": judge_response.content
        }
    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")
        return {"evaluation_type": evaluation_type, "error": str(e)}


def implement_dual_llm_consensus(
    fallback_system: Dict[str, Any],
    judge_prompts: Dict[str, str],
    prompt: str,
    model_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
    
    print(f"[CONSENSUS] Starting dual-LLM consensus generation")
    
    if model_params is None:
        model_params = {"temperature": 0.7, "max_tokens": 500}
    
    responses = {}
    costs = {}
    
    for provider in [LLMProvider.OPENAI, LLMProvider.GOOGLE]:
        if provider in fallback_system["fallback_order"]:
            try:
                print(f"[GENERATE] Getting response from {provider.value}")
                
                if provider == LLMProvider.OPENAI:
                    client = fallback_system.get("clients", {}).get(provider)
                    if not client:
                        # Try to get client directly from fallback_system
                        client = fallback_system.get(provider)
                        if not client:
                            raise Exception("OpenAI client not configured")
                    
                    response = client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=model_params.get("temperature", 0.7),
                        max_tokens=model_params.get("max_tokens", 500)
                    )
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    cost = tokens * 0.0001
                    
                elif provider == LLMProvider.GOOGLE:
                    client = fallback_system.get("clients", {}).get(provider)
                    if not client:
                        # Try to get client directly from fallback_system
                        client = fallback_system.get(provider)
                        if not client:
                            raise Exception("Gemini client not configured")
                    
                    response = client.generate_content(prompt)
                    content = response.text
                    tokens = len(content.split()) * 1.3
                    cost = 0.0
                
                responses[provider.value] = {
                    "content": content,
                    "tokens": int(tokens),
                    "cost": cost
                }
                costs[provider.value] = cost
                
                print(f"[SUCCESS] {provider.value} generated response ({int(tokens)} tokens)")
                
            except Exception as e:
                print(f"[ERROR] {provider.value} failed: {str(e)[:100]}")
                responses[provider.value] = {"error": str(e)}
    
    valid_responses = {k: v for k, v in responses.items() if "error" not in v}
    if len(valid_responses) < 2:
        print("[WARNING] Could not get responses from both providers")
        if valid_responses:
            provider_name = list(valid_responses.keys())[0]
            return {
                "responses": responses,
                "selected_response": valid_responses[provider_name]["content"],
                "selected_provider": provider_name,
                "consensus_confidence": 6,
                "total_cost": sum(costs.values()),
                "fallback_reason": "Only one provider available"
            }
        return {"error": "No valid responses from providers"}
    
    print("[CONSENSUS] Evaluating responses for best selection")
    
    try:
        openai_preview = valid_responses.get("openai", {}).get("content", "")[:200]
        google_preview = valid_responses.get("google", {}).get("content", "")[:200]
        
        simple_judge_prompt = f"""Compare these interview questions and pick the better one.

        Question A: {openai_preview}...
        Question B: {google_preview}...

        Reply with ONLY this JSON:
        {{"winner": "A", "confidence": 8}}"""
        
        judge_response = fallback_system["make_request"](
            simple_judge_prompt,
            {"temperature": 0.1, "max_tokens": 50}
        )
        
        try:
            clean_response = judge_response.content.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1].strip()
            if clean_response.startswith('json'):
                clean_response = clean_response[4:].strip()
            
            evaluation_result = json.loads(clean_response)
            print(f"[JUDGE] Parsed result: {evaluation_result}")
            
        except json.JSONDecodeError:
            print(f"[JUDGE] Raw response: '{judge_response.content}'")
            evaluation_result = {"winner": "A", "confidence": 6}
            print("[JUDGE] Using fallback evaluation")
        
        winner = evaluation_result.get("winner", "A")
        selected_provider = "openai" if winner == "A" else "google"
        confidence = evaluation_result.get("confidence", 5)
        
        result = {
            "responses": responses,
            "selected_response": valid_responses[selected_provider]["content"],
            "selected_provider": selected_provider,
            "consensus_confidence": confidence,
            "total_cost": sum(costs.values()),
            "evaluation_details": evaluation_result,
            "cost_breakdown": costs
        }
        
        print(f"[CONSENSUS] Selected {selected_provider} with confidence {confidence}/10")
        return result
        
    except Exception as e:
        print(f"[ERROR] Consensus evaluation failed: {e}")
        openai_response = valid_responses.get("openai", {})
        if openai_response:
            return {
                "responses": responses,
                "selected_response": openai_response["content"],
                "selected_provider": "openai",
                "consensus_confidence": 7,
                "total_cost": sum(costs.values()),
                "fallback_reason": "Consensus evaluation failed, defaulted to OpenAI"
            }


def create_cost_optimizer(cost_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    
    if cost_thresholds is None:
        cost_thresholds = {
            "openai_per_token": 0.0001,
            "google_per_token": 0.0,
            "max_cost_per_request": 0.10
        }
    
    print(f"[OPTIMIZER] Cost thresholds configured: {cost_thresholds}")
    
    def route_request(prompt: str, complexity: str = "medium") -> str:
        token_estimate = len(prompt.split()) * 2
        
        if complexity == "high" and token_estimate < 8000:
            recommended = "google"
            reason = "Google gemini-2.5-flash free tier handles high complexity well"
        elif token_estimate > 8000:
            recommended = "openai"
            reason = "Large prompts may exceed Google free tier limits"
        else:
            recommended = "google"
            reason = "Google gemini-2.5-flash is free and sufficient for most tasks"
        
        print(f"[ROUTE] {recommended} recommended - {reason}")
        return recommended
    
    return {
        "cost_thresholds": cost_thresholds,
        "route_request": route_request,
        "openai_cost_per_token": cost_thresholds["openai_per_token"],
        "google_cost_per_token": cost_thresholds["google_per_token"]
    }


def export_multi_llm_pipeline() -> Dict[str, Any]:
    print("[EXPORT] Packaging multi-LLM system for production deployment")
    
    production_components = {
        "clients_setup": setup_multi_llm_clients,
        "fallback_system": implement_provider_fallback,
        "judge_prompts": create_judge_prompts,
        "cross_validation": implement_cross_validation,
        "consensus_engine": implement_dual_llm_consensus,
        "cost_optimizer": create_cost_optimizer
    }
    
    configuration = {
        "supported_providers": ["openai", "google"],
        "models": {"openai": "gpt-4.1-nano", "google": "gemini-2.5-flash"},
        "costs": {"openai_per_token": 0.0001, "google_per_token": 0.0},
        "timeouts": {"fallback_timeout": 10.0, "judge_timeout": 30.0}
    }
    
    export_package = {
        "components": production_components,
        "config": configuration,
        "version": "1.0.0",
        "status": "production_ready"
    }
    
    print(f"[SUCCESS] Pipeline exported with {len(production_components)} components")
    print("[READY] System ready for src/ integration")
    
    return export_package


def run_system_validation(fallback_system: Dict[str, Any], judge_prompts: Dict[str, str]):
    print("[VALIDATION] Running final system validation")
    
    test_prompts = [
        "Generate a Python coding interview question",
        "Create a system design question about databases",
        "Design a behavioral question about teamwork"
    ]
    
    results = []
    total_cost = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[TEST {i}] {prompt[:40]}...")
        
        try:
            result = implement_dual_llm_consensus(fallback_system, judge_prompts, prompt)
            
            results.append({
                "test": i,
                "winner": result.get('selected_provider'),
                "confidence": result.get('consensus_confidence'),
                "cost": result.get('total_cost', 0)
            })
            
            total_cost += result.get('total_cost', 0)
            print(f"[RESULT] {result.get('selected_provider')} selected (confidence: {result.get('consensus_confidence')}/10)")
            
        except Exception as e:
            print(f"[ERROR] Test {i} failed: {str(e)[:80]}")
            results.append({"test": i, "error": str(e)})
    
    success_rate = len([r for r in results if 'error' not in r]) / len(results) * 100
    
    print(f"[COMPLETED] Validation finished")
    print(f"[METRICS] Success rate: {success_rate:.0f}% | Total cost: ${total_cost:.4f}")
    
    return {"results": results, "success_rate": success_rate, "total_cost": total_cost}


def validate_api_connection(client: OpenAI) -> bool:
    """Test API connectivity and model availability.
    
    Args:
        client (OpenAI): Configured OpenAI client instance
        
    Returns:
        bool: True if connection is successful, False otherwise.
        
    Raises:
        Exception: If API connection fails or model is unavailable.
    """
    print("[INIT] Validating OpenAI API connection and model availability")
    
    try:
        print("[STEP] Testing API connectivity with minimal request")
        response = client.models.list()
        available_models = [model.id for model in response.data]
        print(f"[SUCCESS] API connection established - {len(available_models)} models available")
        
        required_models = ["gpt-4.1-mini", "gpt-4.1-nano"]
        available_required = [model for model in required_models if model in available_models]
        unavailable_required = [model for model in required_models if model not in available_models]
        
        print(f"[INFO] Required models available: {available_required}")
        if unavailable_required:
            print(f"[WARNING] Required models NOT available: {unavailable_required}")
            print(f"[INFO] Available GPT-4.1 variants: {[m for m in available_models if 'gpt-4.1' in m]}")
        
        test_model = available_required[1] if available_required else "gpt-3.5-turbo"
        print(f"[STEP] Testing model response with: {test_model}")
        
        test_response = client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "Test connection. Respond with 'OK'."}],
            max_tokens=10,
            temperature=0.1
        )
        
        response_content = test_response.choices[0].message.content
        print(f"[SUCCESS] Model response test completed: {response_content}")
        
        tokens_used = test_response.usage.total_tokens
        print(f"[INFO] Token usage tracking functional: {tokens_used} tokens")
        
        print("[COMPLETED] API connection validation successful")
        return True
        
    except Exception as e:
        print(f"[ERROR] API connection validation failed: {str(e)}")
        print("[FAILED] Unable to establish proper connection to OpenAI API")
        raise Exception(f"API validation failed: {str(e)}")