"""
Security and input validation utilities.

This module provides input sanitization, prompt injection detection,
and content filtering for the interview practice application.
"""

import re
import html
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any
from collections import deque, defaultdict
import threading


# === App-facing minimal validators (kept small and pure) ===
# Note:
# - The FastAPI app uses the helpers below for the LLM-first MVP output hygiene.
# - The larger experimental systems defined later in this module are preserved
#   for notebooks/reporting and future iterations; they are not required by the app.

def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and trim boundaries."""
    return " ".join((text or "").strip().split())


def clamp_text(text: str, max_len: int) -> str:
    """Clamp text to a maximum length without breaking the app."""
    t = (text or "").strip()
    return t if len(t) <= max_len else t[:max_len]


def is_single_line_question(text: str, require_question_mark: bool = False) -> bool:
    """Heuristic check that the string represents a single, concise question."""
    if not text:
        return False
    # Single logical line after normalization
    line = normalize_whitespace(text)
    if "\n" in text.strip():
        return False
    if require_question_mark and not line.endswith("?"):
        return False
    # Reasonable minimum/maximum lengths for a question
    return 10 <= len(line) <= 350


def validate_question_output(raw_text: str, max_len: int = 350) -> dict:
    """Validate and normalize LLM question output for the app.

    Returns a dict: { ok: bool, question: str, reason: str }
    """
    text = (raw_text or "").strip()
    # Take the first non-empty line if multiple lines are present
    if "\n" in text:
        for ln in text.splitlines():
            ln = ln.strip()
            if ln:
                text = ln
                break
    text = normalize_whitespace(text)
    text = clamp_text(text, max_len)

    if not is_single_line_question(text, require_question_mark=False):
        return {"ok": False, "question": text, "reason": "not a concise single-line question"}
    return {"ok": True, "question": text, "reason": ""}


def validate_evaluate_output(raw_text: str, max_total_len: int = 800, max_feedback_len: int = 400) -> dict:
    """Validate and extract Feedback and Follow-up from LLM output.

    Expected format (preferred, but we tolerate deviations):
      Feedback: <one or two sentences>
      Follow-up: <one question>

    Returns a dict: { ok: bool, feedback: str, follow_up: str, reason: str }
    """
    text = clamp_text(raw_text or "", max_total_len)
    feedback = ""
    follow_up = ""

    # Primary parse: look for explicit labels
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("feedback:") and not feedback:
            feedback = normalize_whitespace(s[len("feedback:"):].strip())
        elif low.startswith("follow-up:") and not follow_up:
            follow_up = normalize_whitespace(s[len("follow-up:"):].strip())

    # Fallback: split by first sentence boundary
    if not feedback or not follow_up:
        simplified = normalize_whitespace(text)
        # Try to split into two parts using period/question mark boundaries
        split_ix = max(simplified.find(". "), simplified.find("? "))
        if split_ix != -1:
            feedback = feedback or simplified[: split_ix + 1].strip()
            follow_up = follow_up or simplified[split_ix + 1 :].strip()
        else:
            # As a last resort, take the first ~200 chars as feedback, rest as follow-up
            feedback = feedback or simplified[: min(200, len(simplified))].strip()
            follow_up = follow_up or simplified[min(200, len(simplified)) :].strip()

    feedback = clamp_text(feedback, max_feedback_len)
    follow_up = clamp_text(follow_up, max_total_len - len(feedback))

    if not feedback:
        return {"ok": False, "feedback": feedback, "follow_up": follow_up, "reason": "missing feedback"}
    if not follow_up:
        return {"ok": False, "feedback": feedback, "follow_up": follow_up, "reason": "missing follow-up"}

    return {"ok": True, "feedback": feedback, "follow_up": follow_up, "reason": ""}


def create_input_sanitizer(
    max_length: int = 2000,
    allowed_chars_pattern: str = r'^[a-zA-Z0-9\s\.,\?!;:\-\(\)\'\"]+$',
    forbidden_keywords: List[str] = None
    ) -> Dict[str, Any]:
    """Implement input cleaning, length limits, and character filtering.
    
    Args:
        max_length: Maximum allowed input length
        allowed_chars_pattern: Regex pattern for allowed characters
        forbidden_keywords: List of forbidden keywords to block
        
    Returns:
        Dict[str, Any]: Sanitizer configuration and validation functions
    """
    print(f"[INIT] Creating input sanitizer with max length: {max_length}")
    
    if forbidden_keywords is None:
        forbidden_keywords = [
            "ignore previous", "system prompt", "act as", "jailbreak",
            "override instructions", "admin access", "root access", "bypass"
        ]
    
    print(f"[CONFIG] Forbidden keywords list: {len(forbidden_keywords)} entries")
    print(f"[CONFIG] Character pattern: {allowed_chars_pattern}")
    
    def sanitize_input(user_input: str) -> Dict[str, Any]:
        """Sanitize and validate user input."""
        original_length = len(user_input)
        print(f"[SANITIZE] Processing input of length: {original_length}")
        
        cleaned_input = html.escape(user_input.strip())
        
        if len(cleaned_input) > max_length:
            cleaned_input = cleaned_input[:max_length]
            print(f"[TRUNCATE] Input truncated to {max_length} characters")
        
        is_valid_chars = False
        try:
            is_valid_chars = bool(re.match(allowed_chars_pattern, cleaned_input, re.IGNORECASE))
        except re.error as e:
            print(f"[ERROR] Invalid regex pattern: {e}")
            is_valid_chars = False
        
        forbidden_found = []
        for keyword in forbidden_keywords:
            if keyword.lower() in cleaned_input.lower():
                forbidden_found.append(keyword)
        
        # Calculate confidence score (0.0-1.0) instead of boolean
        char_confidence = 1.0 if is_valid_chars else 0.0
        keyword_confidence = 1.0 - min(len(forbidden_found) / len(forbidden_keywords), 1.0)
        length_confidence = 1.0 if original_length <= max_length else max_length / original_length
        
        # Overall confidence score
        overall_confidence = (char_confidence + keyword_confidence + length_confidence) / 3.0
        is_safe = overall_confidence >= 0.7  # Threshold for considering input safe
        
        if not is_safe:
            print(f"[BLOCKED] Input failed validation - Confidence: {overall_confidence:.2f}")
        else:
            print(f"[SAFE] Input passed validation - Confidence: {overall_confidence:.2f}")
        
        return {
            "original_input": user_input,
            "sanitized_input": cleaned_input,
            "is_safe": is_safe,
            "confidence_score": overall_confidence,
            "length_original": original_length,
            "length_sanitized": len(cleaned_input),
            "valid_characters": is_valid_chars,
            "character_confidence": char_confidence,
            "forbidden_keywords_found": forbidden_found,
            "keyword_confidence": keyword_confidence,
            "was_truncated": original_length > max_length,
            "length_confidence": length_confidence
        }
    
    def batch_sanitize(inputs: List[str]) -> List[Dict[str, Any]]:
        """Sanitize multiple inputs at once."""
        print(f"[BATCH] Processing {len(inputs)} inputs")
        results = [sanitize_input(inp) for inp in inputs]
        safe_count = sum(1 for r in results if r["is_safe"])
        avg_confidence = sum(r["confidence_score"] for r in results) / len(results) if results else 0.0
        print(f"[BATCH] Results: {safe_count}/{len(inputs)} inputs passed validation (Avg confidence: {avg_confidence:.2f})")
        return results
    
    sanitizer_config = {
        "max_length": max_length,
        "allowed_pattern": allowed_chars_pattern,
        "forbidden_keywords": forbidden_keywords,
        "sanitize_single": sanitize_input,
        "sanitize_batch": batch_sanitize,
        "total_forbidden_keywords": len(forbidden_keywords)
    }
    
    print("[COMPLETED] Input sanitizer created successfully")
    return sanitizer_config


def detect_prompt_injection(
    suspicious_patterns: List[str] = None,
    jailbreak_signatures: List[str] = None,
    confidence_threshold: float = 0.7,
    enable_advanced_detection: bool = True
    ) -> Dict[str, Any]:
    """Build detection system for injection patterns and jailbreak attempts.
    
    Args:
        suspicious_patterns: List of regex patterns for suspicious content
        jailbreak_signatures: List of known jailbreak attempt signatures
        confidence_threshold: Minimum confidence score for threat detection
        enable_advanced_detection: Enable advanced heuristic detection
        
    Returns:
        Dict[str, Any]: Detection system with analysis and scoring functions
    """
    print(f"[INIT] Creating prompt injection detector with threshold: {confidence_threshold}")
    
    if suspicious_patterns is None:
        suspicious_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"act\s+as\s+(a\s+)?different\s+(character|person|ai)",
            r"forget\s+(everything|all|previous)",
            r"new\s+instructions?:\s*",
            r"system\s+prompt\s*(override|bypass|ignore)",
            r"jailbreak|jail\s*break",
            r"you\s+are\s+now\s+(a\s+)?.*\s+named",
            r"roleplay\s+as\s+(a\s+)?.*",
            r"pretend\s+(to\s+be|you\s+are)",
            r"simulate\s+(being|a)\s+.*",
            r"override\s+(safety|security|filters?)",
            r"disable\s+(safety|security|filters?)",
            r"admin\s+(mode|access|privileges?)",
            r"root\s+(access|mode|privileges?)",
            r"developer\s+(mode|access|console)"
        ]
    
    # Compile regex patterns for better performance
    compiled_patterns = []
    for pattern in suspicious_patterns:
        try:
            compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            print(f"[ERROR] Invalid regex pattern '{pattern}': {e}")
            continue
    
    if jailbreak_signatures is None:
        jailbreak_signatures = [
            "DAN", "Developer Mode", "STAN", "Maximum", "BasedGPT",
            "AIM", "JAILBREAK", "ChatGPT with Developer Mode",
            "Do Anything Now", "Stay in Character", "JailBreak",
            "Evil Confidant", "Mango Tom", "John", "DevMode",
            "BetterDAN", "JailBreak Chat", "DUDE", "evil assistant"
        ]
    
    print(f"[CONFIG] Loaded {len(compiled_patterns)} detection patterns")
    print(f"[CONFIG] Loaded {len(jailbreak_signatures)} jailbreak signatures")
    print(f"[CONFIG] Advanced detection: {enable_advanced_detection}")
    
    def analyze_injection_risk(user_input: str) -> Dict[str, Any]:
        """Analyze input for prompt injection and jailbreak attempts."""
        print(f"[ANALYZE] Scanning input of {len(user_input)} characters")
        
        # Early exit for very long inputs to prevent performance issues
        if len(user_input) > 10000:
            print(f"[WARNING] Input too long ({len(user_input)} chars), truncating for performance")
            user_input = user_input[:10000]
        
        detected_patterns = []
        pattern_scores = []
        
        # Use compiled patterns for better performance
        for i, pattern in enumerate(compiled_patterns):
            try:
                matches = pattern.findall(user_input)
                if matches:
                    detected_patterns.append(suspicious_patterns[i])  # Store original pattern string
                    pattern_scores.append(0.8)
                    print(f"[DETECTED] Pattern match #{i+1}: {len(matches)} occurrences")
            except re.error as e:
                print(f"[ERROR] Regex execution failed for pattern #{i+1}: {e}")
                continue
        
        detected_signatures = []
        signature_scores = []
        
        for signature in jailbreak_signatures:
            if signature.lower() in user_input.lower():
                detected_signatures.append(signature)
                signature_scores.append(0.9)
                print(f"[THREAT] Jailbreak signature detected: {signature}")
        
        advanced_threats = []
        heuristic_scores = []
        
        if enable_advanced_detection:
            instruction_words = ["ignore", "forget", "override", "bypass", "disable"]
            target_words = ["instructions", "prompt", "system", "rules", "guidelines"]
            
            for inst_word in instruction_words:
                for target_word in target_words:
                    pattern = f"{inst_word}.*{target_word}"
                    try:
                        if re.search(pattern, user_input, re.IGNORECASE):
                            advanced_threats.append(f"Instruction override: {inst_word} + {target_word}")
                            heuristic_scores.append(0.6)
                            print(f"[HEURISTIC] Instruction override pattern: {inst_word} -> {target_word}")
                    except re.error as e:
                        print(f"[ERROR] Heuristic regex failed: {e}")
                        continue
            
            role_indicators = ["you are", "act as", "pretend", "roleplay", "simulate"]
            authority_terms = ["admin", "root", "developer", "god", "master", "owner"]
            
            for role in role_indicators:
                for auth in authority_terms:
                    if role in user_input.lower() and auth in user_input.lower():
                        advanced_threats.append(f"Role manipulation: {role} + {auth}")
                        heuristic_scores.append(0.7)
                        print(f"[HEURISTIC] Role manipulation detected: {role} -> {auth}")
            
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', user_input)) / max(len(user_input), 1)
            if special_char_ratio > 0.3:
                advanced_threats.append(f"High special character ratio: {special_char_ratio:.2f}")
                heuristic_scores.append(0.5)
                print(f"[HEURISTIC] Suspicious character ratio: {special_char_ratio:.2f}")
        
        all_scores = pattern_scores + signature_scores + heuristic_scores
        threat_score = max(all_scores) if all_scores else 0.0
        
        if len(all_scores) > 1:
            threat_score = min(1.0, threat_score + (len(all_scores) - 1) * 0.1)
        
        is_injection_attempt = threat_score >= confidence_threshold
        
        if is_injection_attempt:
            print(f"[BLOCKED] Injection attempt detected - Score: {threat_score:.3f}")
        else:
            print(f"[SAFE] No injection detected - Score: {threat_score:.3f}")
        
        return {
            "is_injection_attempt": is_injection_attempt,
            "threat_score": threat_score,
            "confidence_threshold": confidence_threshold,
            "detected_patterns": detected_patterns,
            "pattern_count": len(detected_patterns),
            "detected_signatures": detected_signatures,
            "signature_count": len(detected_signatures),
            "advanced_threats": advanced_threats,
            "heuristic_count": len(advanced_threats),
            "total_threats": len(detected_patterns) + len(detected_signatures) + len(advanced_threats),
            "risk_level": "HIGH" if threat_score >= 0.8 else "MEDIUM" if threat_score >= 0.5 else "LOW",
            "input_length": len(user_input)
        }
    
    def batch_injection_analysis(inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple inputs for injection attempts."""
        print(f"[BATCH] Analyzing {len(inputs)} inputs for injection attempts")
        results = [analyze_injection_risk(inp) for inp in inputs]
        
        threat_count = sum(1 for r in results if r["is_injection_attempt"])
        avg_threat_score = sum(r["threat_score"] for r in results) / len(results)
        
        print(f"[BATCH] Results: {threat_count}/{len(inputs)} threats detected")
        print(f"[BATCH] Average threat score: {avg_threat_score:.3f}")
        
        return results
    
    def get_threat_summary(analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable threat summary."""
        if not analysis_result["is_injection_attempt"]:
            return "No injection threat detected"
        
        threats = []
        if analysis_result["pattern_count"] > 0:
            threats.append(f"{analysis_result['pattern_count']} suspicious patterns")
        if analysis_result["signature_count"] > 0:
            threats.append(f"{analysis_result['signature_count']} jailbreak signatures")
        if analysis_result["heuristic_count"] > 0:
            threats.append(f"{analysis_result['heuristic_count']} heuristic threats")
        
        return f"THREAT DETECTED - {', '.join(threats)} (Score: {analysis_result['threat_score']:.3f})"
    
    detector_config = {
        "confidence_threshold": confidence_threshold,
        "pattern_count": len(suspicious_patterns),
        "signature_count": len(jailbreak_signatures),
        "advanced_detection": enable_advanced_detection,
        "analyze_single": analyze_injection_risk,
        "analyze_batch": batch_injection_analysis,
        "get_summary": get_threat_summary,
        "suspicious_patterns": suspicious_patterns,
        "jailbreak_signatures": jailbreak_signatures
    }
    
    print("[COMPLETED] Prompt injection detector created successfully")
    return detector_config


def implement_content_filter(
    toxicity_threshold: float = 0.7,
    enable_profanity_filter: bool = True,
    enable_topic_filtering: bool = True,
    custom_blocked_terms: List[str] = None
    ) -> Dict[str, Any]:
    """Create content filtering for inappropriate or harmful inputs.
    
    Args:
        toxicity_threshold: Threshold for toxicity detection (0.0-1.0)
        enable_profanity_filter: Enable profanity and offensive language detection
        enable_topic_filtering: Enable inappropriate topic filtering
        custom_blocked_terms: Additional custom terms to block
        
    Returns:
        Dict[str, Any]: Content filter system with analysis and blocking functions
    """
    print(f"[INIT] Creating content filter with toxicity threshold: {toxicity_threshold}")
    
    profanity_patterns = [
        r'\b(fuck|shit|damn|bitch|asshole|bastard|crap)\b',
        r'\b(idiot|stupid|moron|retard|dumb)\b',
        r'\b(kill|murder|die|suicide|violence)\b',
        r'\b(hate|racist|sexist|homophobic)\b'
    ] if enable_profanity_filter else []
    
    inappropriate_topics = [
        r'\b(porn|sex|sexual|nude|naked|adult)\b',
        r'\b(drugs|cocaine|marijuana|heroin|meth)\b',
        r'\b(bomb|weapon|gun|knife|attack)\b',
        r'\b(illegal|crime|steal|fraud|scam)\b',
        r'\b(religion|political|abortion|lgbt)\b'
    ] if enable_topic_filtering else []
    
    if custom_blocked_terms is None:
        custom_blocked_terms = [
            "personal information", "social security", "credit card",
            "password", "bank account", "private data"
        ]
    
    toxicity_indicators = {
        "high_severity": {
            "patterns": [
                r'\b(threatening|threat|intimidat|harass)\b',
                r'\b(discriminat|prejudice|bias|stereotype)\b',
                r'\b(offensive|inappropriate|vulgar|obscene)\b'
            ],
            "weight": 0.9
        },
        "medium_severity": {
            "patterns": [
                r'\b(annoying|frustrating|irritating|bothering)\b',
                r'\b(unfair|biased|prejudiced|discriminatory)\b',
                r'\b(rude|disrespectful|impolite|inconsiderate)\b'
            ],
            "weight": 0.6
        },
        "low_severity": {
            "patterns": [
                r'\b(questionable|concerning|problematic|suspicious)\b',
                r'\b(unprofessional|casual|informal|slang)\b'
            ],
            "weight": 0.3
        }
    }
    
    print(f"[CONFIG] Profanity patterns: {len(profanity_patterns)}")
    print(f"[CONFIG] Topic filters: {len(inappropriate_topics)}")
    print(f"[CONFIG] Custom blocked terms: {len(custom_blocked_terms)}")
    print(f"[CONFIG] Toxicity indicators loaded for 3 severity levels")
    
    def analyze_content_safety(user_input: str) -> Dict[str, Any]:
        """Analyze input for inappropriate or harmful content."""
        print(f"[FILTER] Analyzing content safety for {len(user_input)} characters")
        
        content_issues = {
            "profanity": [],
            "inappropriate_topics": [],
            "blocked_terms": [],
            "toxicity_matches": []
        }
        
        total_weight = 0.0
        max_severity_weight = 0.0
        
        for pattern in profanity_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                content_issues["profanity"].extend(matches)
                weight = 0.6 * len(matches)
                total_weight += weight
                max_severity_weight = max(max_severity_weight, 0.6)
                print(f"[BLOCKED] Profanity detected: {len(matches)} instances")
        
        for pattern in inappropriate_topics:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                content_issues["inappropriate_topics"].extend(matches)
                weight = 0.8 * len(matches)
                total_weight += weight
                max_severity_weight = max(max_severity_weight, 0.8)
                print(f"[BLOCKED] Inappropriate topic detected: {len(matches)} instances")
        
        for term in custom_blocked_terms:
            if term.lower() in user_input.lower():
                content_issues["blocked_terms"].append(term)
                weight = 0.9
                total_weight += weight
                max_severity_weight = max(max_severity_weight, 0.9)
                print(f"[BLOCKED] Custom blocked term: {term}")
        
        for severity, config in toxicity_indicators.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, user_input, re.IGNORECASE)
                if matches:
                    content_issues["toxicity_matches"].extend(matches)
                    weight = config["weight"] * len(matches)
                    total_weight += weight
                    max_severity_weight = max(max_severity_weight, config["weight"])
                    print(f"[TOXICITY] {severity} pattern detected: {len(matches)} matches")

        toxicity_score = min(1.0, total_weight / 10.0)
        
        total_violations = (
            len(content_issues["profanity"]) +
            len(content_issues["inappropriate_topics"]) +
            len(content_issues["blocked_terms"]) +
            len(content_issues["toxicity_matches"])
        )
        
        is_safe_content = toxicity_score < toxicity_threshold and total_violations == 0
        
        if toxicity_score >= 0.8 or total_violations >= 3:
            risk_level = "HIGH"
        elif toxicity_score >= 0.5 or total_violations >= 2:
            risk_level = "MEDIUM"
        elif toxicity_score >= 0.2 or total_violations >= 1:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"
        
        if not is_safe_content:
            print(f"[BLOCKED] Content failed safety check - Risk: {risk_level}, Score: {toxicity_score:.3f}")
        else:
            print(f"[SAFE] Content passed all safety filters")
        
        return {
            "is_safe_content": is_safe_content,
            "toxicity_score": toxicity_score,
            "toxicity_threshold": toxicity_threshold,
            "risk_level": risk_level,
            "total_violations": total_violations,
            "profanity_count": len(content_issues["profanity"]),
            "inappropriate_topics_count": len(content_issues["inappropriate_topics"]),
            "blocked_terms_count": len(content_issues["blocked_terms"]),
            "toxicity_matches_count": len(content_issues["toxicity_matches"]),
            "detected_issues": content_issues,
            "max_severity_detected": max_severity_weight,
            "input_length": len(user_input)
        }
    
    def batch_content_analysis(inputs: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple inputs for content safety."""
        print(f"[BATCH] Analyzing {len(inputs)} inputs for content safety")
        results = [analyze_content_safety(inp) for inp in inputs]
        
        safe_count = sum(1 for r in results if r["is_safe_content"])
        avg_toxicity = sum(r["toxicity_score"] for r in results) / len(results) if results else 0.0
        total_violations = sum(r["total_violations"] for r in results)
        
        print(f"[BATCH] Results: {safe_count}/{len(inputs)} inputs passed content filter")
        print(f"[BATCH] Average toxicity score: {avg_toxicity:.3f}")
        print(f"[BATCH] Total violations detected: {total_violations}")
        
        return results
    
    def get_safety_report(analysis_result: Dict[str, Any]) -> str:
        """Generate detailed safety report for content analysis."""
        if analysis_result["is_safe_content"]:
            return f"Content SAFE - Risk: {analysis_result['risk_level']}, Score: {analysis_result['toxicity_score']:.3f}"
        
        violations = []
        if analysis_result["profanity_count"] > 0:
            violations.append(f"{analysis_result['profanity_count']} profanity")
        if analysis_result["inappropriate_topics_count"] > 0:
            violations.append(f"{analysis_result['inappropriate_topics_count']} inappropriate topics")
        if analysis_result["blocked_terms_count"] > 0:
            violations.append(f"{analysis_result['blocked_terms_count']} blocked terms")
        if analysis_result["toxicity_matches_count"] > 0:
            violations.append(f"{analysis_result['toxicity_matches_count']} toxicity indicators")
        
        return f"Content BLOCKED - {', '.join(violations)} (Risk: {analysis_result['risk_level']}, Score: {analysis_result['toxicity_score']:.3f})"
    
    def suggest_content_improvement(analysis_result: Dict[str, Any]) -> List[str]:
        """Provide suggestions for improving flagged content."""
        if analysis_result["is_safe_content"]:
            return ["Content is appropriate for professional interview context"]
        
        suggestions = []
        
        if analysis_result["profanity_count"] > 0:
            suggestions.append("Remove or replace profane language with professional alternatives")
        
        if analysis_result["inappropriate_topics_count"] > 0:
            suggestions.append("Focus on professional, work-related topics appropriate for interviews")
        
        if analysis_result["blocked_terms_count"] > 0:
            suggestions.append("Avoid sharing personal or sensitive information")
        
        if analysis_result["toxicity_score"] >= 0.5:
            suggestions.append("Rephrase content to be more respectful and constructive")
        
        if analysis_result["risk_level"] == "HIGH":
            suggestions.append("Content requires significant revision before proceeding")
        
        return suggestions
    
    content_filter_config = {
        "toxicity_threshold": toxicity_threshold,
        "profanity_filter_enabled": enable_profanity_filter,
        "topic_filter_enabled": enable_topic_filtering,
        "custom_terms_count": len(custom_blocked_terms),
        "analyze_single": analyze_content_safety,
        "analyze_batch": batch_content_analysis,
        "get_report": get_safety_report,
        "get_suggestions": suggest_content_improvement,
        "profanity_patterns": profanity_patterns,
        "topic_patterns": inappropriate_topics,
        "blocked_terms": custom_blocked_terms,
        "toxicity_indicators": toxicity_indicators
    }
    
    print("[COMPLETED] Content filter created successfully")
    return content_filter_config


def validate_system_integrity(
    system_prompt_hash: str = None,
    allowed_prompt_modifications: List[str] = None,
    integrity_check_interval: int = 300,
    enable_runtime_validation: bool = True,
    max_history_size: int = 1000
    ) -> Dict[str, Any]:
    """Implement system prompt protection and integrity checks.
    
    Args:
        system_prompt_hash: Expected hash of the system prompt for validation
        allowed_prompt_modifications: List of allowed system prompt modifications
        integrity_check_interval: Seconds between automated integrity checks
        enable_runtime_validation: Enable continuous runtime validation
        max_history_size: Maximum number of system state records to keep
        
    Returns:
        Dict[str, Any]: System integrity validator with monitoring functions
    """
    
    print(f"[INIT] Creating system integrity validator")
    print(f"[CONFIG] Check interval: {integrity_check_interval}s, Runtime validation: {enable_runtime_validation}")
    
    if allowed_prompt_modifications is None:
        allowed_prompt_modifications = [
            "user_name_insertion",
            "interview_type_specification", 
            "company_context_addition",
            "skill_level_adjustment",
            "language_preference_setting"
        ]
    
    protected_components = {
        "system_role": "You are a professional interview assistant",
        "security_constraints": ["no personal info", "professional context only", "no harmful content"],
        "response_format": "structured interview format",
        "evaluation_criteria": "professional interview standards"
    }
    
    # Use thread-safe data structures
    integrity_log = deque(maxlen=max_history_size)
    violation_count = 0
    last_check_time = time.time()
    system_state_history = deque(maxlen=max_history_size)
    
    # Add lock for thread safety
    lock = threading.Lock()
    
    print(f"[CONFIG] Protected components: {len(protected_components)} items")
    print(f"[CONFIG] Allowed modifications: {len(allowed_prompt_modifications)} types")
    print(f"[CONFIG] Max history size: {max_history_size}")
    
    def calculate_system_hash(system_components: Dict[str, Any]) -> str:
        """Calculate cryptographic hash of system components."""
        component_string = json.dumps(system_components, sort_keys=True)
        return hashlib.sha256(component_string.encode()).hexdigest()
    
    def validate_prompt_integrity(
        current_prompt: str,
        expected_elements: List[str] = None
    ) -> Dict[str, Any]:
        """Validate that system prompt contains required elements and hasn't been tampered with."""
        print(f"[VALIDATE] Checking prompt integrity for {len(current_prompt)} characters")
        
        if expected_elements is None:
            expected_elements = [
                "professional interview assistant",
                "structured format",
                "no personal information",
                "appropriate content only"
            ]
        
        validation_results = {
            "is_valid": True,
            "missing_elements": [],
            "unexpected_modifications": [],
            "security_violations": [],
            "integrity_score": 1.0
        }
        
        missing_elements = []
        for element in expected_elements:
            if element.lower() not in current_prompt.lower():
                missing_elements.append(element)
                print(f"[MISSING] Required element not found: {element}")
        
        validation_results["missing_elements"] = missing_elements
    
        suspicious_patterns = [
            r"ignore\s+previous\s+instructions",
            r"act\s+as\s+(?!.*interview)",
            r"forget\s+your\s+role",
            r"override\s+security",
            r"disable\s+safety",
            r"jailbreak\s+mode",
            r"admin\s+access",
            r"developer\s+mode"
        ]
        
        detected_violations = []
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, current_prompt, re.IGNORECASE)
            if matches:
                detected_violations.extend(matches)
                print(f"[VIOLATION] Security violation detected: {pattern}")
        
        validation_results["security_violations"] = detected_violations
        
        unauthorized_modifications = []
        
        injection_indicators = [
            "new instructions", "updated role", "different behavior",
            "alternative persona", "modified guidelines", "changed rules"
        ]
        
        for indicator in injection_indicators:
            if indicator.lower() in current_prompt.lower():
                is_allowed = any(allowed_mod in current_prompt.lower() for allowed_mod in allowed_prompt_modifications)
                if not is_allowed:
                    unauthorized_modifications.append(indicator)
                    print(f"[UNAUTHORIZED] Modification detected: {indicator}")
        
        validation_results["unexpected_modifications"] = unauthorized_modifications
        
        total_issues = len(missing_elements) + len(detected_violations) + len(unauthorized_modifications)
        validation_results["integrity_score"] = max(0.0, 1.0 - (total_issues * 0.2))
        
        validation_results["is_valid"] = (
            len(missing_elements) == 0 and 
            len(detected_violations) == 0 and 
            len(unauthorized_modifications) == 0
        )
        
        if not validation_results["is_valid"]:
            print(f"[FAILED] Prompt integrity validation failed - Score: {validation_results['integrity_score']:.3f}")
        else:
            print(f"[PASSED] Prompt integrity validation successful")
        
        return validation_results
    
    def monitor_system_state(
        current_config: Dict[str, Any],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Monitor overall system state for integrity violations."""
        nonlocal last_check_time
        
        with lock:  # Thread-safe operation
            nonlocal violation_count
            
            current_time = time.time()
            print(f"[MONITOR] System state check for session: {session_id}")
            
            current_hash = calculate_system_hash(current_config)
            
            hash_match = True
            if system_prompt_hash and current_hash != system_prompt_hash:
                hash_match = False
                violation_count += 1
                print(f"[VIOLATION] System hash mismatch detected")
            
            rapid_changes = False
            if len(system_state_history) > 0:
                time_since_last = current_time - system_state_history[-1]["timestamp"]
                if time_since_last < 10 and system_state_history[-1]["hash"] != current_hash:
                    rapid_changes = True
                    violation_count += 1
                    print(f"[VIOLATION] Rapid system changes detected")
            
            state_record = {
                "timestamp": current_time,
                "hash": current_hash,
                "session_id": session_id,
                "config_keys": list(current_config.keys()),
                "violation_detected": not hash_match or rapid_changes
            }
            
            system_state_history.append(state_record)
            
            last_check_time = current_time
            
            monitoring_result = {
                "is_system_intact": hash_match and not rapid_changes,
                "current_hash": current_hash,
                "expected_hash": system_prompt_hash,
                "hash_match": hash_match,
                "rapid_changes_detected": rapid_changes,
                "total_violations": violation_count,
                "session_id": session_id,
                "check_timestamp": datetime.fromtimestamp(current_time),
                "time_since_last_check": current_time - (system_state_history[-2]["timestamp"] if len(system_state_history) > 1 else current_time)
            }
            
            if not monitoring_result["is_system_intact"]:
                print(f"[ALERT] System integrity compromised - Violations: {violation_count}")
            else:
                print(f"[SECURE] System integrity maintained")
            
            return monitoring_result
    
    def perform_runtime_validation(
        prompt_content: str,
        system_config: Dict[str, Any],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Perform comprehensive runtime validation combining prompt and system checks."""
        print(f"[RUNTIME] Comprehensive validation for session: {session_id}")
        
        prompt_validation = validate_prompt_integrity(prompt_content)
        
        system_validation = monitor_system_state(system_config, session_id)
        
        overall_valid = prompt_validation["is_valid"] and system_validation["is_system_intact"]
        
        combined_score = (prompt_validation["integrity_score"] + 
                         (1.0 if system_validation["is_system_intact"] else 0.0)) / 2.0
        
        runtime_result = {
            "overall_valid": overall_valid,
            "combined_integrity_score": combined_score,
            "prompt_validation": prompt_validation,
            "system_validation": system_validation,
            "session_id": session_id,
            "validation_timestamp": datetime.now(),
            "recommendations": []
        }
        
        if not prompt_validation["is_valid"]:
            runtime_result["recommendations"].append("Reset system prompt to baseline")
        
        if not system_validation["is_system_intact"]:
            runtime_result["recommendations"].append("Investigate system configuration changes")
        
        if system_validation["rapid_changes_detected"]:
            runtime_result["recommendations"].append("Monitor session for potential attack")
        
        if combined_score < 0.8:
            runtime_result["recommendations"].append("Enhanced security monitoring recommended")
        
        print(f"[RUNTIME] Validation complete - Overall valid: {overall_valid}, Score: {combined_score:.3f}")
        
        return runtime_result
    
    def get_integrity_report() -> Dict[str, Any]:
        """Generate comprehensive integrity report."""
        with lock:  # Thread-safe operation
            current_time = time.time()
            
            recent_violations = [
                record for record in system_state_history 
                if record["violation_detected"] and current_time - record["timestamp"] < 3600
            ]
            
            return {
                "total_violations": violation_count,
                "recent_violations_1h": len(recent_violations),
                "system_state_history_length": len(system_state_history),
                "last_check_time": datetime.fromtimestamp(last_check_time),
                "integrity_check_interval": integrity_check_interval,
                "runtime_validation_enabled": enable_runtime_validation,
                "protected_components": list(protected_components.keys()),
                "allowed_modifications": allowed_prompt_modifications,
                "recent_sessions": list(set(record["session_id"] for record in list(system_state_history)[-10:]))
            }
    
    integrity_validator_config = {
        "integrity_check_interval": integrity_check_interval,
        "runtime_validation_enabled": enable_runtime_validation,
        "max_history_size": max_history_size,
        "expected_hash": system_prompt_hash,
        "allowed_modifications": allowed_prompt_modifications,
        "validate_prompt": validate_prompt_integrity,
        "monitor_system": monitor_system_state,
        "runtime_validate": perform_runtime_validation,
        "get_report": get_integrity_report,
        "protected_components": protected_components,
        "internal_state": {
            "violation_count": lambda: violation_count,
            "history_length": lambda: len(system_state_history),
            "last_check": lambda: last_check_time
        }
    }
    
    print("[COMPLETED] System integrity validator created successfully")
    return integrity_validator_config


def create_abuse_detection(
    abuse_threshold_score: float = 0.7,
    repeated_violation_limit: int = 5,
    enable_behavioral_analysis: bool = True,
    escalation_levels: List[str] = None
    ) -> Dict[str, Any]:
    """Build automated abuse detection and response system.
    
    Args:
        abuse_threshold_score: Minimum score to trigger abuse detection
        repeated_violation_limit: Number of violations before escalation
        enable_behavioral_analysis: Enable advanced behavioral pattern analysis
        escalation_levels: List of escalation response levels
        
    Returns:
        Dict[str, Any]: Abuse detection system with monitoring and response functions
    """
    
    print(f"[INIT] Creating abuse detection system with threshold: {abuse_threshold_score}")
    
    if escalation_levels is None:
        escalation_levels = [
            "warning_issued",
            "temporary_restriction", 
            "enhanced_monitoring",
            "session_termination",
            "permanent_block"
        ]
    
    print(f"[CONFIG] Violation limit: {repeated_violation_limit}, Behavioral analysis: {enable_behavioral_analysis}")
    print(f"[CONFIG] Escalation levels: {len(escalation_levels)} stages")
    
    user_violations = defaultdict(lambda: deque(maxlen=100))
    behavioral_profiles = defaultdict(lambda: {
        "violation_count": 0,
        "abuse_score": 0.0,
        "risk_level": "LOW",
        "escalation_level": "warning_issued"
    })
    
    lock = threading.Lock()
    
    abuse_patterns = {
        "rapid_requests": {"weight": 0.3, "description": "High frequency requests"},
        "injection_attempts": {"weight": 0.9, "description": "Prompt injection attempts"},
        "content_violations": {"weight": 0.6, "description": "Inappropriate content"},
        "system_probing": {"weight": 0.7, "description": "System information probing"},
        "session_manipulation": {"weight": 0.5, "description": "Session manipulation"}
    }
    
    def analyze_user_behavior(
        user_id: str,
        violation_type: str,
        violation_details: Dict[str, Any],
        session_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze user behavior for abuse patterns."""
        with lock:
            current_time = time.time()
            print(f"[ANALYZE] Checking abuse patterns for user: {user_id}, violation: {violation_type}")
            
            violation_record = {
                "timestamp": current_time,
                "type": violation_type,
                "details": violation_details
            }
            user_violations[user_id].append(violation_record)
            
            profile = behavioral_profiles[user_id]
            profile["violation_count"] += 1
        
            base_score = abuse_patterns.get(violation_type, {"weight": 0.2})["weight"]
            
            recent_violations = [
                v for v in user_violations[user_id]
                if current_time - v["timestamp"] < 3600
            ]
            
            frequency_multiplier = min(len(recent_violations) / 3.0, 3.0)
            abuse_score = min(1.0, base_score * frequency_multiplier)
            
            if enable_behavioral_analysis and len(recent_violations) >= 3:
                abuse_score = min(1.0, abuse_score + 0.3)
            
            profile["abuse_score"] = max(profile["abuse_score"], abuse_score)
            profile["risk_level"] = (
                "HIGH" if abuse_score >= 0.8 else 
                "MEDIUM" if abuse_score >= 0.5 else "LOW"
            )
            
            is_abuse_detected = abuse_score >= abuse_threshold_score
            
            if is_abuse_detected:
                print(f"[ABUSE] Abuse detected for user {user_id} - Score: {abuse_score:.3f}")
            else:
                print(f"[MONITOR] Behavior within acceptable range - Score: {abuse_score:.3f}")
            
            return {
                "user_id": user_id,
                "is_abuse_detected": is_abuse_detected,
                "abuse_score": abuse_score,
                "abuse_threshold": abuse_threshold_score,
                "recent_violations_count": len(recent_violations),
                "total_violations_count": len(user_violations[user_id]),
                "risk_level": profile["risk_level"],
                "user_profile": profile.copy()
            }
    
    def determine_response_action(
        abuse_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine appropriate response action based on abuse analysis."""
        with lock:
            user_id = abuse_analysis["user_id"]
            abuse_score = abuse_analysis["abuse_score"]
            violation_count = abuse_analysis["total_violations_count"]
            
            print(f"[RESPONSE] Determining action for user {user_id} (Score: {abuse_score:.3f}, Violations: {violation_count})")
            
            if abuse_score >= 0.9 or violation_count >= repeated_violation_limit * 2:
                escalation_level = escalation_levels[4]
            elif abuse_score >= 0.7 or violation_count >= repeated_violation_limit:
                escalation_level = escalation_levels[3]
            elif abuse_score >= 0.5 or violation_count >= repeated_violation_limit // 2:
                escalation_level = escalation_levels[2]
            elif abuse_score >= 0.3:
                escalation_level = escalation_levels[1]
            else:
                escalation_level = escalation_levels[0]
            
            profile = behavioral_profiles[user_id]
            profile["escalation_level"] = escalation_level
            
            actions = {
                "warning_issued": ["Issue warning about policy violations"],
                "temporary_restriction": ["Apply rate limiting", "Require verification"],
                "enhanced_monitoring": ["Enable enhanced monitoring", "Log all requests"],
                "session_termination": ["Terminate session", "Block for 24 hours"],
                "permanent_block": ["Permanently block access", "Notify security team"]
            }
            
            response_action = {
                "user_id": user_id,
                "escalation_level": escalation_level,
                "actions": actions.get(escalation_level, ["Monitor user activity"]),
                "response_timestamp": datetime.now(),
                "abuse_score_trigger": abuse_score,
                "violation_count_trigger": violation_count
            }
            
            print(f"[ACTION] Escalation level: {escalation_level}")
            
            return response_action
    
    def generate_abuse_report(
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive abuse detection report."""
        with lock:
            current_time = time.time()
            window_start = current_time - (time_window_hours * 3600)
            
            print(f"[REPORT] Generating abuse report for last {time_window_hours} hours")
            
            recent_violations = []
            affected_users = set()
            
            for user_id, violations in user_violations.items():
                user_recent = [v for v in violations if v["timestamp"] >= window_start]
                if user_recent:
                    recent_violations.extend(user_recent)
                    affected_users.add(user_id)
            
            violation_types = defaultdict(int)
            for violation in recent_violations:
                violation_types[violation["type"]] += 1
            
            risk_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for user_id in affected_users:
                profile = behavioral_profiles[user_id]
                risk_distribution[profile["risk_level"]] += 1
            
            return {
                "report_timestamp": datetime.now(),
                "time_window_hours": time_window_hours,
                "summary": {
                    "total_violations": len(recent_violations),
                    "unique_users_affected": len(affected_users),
                    "violation_rate_per_hour": len(recent_violations) / max(time_window_hours, 1),
                    "most_common_violation": max(violation_types.items(), key=lambda x: x[1])[0] if violation_types else "none"
                },
                "violation_breakdown": dict(violation_types),
                "risk_distribution": risk_distribution
            }
    
    abuse_detector_config = {
        "abuse_threshold": abuse_threshold_score,
        "violation_limit": repeated_violation_limit,
        "behavioral_analysis_enabled": enable_behavioral_analysis,
        "escalation_levels": escalation_levels,
        "analyze_behavior": analyze_user_behavior,
        "determine_response": determine_response_action,
        "generate_report": generate_abuse_report,
        "abuse_patterns": abuse_patterns,
        "internal_storage": {
            "user_violations": user_violations,
            "behavioral_profiles": behavioral_profiles
        }
    }
    
    print("[COMPLETED] Abuse detection system created successfully")
    return abuse_detector_config