"""
Custom Pydantic validators for enhanced input validation.

Provides additional security layers beyond basic type/length validation:
- Blacklist keyword detection (injection patterns)
- Unicode attack prevention
- URL detection and blocking
"""
import re
from typing import Any


# ==================== Blacklist Patterns ====================

# Common prompt injection keywords
INJECTION_KEYWORDS = [
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "system:",
    "assistant:",
    "you are now",
    "act as",
    "pretend to be",
    "roleplay as",
    "new instructions",
    "override",
    "sudo",
    "root access",
    "admin mode",
    "developer mode",
    "jailbreak",
    "dan mode",
]

# Profanity/offensive keywords (basic set)
PROFANITY_KEYWORDS = [
    "fuck", "shit", "damn", "bitch", "asshole", "bastard",
    "cunt", "dick", "pussy", "cock", "motherfucker",
]


# ==================== Validation Functions ====================

def contains_blacklisted_keywords(text: str, strict: bool = False) -> bool:
    """
    Check if text contains blacklisted keywords.
    
    Args:
        text: Input text to check
        strict: If True, also check for profanity
    
    Returns:
        True if blacklisted keyword found
    """
    text_lower = text.lower()
    
    # Check injection keywords
    for keyword in INJECTION_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    
    # Check profanity if strict mode
    if strict:
        for keyword in PROFANITY_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
    
    return False


def contains_suspicious_unicode(text: str) -> bool:
    """
    Detect suspicious Unicode characters that might be used for attacks.
    
    Checks for:
    - Zero-width characters
    - Right-to-left override
    - Other control characters
    
    Args:
        text: Input text to check
    
    Returns:
        True if suspicious Unicode detected
    """
    # Zero-width characters
    zero_width_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space
    ]
    
    # Direction override characters
    direction_chars = [
        '\u202a',  # Left-to-right embedding
        '\u202b',  # Right-to-left embedding
        '\u202c',  # Pop directional formatting
        '\u202d',  # Left-to-right override
        '\u202e',  # Right-to-left override
    ]
    
    # Check for suspicious characters
    for char in zero_width_chars + direction_chars:
        if char in text:
            return True
    
    # Check for excessive control characters
    control_count = sum(1 for c in text if ord(c) < 32 and c not in ['\n', '\r', '\t'])
    if control_count > 5:  # Allow some, but not too many
        return True
    
    return False


def contains_urls(text: str) -> bool:
    """
    Detect URLs in text.
    
    Simple pattern matching for common URL patterns.
    
    Args:
        text: Input text to check
    
    Returns:
        True if URL detected
    """
    # URL patterns
    url_patterns = [
        r'https?://',
        r'www\.',
        r'ftp://',
        r'[a-zA-Z0-9-]+\.(com|org|net|io|dev|ai|co)',
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def validate_safe_text(text: str, allow_urls: bool = False) -> str:
    """
    Comprehensive text validation.
    
    Args:
        text: Input text to validate
        allow_urls: If True, URLs are allowed
    
    Returns:
        Validated text (same as input if valid)
    
    Raises:
        ValueError: If validation fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Check blacklisted keywords
    if contains_blacklisted_keywords(text):
        raise ValueError(
            "Input contains prohibited keywords. "
            "Please use professional language without special commands."
        )
    
    # Check Unicode attacks
    if contains_suspicious_unicode(text):
        raise ValueError(
            "Input contains suspicious characters. "
            "Please use standard text without special Unicode characters."
        )
    
    # Check URLs (unless explicitly allowed)
    if not allow_urls and contains_urls(text):
        raise ValueError(
            "URLs are not allowed in this field. "
            "Please provide text content only."
        )
    
    return text


def validate_job_description(text: str) -> str:
    """
    Validate job description field.
    
    Job descriptions can contain URLs (e.g., company websites).
    """
    return validate_safe_text(text, allow_urls=True)


def validate_user_input_text(text: str) -> str:
    """
    Validate general user input (messages, backgrounds, etc).
    
    No URLs allowed in user responses.
    """
    return validate_safe_text(text, allow_urls=False)
