"""
Complete JSON processing pipeline for production use.

NOTE: Experimental functions currently used in notebooks for research and reporting.
Potential future integration into production API.

This module packages all JSON processing functionality for the
interview practice application.
"""

import json
from typing import Dict, Any
from jsonschema import validate

# Import from core modules
from .json_utils import (
    implement_schema_validation,
    parse_llm_responses,
    implement_error_recovery
)
from core.schemas import create_question_schema, create_response_schema
from core.llm_integration import (
    implement_structured_generation,
    validate_json_structure
)


def export_json_utilities(
    question_schema_config: Dict[str, Any],
    response_schema_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Package complete JSON processing pipeline for production use.
    
    Args:
        question_schema_config: Question schema configuration
        response_schema_config: Response schema configuration
        
    Returns:
        Dict[str, Any]: Complete JSON utilities package
    """
    def validate_json_structure_wrapper(data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """Validate JSON data against schemas."""
        if schema_type == "question":
            schema = question_schema_config["schema"]
        elif schema_type == "response":
            schema = response_schema_config["schema"]
        else:
            return {"valid": False, "error": "Unknown schema type"}
        
        try:
            validate(instance=data, schema=schema)
            return {"valid": True}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    json_utilities = {
        "version": "1.0.0",
        "schemas": {
            "question": question_schema_config,
            "response": response_schema_config
        },
        "pipeline": {
            "validate_structure": validate_json_structure_wrapper,
            "parse_llm_response": parse_llm_responses,
            "repair_json": implement_error_recovery
        },
        "configuration": {
            "default_model": "gpt-4o-mini",
            "max_retries": 3,
            "enable_repair": True
        }
    }
    
    return json_utilities


# Create default schemas
question_schema_config = create_question_schema()
response_schema_config = create_response_schema()

# Create default utilities package
json_utilities = export_json_utilities(
    question_schema_config=question_schema_config,
    response_schema_config=response_schema_config
)