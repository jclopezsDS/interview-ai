"""
JSON utilities for parsing and validation.

NOTE: Experimental functions currently used in notebooks for research and reporting.
Potential future integration into production API.

This module provides utilities for parsing, validating, and repairing
JSON responses from LLMs in the interview practice application.
"""

import re
import json
from typing import Dict, Any, List
from jsonschema import validate, ValidationError


def implement_schema_validation(
    schema_config: Dict[str, Any],
    enable_strict_mode: bool = True
    ) -> Dict[str, Any]:
    """Validate outputs against defined schemas.
    
    Args:
        schema_config: Schema configuration from create_*_schema functions
        enable_strict_mode: Enforce strict validation rules
        
    Returns:
        Dict[str, Any]: Validation system with validation functions
    """
    print(f"[JSON] Creating schema validator")
    
    schema = schema_config["schema"]
    version = schema_config.get("version", "1.0")
    
    def validate_json_output(
        data: Dict[str, Any],
        return_errors: bool = False
    ) -> Dict[str, Any]:
        """Validate JSON data against schema."""
        try:
            validate(instance=data, schema=schema)
            
            validation_result = {
                "is_valid": True,
                "errors": [],
                "validated_fields": list(data.keys()),
                "schema_version": version
            }
            
            return validation_result
            
        except ValidationError as e:
            error_details = {
                "field": ".".join(str(x) for x in e.absolute_path) if e.absolute_path else "root",
                "message": e.message,
                "invalid_value": e.instance if hasattr(e, 'instance') else None
            }
            
            validation_result = {
                "is_valid": False,
                "errors": [error_details],
                "validated_fields": [],
                "schema_version": version
            }
            
            if return_errors:
                return validation_result
            else:
                raise ValidationError(f"Schema validation failed: {error_details['message']}")
    
    def validate_required_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all required fields are present."""
        required_fields = schema.get("required", [])
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return {
                "valid": False,
                "missing_fields": missing_fields,
                "present_fields": [f for f in required_fields if f in data]
            }
        
        return {
            "valid": True,
            "missing_fields": [],
            "present_fields": required_fields
        }
    
    def batch_validate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate multiple JSON objects."""
        results = []
        valid_count = 0
        
        for i, data in enumerate(data_list):
            try:
                result = validate_json_output(data, return_errors=True)
                results.append(result)
                if result["is_valid"]:
                    valid_count += 1
            except Exception as e:
                results.append({
                    "is_valid": False,
                    "errors": [{"field": "unknown", "message": str(e)}],
                    "index": i
                })
        
        batch_result = {
            "total_items": len(data_list),
            "valid_items": valid_count,
            "invalid_items": len(data_list) - valid_count,
            "success_rate": valid_count / len(data_list) if data_list else 0,
            "results": results
        }
        
        return batch_result
    
    validator_config = {
        "schema": schema,
        "version": version,
        "strict_mode": enable_strict_mode,
        "validate_single": validate_json_output,
        "validate_required": validate_required_fields,
        "validate_batch": batch_validate,
        "schema_info": {
            "required_fields": schema.get("required", []),
            "optional_fields": [
                field for field in schema.get("properties", {}).keys() 
                if field not in schema.get("required", [])
            ],
            "total_fields": len(schema.get("properties", {}))
        }
    }
    
    print(f"[COMPLETED] Schema validator created for {len(schema.get('required', []))} required fields")
    return validator_config


def implement_error_recovery(malformed_json: str) -> Dict[str, Any]:
    """Handle and recover from malformed or incomplete outputs.
    
    Args:
        malformed_json: Malformed JSON string to repair
        
    Returns:
        Dict[str, Any]: Recovery results with repaired JSON
    """
    print(f"[REPAIR] Attempting to fix malformed JSON")
    
    fixes_applied = []
    repaired = malformed_json.strip()
    
    # Fix 1: Remove trailing commas
    if re.search(r',\s*[}\]]', repaired):
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        fixes_applied.append("trailing_commas")
    
    # Fix 2: Add missing quotes to keys
    repaired = re.sub(r'(\w+):', r'"\1":', repaired)
    fixes_applied.append("quote_keys")
    
    # Fix 3: Fix single quotes to double quotes
    if "'" in repaired:
        repaired = repaired.replace("'", '"')
        fixes_applied.append("quote_normalization")
    
    # Fix 4: Complete incomplete objects
    if repaired.count('{') > repaired.count('}'):
        repaired += '}'
        fixes_applied.append("complete_object")
    
    # Fix 5: Complete incomplete arrays
    if repaired.count('[') > repaired.count(']'):
        repaired += ']'
        fixes_applied.append("complete_array")
    
    print(f"[REPAIR] Applied fixes: {', '.join(fixes_applied)}")
    
    return {
        "success": True,
        "repaired_json": repaired,
        "fixes_applied": fixes_applied
    }


def parse_llm_responses(
    json_text: str,
    expected_schema: Dict[str, Any],
    enable_repair: bool = True
    ) -> Dict[str, Any]:
    """Parse and validate LLM responses for malformed JSON.
    
    Args:
        json_text: Raw JSON string from LLM
        expected_schema: Schema to validate against
        enable_repair: Attempt to repair malformed JSON
        
    Returns:
        Dict[str, Any]: Parsing results with validation status
    """
    print(f"[PARSE] Processing LLM response of {len(json_text)} characters")
    
    # Step 1: Try direct JSON parsing
    try:
        parsed_data = json.loads(json_text)
        
        # Step 2: Validate against schema
        try:
            validate(instance=parsed_data, schema=expected_schema)
            print(f"[SUCCESS] JSON parsed successfully")
            print(f"[VALIDATION] Schema validation passed")
            
            return {
                "success": True,
                "parsed_data": parsed_data,
                "validation_passed": True,
                "repairs_applied": []
            }
        except ValidationError as ve:
            print(f"[WARNING] Schema validation failed: {str(ve)[:100]}...")
            return {
                "success": True,
                "parsed_data": parsed_data,
                "validation_passed": False,
                "validation_error": str(ve),
                "repairs_applied": []
            }
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {str(e)}")
        
        # Step 3: Attempt repair if enabled
        if enable_repair:
            print(f"[REPAIR] Attempting JSON repair")
            repaired_result = implement_error_recovery(json_text)
            
            if repaired_result["success"]:
                print(f"[SUCCESS] JSON repaired successfully")
                # Parse repaired JSON (no recursion)
                try:
                    repaired_data = json.loads(repaired_result["repaired_json"])
                    
                    # Validate repaired data
                    try:
                        validate(instance=repaired_data, schema=expected_schema)
                        validation_passed = True
                        validation_error = None
                        print(f"[VALIDATION] Schema validation passed")
                    except ValidationError as ve:
                        validation_passed = False
                        validation_error = str(ve)
                        print(f"[WARNING] Schema validation failed: {str(ve)[:100]}...")
                    
                    return {
                        "success": True,
                        "parsed_data": repaired_data,
                        "validation_passed": validation_passed,
                        "validation_error": validation_error,
                        "repairs_applied": repaired_result["fixes_applied"]
                    }
                except json.JSONDecodeError:
                    print(f"[ERROR] Repair failed - still invalid JSON")
        
        return {
            "success": False,
            "error": "json_parse_error",
            "details": str(e),
            "raw_json": json_text[:200]
        }
