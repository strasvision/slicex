#!/usr/bin/env python3
"""
Validate that JSON feature files match the expected schema.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Define the expected schema for our video features
SCHEMA = {
    "type": "object",
    "properties": {
        "video_id": {"type": "string"},
        "scene_cuts": {
            "type": "array",
            "items": {
                "type": "array",
                "items": [{"type": "number"}, {"type": "number"}],
                "minItems": 2,
                "maxItems": 2
            }
        },
        "cut_intervals": {
            "type": "array",
            "items": {"type": "number"}
        },
        "avg_scene_length": {"type": "number"},
        "has_background_music": {"type": "boolean"},
        "beat_density": {"type": "number"},
        "b_roll_usage": {"type": "number", "minimum": 0, "maximum": 1},
        "transition_types": {
            "type": "array",
            "items": {"type": "string"}
        },
        "total_duration": {"type": "number", "minimum": 0},
        "num_scenes": {"type": "integer", "minimum": 1},
        "num_broll_scenes": {"type": "integer", "minimum": 0},
        "num_human_scenes": {"type": "integer", "minimum": 0},
        "style_label": {"type": "string", "enum": ["vlog", "tutorial", "showcase"]},
        "energy_level": {"type": "string", "enum": ["low", "medium", "high"]}
    },
    "required": [
        "video_id", "scene_cuts", "cut_intervals", "avg_scene_length",
        "has_background_music", "beat_density", "b_roll_usage",
        "transition_types", "total_duration", "num_scenes",
        "num_broll_scenes", "num_human_scenes", "style_label", "energy_level"
    ]
}

def validate_json_schema(data: Dict[str, Any], schema: Dict) -> List[str]:
    """Validate data against a JSON schema and return a list of errors."""
    errors = []
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types and constraints
    for field, value in data.items():
        if field not in schema["properties"]:
            continue
            
        prop_schema = schema["properties"][field]
        
        # Check type
        expected_type = prop_schema.get("type")
        if not expected_type:
            continue
            
        if expected_type == "array":
            if not isinstance(value, list):
                errors.append(f"{field}: Expected array, got {type(value).__name__}")
                continue
                
            # Validate array items
            if "items" in prop_schema:
                items_schema = prop_schema["items"]
                if isinstance(items_schema, dict) and "type" in items_schema:
                    # Variable-length array with item type
                    item_type = items_schema["type"]
                    for i, item in enumerate(value):
                        if item_type == "number" and not isinstance(item, (int, float)):
                            errors.append(f"{field}[{i}]: Expected number, got {type(item).__name__}")
                        elif item_type == "string" and not isinstance(item, str):
                            errors.append(f"{field}[{i}]: Expected string, got {type(item).__name__}")
                elif isinstance(items_schema, list):
                    # Fixed-length array schema (like for scene_cuts)
                    if len(value) != len(items_schema):
                        errors.append(f"{field}: Expected {len(items_schema)} items, got {len(value)}")
                    else:
                        for i, (item, item_schema) in enumerate(zip(value, items_schema)):
                            if not isinstance(item, list) or len(item) != 2 or \
                               not all(isinstance(x, (int, float)) for x in item):
                                errors.append(f"{field}[{i}]: Expected [number, number]")
                            
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"{field}: Expected number, got {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"{field}: Expected integer, got {type(value).__name__}")
        elif expected_type == "string" and not isinstance(value, str):
            errors.append(f"{field}: Expected string, got {type(value).__name__}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(f"{field}: Expected boolean, got {type(value).__name__}")
        
        # Check enum values if specified
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            errors.append(f"{field}: Value '{value}' not in {prop_schema['enum']}")
            
        # Check numeric constraints
        if isinstance(value, (int, float)):
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(f"{field}: Value {value} is less than minimum {prop_schema['minimum']}")
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(f"{field}: Value {value} is greater than maximum {prop_schema['maximum']}")
    
    return errors

def validate_json_file(filepath: str) -> bool:
    """Validate a single JSON file against the schema."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both single video and combined format
        if isinstance(data, dict) and "video_id" in data:
            # Single video format
            errors = validate_json_schema(data, SCHEMA)
            if errors:
                print(f"❌ Validation failed for {filepath}:")
                for error in errors:
                    print(f"  - {error}")
                return False
            return True
        elif isinstance(data, dict):
            # Combined format with video names as keys
            all_valid = True
            for video_name, video_data in data.items():
                errors = validate_json_schema(video_data, SCHEMA)
                if errors:
                    print(f"❌ Validation failed for {video_name} in {filepath}:")
                    for error in errors:
                        print(f"  - {error}")
                    all_valid = False
            return all_valid
        else:
            print(f"❌ Invalid JSON structure in {filepath}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate JSON feature files against schema')
    parser.add_argument('files', nargs='+', help='JSON files or directories to validate')
    args = parser.parse_args()
    
    # Expand directories to files
    files_to_validate = []
    for path in args.files:
        if os.path.isdir(path):
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith('.json'):
                        files_to_validate.append(os.path.join(root, filename))
        else:
            files_to_validate.append(path)
    
    if not files_to_validate:
        print("No JSON files found to validate.")
        return
    
    # Validate each file
    valid_count = 0
    for filepath in files_to_validate:
        if validate_json_file(filepath):
            print(f"✅ {filepath} is valid")
            valid_count += 1
    
    # Print summary
    print(f"\nValidation complete: {valid_count}/{len(files_to_validate)} files are valid")
    if valid_count < len(files_to_validate):
        sys.exit(1)

if __name__ == "__main__":
    main()
