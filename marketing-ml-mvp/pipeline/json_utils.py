"""
JSON serialization utilities for handling numpy types
"""
import json
import numpy as np

def make_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def safe_json_dump(obj, file, **kwargs):
    """
    JSON dump with automatic numpy type conversion
    """
    serializable_obj = make_serializable(obj)
    json.dump(serializable_obj, file, **kwargs)