import inspect
from typing import Callable, Dict, Any, List

TOOL_REGISTRY: Dict[str, Callable] = {}
TOOLS_SCHEMA: List[Dict[str, Any]] = []

def diagnostic_tool(name: str, description: str):
    """Decorator that registers a diagnostic tool and auto-generates its OpenAI function schema."""
    def decorator(func):
        # Auto-generate JSON schema from function signature + type annotations
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array"}
        
        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            param_type = type_mapping.get(annotation, "string")
            param_desc = f"Parameter: {param_name}"
            
            # Extract description from docstring if available
            properties[param_name] = {"type": param_type, "description": param_desc}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        }
        
        TOOLS_SCHEMA.append(schema)
        TOOL_REGISTRY[name] = func
        return func
    return decorator
