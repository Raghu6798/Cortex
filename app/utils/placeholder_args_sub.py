from typing import Any, Dict

def _substitute_placeholders(template: Any, values: Dict[str, Any]) -> Any:
    """Recursively substitutes placeholders like {{key}} in strings, dicts, or lists."""
    if isinstance(template, str):
        for key, val in values.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(val))
        return template
    if isinstance(template, dict):
        return {k: _substitute_placeholders(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [_substitute_placeholders(i, values) for i in template]
    return template