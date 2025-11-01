from typing import Any, Dict

def substitute_placeholders(template: Any, values: Dict[str, Any]) -> Any:
    """Recursively substitutes placeholders like {{key}} in strings, dicts, or lists."""
    if isinstance(template, str):
        for key, val in values.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(val))
        return template
    if isinstance(template, dict):
        return {k: substitute_placeholders(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [substitute_placeholders(i, values) for i in template]
    return template



if __name__ == "__main__":
    template = "Hello, {{name}}! How are you?"
    values = {"name": "John"}
    print(substitute_placeholders(template, values))