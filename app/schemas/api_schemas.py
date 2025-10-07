# NEW FILE: app/schemas/api_schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from app.config.settings import AgentSettings

class HealthStatus(BaseModel):
    status: str
    message: str

class ToolConfigSchema(BaseModel):
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: List[Dict[str, str]] = []
    api_query_params: List[Dict[str, str]] = []

class InvokeRequestSchema(AgentSettings):
    message: str
    tools: List[ToolConfigSchema] = []

class CortexResponseFormat(BaseModel):
    response: str