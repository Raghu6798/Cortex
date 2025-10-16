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

class CortexInvokeRequestSchema(AgentSettings):
    message: str
    tools: List[ToolConfigSchema] = []
    provider_id: Optional[str] = "openai"  # Provider ID from dropdown
    model_id: Optional[str] = None  # Model ID from dropdown

class CortexResponseFormat(BaseModel):
    response: str