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
    api_headers: Dict[str, str] = {}
    api_query_params: Dict[str, str] = {}
    api_path_params: Dict[str, str] = {}
    dynamic_boolean: bool = False
    dynamic_variables: Dict[str, str] = {}
    request_payload: str = ""

class CortexInvokeRequestSchema(AgentSettings):
    message: str
    base_url: Optional[str] = None
    tools: List[ToolConfigSchema] = []
    provider_id: Optional[str] = "openai"  # Provider ID from dropdown
    model_id: Optional[str] = None  # Model ID from dropdown
    agent_type: Optional[str] = "general"  # Agent type: general, coding, data_analysis, etc.

class CortexResponseFormat(BaseModel):
    response: str