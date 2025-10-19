from typing import Optional,List,Dict,Any
from pydantic import BaseModel

class ToolConfigSchema(BaseModel):
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: Dict[str, str] = {}
    api_query_params: Dict[str, str] = {}
    api_path_params: Dict[str, str] = {}
    request_payload: str = ""


class AgentConfigSchema(BaseModel):
    provider_id:str
    api_key: str
    model_name: str = "mistral-small-2506"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 5
    system_prompt: str = "You are a helpful AI assistant."

