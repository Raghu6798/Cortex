from typing import Optional,List,Dict,Any

class ToolConfigSchema(BaseModel):
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: List[Dict[str, str]] = []
    api_query_params: List[Dict[str, str]] = []


class AgentConfigSchema(BaseModel):
    base_url: Optional[str] = Field(default="https://api.mistral.ai/v1/")
    api_key: str
    model_name: str = "mistral-small-2506"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 5
    system_prompt: str = "You are a helpful AI assistant."