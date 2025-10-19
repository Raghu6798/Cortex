from pydantic import BaseModel,Field
from typing import List,Dict,Any

class ToolConfigSchema(BaseModel):
    user_id: str
    agent_id: str
    is_active: bool = True
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: List[Dict[str, str]] = []
    api_query_params: List[Dict[str, str]] = []
    api_path_params: List[Dict[str, str]] = []
    request_payload: Dict[str, Any] = {}
    response_payload: Dict[str, Any] = {}

class ToolCallSchema(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to call")
    tool_args: Dict[str, Any] = Field(..., description="The arguments to pass to the tool")
    tool_result: Any = Field(..., description="The result of the tool call")

class ToolCallResponse(BaseModel):
    tool_calls: List[ToolCallSchema] = Field(..., description="The tool calls to make")
