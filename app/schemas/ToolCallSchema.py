from pydantic import BaseModel,Field

class ToolCallSchema(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to call")
    tool_args: Dict[str, Any] = Field(..., description="The arguments to pass to the tool")
    tool_result: Any = Field(..., description="The result of the tool call")

class ToolCallEvent(BaseModel):
    tool_calls: List[ToolCallSchema] = Field(..., description="The tool calls to make")

class ToolCallResponse(BaseModel):
    tool_calls: List[ToolCallSchema] = Field(..., description="The tool calls to make")
