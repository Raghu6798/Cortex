import requests
from typing import List, Any, Optional, Dict, Sequence, Union, Type, Callable
import json
from pydantic import BaseModel, SecretStr, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.config.settings import settings


class ChatBaseOpenAI(BaseChatModel):
    """LangChain wrapper for a local OpenAI-compatible server with full tool support."""

    base_url: str
    model: str
    temperature: float = 0.7
    api_key: SecretStr

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "Qwen2.5-7B-Instruct-with-Tools" 

    def bind_tools(self, tools: Sequence[Union[Dict, Type[BaseModel], Callable, BaseTool]], **kwargs: Any) -> Runnable:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # Remove response_format to prevent Cerebras API errors
        if "response_format" in kwargs:
            del kwargs["response_format"]
        return self.bind(tools=formatted_tools, **kwargs)

    def _send_request(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        processed_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                processed_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                processed_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                        for tc in m.tool_calls
                    ]
                processed_messages.append(msg_dict)
            elif isinstance(m, ToolMessage):
                processed_messages.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")

        # Filter out problematic parameters for Cerebras
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["response_format", "structured_output"]}
        payload = {"model": self.model, "messages": processed_messages, "temperature": self.temperature, **filtered_kwargs}
        
        # Handle tools for Cerebras - they support tools but with specific requirements
        if "tools" in payload and payload["tools"]:
            # Cerebras requires parallel_tool_calls=False for some models
            payload["parallel_tool_calls"] = False
            # Ensure tools have the correct format for Cerebras
            for tool in payload["tools"]:
                if "function" in tool and "strict" not in tool["function"]:
                    tool["function"]["strict"] = True
        
        # Remove response_format parameter that causes issues with Cerebras
        if "response_format" in payload:
            del payload["response_format"]
        
        # Also remove any nested response_format in kwargs
        if "response_format" in kwargs:
            del kwargs["response_format"]
        
        print(f"Sending request to {self.base_url}/chat/completions")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"All payload keys: {list(payload.keys())}")
        
        # Check for any hidden response_format
        if "response_format" in str(payload):
            print("WARNING: response_format found in payload!")
        
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers={
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        })
        
        if response.status_code != 200:
            print(f"Error response: {response.status_code} - {response.text}")
        
        response.raise_for_status()
        return response.json()

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        resp_json = self._send_request(messages, **kwargs)
        message_data = resp_json["choices"][0]["message"]
        content = message_data.get("content", "")
        
        # Handle tool calls for Cerebras
        tool_calls = []
        if "tool_calls" in message_data and message_data["tool_calls"]:
            for tc in message_data["tool_calls"]:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                    "type": "function"
                })
        
        # Create AIMessage with tool calls if present
        ai_message = AIMessage(content=content, tool_calls=tool_calls if tool_calls else None)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"The weather of {city} is 晴天"

if __name__ == "__main__":
    llm = ChatBaseOpenAI(
        base_url="https://api.cerebras.ai/v1",
        model="qwen-3-235b-a22b-instruct-2507",
        temperature=0.7,
        api_key=SecretStr(settings.CEREBRAS_API_KEY),
    )
    response = llm.bind_tools([get_weather]).invoke("What is the weather of Beijing?")
    result = response.content
    print(result)