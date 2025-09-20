from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import BaseChatModel


class ChatBaseOpenAI(BaseChatModel):
    """LangChain wrapper for a local OpenAI-compatible server with full tool support."""
    base_url: str = "http://127.0.0.1:8080"
    model_name: str = "Qwen2.5-7B-Instruct"
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str: 
        return "Qwen2.5-7B-Instruct-with-Tools"
        
    def _send_request(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        processed_messages = []
        for m in messages:
            if isinstance(m, SystemMessage): processed_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage): processed_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg_dict["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m.tool_calls]
                processed_messages.append(msg_dict)
            elif isinstance(m, ToolMessage): processed_messages.append({"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id})
            else: raise TypeError(f"Unsupported message type: {type(m)}")
        payload = {"model": self.model_name, "messages": processed_messages, "temperature": self.temperature, **kwargs}
        if "tools" in payload and payload["tools"] is None: del payload["tools"]
        response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        resp_json = self._send_request(messages, **kwargs)
        message_data = resp_json["choices"][0]["message"]
        content = message_data.get("content", "") or ""
        tool_calls = parse_tool_calls(message_data["tool_calls"]) if message_data.get("tool_calls") else []
        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(self, tools: Sequence[Union[Dict, Type[BaseModel], Callable, BaseTool]], **kwargs: Any) -> Runnable:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)
    
    def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        raise NotImplementedError("Async generation not implemented for local model.")

