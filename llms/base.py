import sys
import os
import json
import requests
import httpx
from loguru import logger
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Sequence, Union, Callable,Literal

from pydantic import SecretStr

# --- LlamaIndex Imports ---
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole, LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.openai.utils import (
    is_chat_model,
    is_function_calling_model,
    openai_modelname_to_contextsize,
)

# --- LangChain Imports ---
from langchain.agents import create_agent
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import (
    ToolMessage,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_genai._function_utils import convert_to_genai_function_declarations
from google.ai.generativelanguage_v1beta.types import Tool as GoogleTool

load_dotenv()


os.makedirs("log", exist_ok=True)
logger.remove()
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(sys.stderr, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add("log/app.log", level="DEBUG", format=log_format, rotation="10 MB", retention="10 days", compression="zip", enqueue=True)

class ChatOpenAI(BaseChatModel):
    """
    A custom LangChain chat model that can connect to any OpenAI-compatible API endpoint.
    Includes provider-specific logic for handling tool binding and payload creation.
    """

    provider_id: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: SecretStr
    base_url: str

    @property
    def _llm_type(self) -> str:
        return "openai_compatible_chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, "BaseTool"]],
        *,
        tool_choice: Optional[Union[str, bool, Literal["auto", "any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        Bind tools to the model, applying provider-specific logic.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        if self.provider_id == "google":
            try:
                genai_tools = tool_to_dict(convert_to_genai_function_declarations(tools))
                kwargs["tools"] = genai_tools["function_declarations"]
            except Exception:
                kwargs["tools"] = formatted_tools
        else:
            kwargs["tools"] = formatted_tools

        if tool_choice is not None:
            if self.provider_id == "groq" and tool_choice == "any":
                kwargs["tool_choice"] = "required"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
            else:
                kwargs["tool_choice"] = tool_choice
        
        return self.bind(**kwargs)

    def _create_payload(self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Creates the JSON payload for the API request, with provider-specific adjustments."""
        tool_call_map = {tc["id"]: tc["name"] for m in messages if isinstance(m, AIMessage) and m.tool_calls for tc in m.tool_calls}

        message_dicts = []
        for m in messages:
            if isinstance(m, HumanMessage):
                message_dicts.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}} for tc in m.tool_calls]
                message_dicts.append(msg)
            elif isinstance(m, SystemMessage):
                message_dicts.append({"role": "system", "content": m.content})
            elif isinstance(m, ToolMessage):
                tool_msg = {"role": "tool", "content": m.content, "tool_call_id": m.tool_call_id}
                if self.provider_id == "cerebras":
                    tool_name = tool_call_map.get(m.tool_call_id)
                    if tool_name: tool_msg["name"] = tool_name
                    else: logger.warning(f"Could not find name for tool_call_id: {m.tool_call_id}")
                message_dicts.append(tool_msg)
        
        payload = {"model": self.model, "messages": message_dicts, "temperature": self.temperature, **kwargs}
        if self.max_tokens: payload["max_tokens"] = self.max_tokens
        if stop: payload["stop"] = stop

        if self.provider_id == "cerebras" and "tools" in payload:
            payload["parallel_tool_calls"] = False
            logger.debug("Set parallel_tool_calls=False for Cerebras provider.")
            
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous call to an OpenAI-compatible API."""
        payload = self._create_payload(messages, stop=stop, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }


        try:
            logger.info(f"The base_url being hit is {self.base_url} , payload : {payload} and {headers}")
            response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

        # 3. Parse the response into a LangChain ChatResult
        return self._parse_response(data)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous call to an OpenAI-compatible API."""
        # 1. Prepare messages and payload
        payload = self._create_payload(messages, stop=stop, **kwargs)
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        # 2. Make the async HTTP request
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0, # Add a reasonable timeout
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"Async API call failed: {e}")
            raise

        # 3. Parse the response into a LangChain ChatResult
        return self._parse_response(data)
    
    def _create_payload(self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Creates the JSON payload for the API request."""
        message_dicts = []
        for m in messages:
            if isinstance(m, HumanMessage):
                message_dicts.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                msg = {"role": "assistant", "content": m.content or ""}

                if m.tool_calls:
                
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function", 
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                        for tc in m.tool_calls
                    ]
                message_dicts.append(msg)
            elif isinstance(m, SystemMessage):
                message_dicts.append({"role": "system", "content": m.content})
            elif isinstance(m, ToolMessage):
                message_dicts.append({
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                })
        
        # Build final payload
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
            **kwargs,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop
            
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> ChatResult:
        """Parses the API JSON response into a ChatResult."""
        choice = data["choices"][0]
        message_data = choice["message"]
        
        content = message_data.get("content", "")
        
        tool_calls = []
        if message_data.get("tool_calls"):
            for tc in message_data["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {} 
                
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": args,
                })

        ai_message = AIMessage(
            content=content or "",
            tool_calls=tool_calls if tool_calls else []
        )
        
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


class OpenAICompatible(OpenAI):
    """
    A custom LlamaIndex LLM class that inherits from the official OpenAI one,
    but overrides the metadata property to avoid the hardcoded model check.
    This allows it to work with any OpenAI-compatible API like NVIDIA or Cerebras.
    """
    context_window: int = Field(
        default=32768, description="The context window of the model."
    )

    @property
    def metadata(self) -> LLMMetadata:
        """
        Override metadata to bypass the model name lookup.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,  
            is_function_calling_model=is_function_calling_model(model=self.model),
            model_name=self.model,
            system_role="system", 
        )

class LlamaIndexLLMFactory:
    """A factory class to create LlamaIndex LLM instances based on provider."""

    @staticmethod
    def create_llm(
        provider_id: str,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        context_window: int = 32768,
        **kwargs: Any,
    ) -> LLM:
        """
        Creates and returns a LlamaIndex LLM instance.
        """
        logger.info(f"Creating LlamaIndex LLM for provider: '{provider_id}' with model: '{model_id}'")

       
        
        if provider_id in ["nvidia","groq","novita","openai", "mistral", "cerebras", "sambanova", "google"]:
            if not base_url:
                raise ValueError(f"The '{provider_id}' provider requires a 'base_url'.")
            
            return OpenAICompatible(
                model=model_id,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window, # Pass the context window here
                **kwargs
            )
        else:
            logger.warning(f"Provider '{provider_id}' not explicitly handled. Falling back to generic client.")
            if not base_url:
                raise ValueError(f"A 'base_url' is required for the generic fallback.")
            
            return OpenAICompatible(
                model=model_id,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window, 
                **kwargs
            )

class LangChainLLMFactory:
    """A factory class to create LangChain BaseChatModel instances based on provider."""

    @staticmethod
    def create_llm(
        provider_id: str,
        model_id: str,
        api_key: str,
        base_url: str, # Base URL is now essential
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Creates and returns a generic, OpenAI-compatible LangChain BaseChatModel.
        """
        logger.info(f"Creating generic LangChain LLM for provider: '{provider_id}' with model: '{model_id}'")
        
        # Always use our universal compatible class
        return ChatOpenAI(
            provider_id=provider_id,
            model=model_id,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

@tool 
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"The weather of {city} is sunny"

@tool
def get_news(topic: str) -> str:
    """Get the news of a topic"""
    return f"The news of {topic} is that the stock market is up 100 points"


if __name__ == "__main__":
   
    provider_id="groq"
    model_id="llama-3.3-70b-versatile"
    api_key=os.getenv("GROQ_API_KEY")
    base_url="https://api.groq.com/openai/v1"
    
    temperature=0.7
    max_tokens=1000
    context_window=32768

    # logger.info("Creating LLM with the factory...")
    # llm = LlamaIndexLLMFactory.create_llm(
    #     provider_id, 
    #     model_id, 
    #     api_key, 
    #     base_url, 
    #     temperature, 
    #     max_tokens,
    #     context_window=context_window
    # )
    # logger.success("LLM created successfully.")

    # # 2. Prepare the messages
    # messages = [
    #     ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    #     ChatMessage(role=MessageRole.USER, content="Give a brief descrption about yourself"),
    # ]
    
    # logger.info("Sending request to the LLM using .chat()...")
    # response = llm.chat(messages)
    
    # print("\n--- LLM Response ---")
    # print(response.message.content)
    # print("--------------------\n")

    llm = LangChainLLMFactory.create_llm(
        provider_id,
        model_id,
        api_key,
        base_url,
        temperature,
        max_tokens,
    )
    from langgraph.checkpoint.memory import InMemorySaver
    
    agent = create_agent(
        model=llm,
        tools=[get_weather, get_news],
        prompt="You are a helpful assistant that can get the weather of a city."
    )
    
    config = {"recursion_limit": 50}
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo , use the get_weather tool to get the weather of the city and use the get_news tool to get the news of the topic")]},
        config
    )
    
    if response.get("messages"):
        final_message = response["messages"][-1]
        if hasattr(final_message, "content"):
            print(final_message.content)
        else:
            print(response)
    else:
        print(response)
