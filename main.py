import os
import sys
from uuid import uuid4
import httpx
from pydantic import BaseModel, Field
import traceback
from typing import Optional,AsyncIterator
from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

from loguru import logger

logger.remove()


logger.add(
    sys.stderr,  
    level="INFO", 
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True
)
log = logger
app = FastAPI(title="Secure Agenta ADE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# C:\Users\Raghu\Downloads\Elisia_Decision_Tree\ADE\backend\app\config\settings.py

import os,sys
import multiprocessing
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

# Use pydantic's dotenv functionality directly, no need for separate load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_cerebras import ChatCerebras


from dotenv import load_dotenv 

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

load_dotenv()
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

base_urls = [
    "https://api.mistral.ai/v1/",
    "https://api.cerebras.ai/v1/",
    "https://openrouter.ai/api/v1",
    "https://api.groq.com/openai/v1",
    # "https://api.sambanova.ai/v1",
    "https://api.together.xyz/v1",
    "http://localhost:11434/v1/",
    "http://localhost:8080/v1/",
    "https://integrate.api.nvidia.com/v1",
]

Provider = Literal[
    "openai", "google", "groq", "ollama", "mistral", "together",
    "openrouter", "nvidia", "cerebras", "sambanova", "llama_cpp",
    "custom_local"
]

class AgentSettings(BaseSettings):
    """
    A Pydantic settings model to hold the configuration for an AI agent.
    """
    # --- Model Configuration ---
    model_name: str = Field(..., description="The name of the model to use.")
    api_key: Optional[SecretStr] = Field(None, description="The API key for the selected LLM provider.")
    base_url: Optional[str] = Field(None, description="The base URL for the API endpoint.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=1024, description="Maximum number of tokens to generate.")
    system_prompt: str = Field(default="You are a helpful AI assistant.", description="The system prompt.")
    

    model_path: Optional[str] = Field(None, description="The local path to the model file (for LlamaCpp).")
    n_gpu_layers: int = Field(default=8, description="Number of GPU layers for LlamaCpp.")
    n_batch: int = Field(default=300, description="Batch size for LlamaCpp.")
    n_ctx: int = Field(default=10000, description="Context size for LlamaCpp.")

    provider: Optional[Provider] = Field(default=None, exclude=True)


    model_config = SettingsConfigDict(env_file=DOTENV_PATH, env_file_encoding='utf-8', extra='ignore')

    @model_validator(mode='after')
    def determine_provider(self) -> 'AgentSettings':
        if self.provider: return self
        if self.model_path: self.provider = "llama_cpp"; return self
        if self.model_name.lower().startswith('gemini'): self.provider = "google"; return self
        if self.base_url:
            hostname = urlparse(self.base_url).hostname
            if hostname:
                if "groq" in hostname: self.provider = "groq"; return self
                if "mistral" in hostname: self.provider = "mistral"; return self
                if "together" in hostname: self.provider = "together"; return self
                if "openrouter" in hostname: self.provider = "openrouter"; return self
                if "nvidia" in hostname: self.provider = "nvidia"; return self
                if "cerebras" in hostname: self.provider = "cerebras"; return self
                if "localhost" in hostname or "127.0.0.1" in hostname: self.provider = "ollama"; return self
        self.provider = "openai"
        return self

def get_chat_model(settings: AgentSettings) -> BaseChatModel:
    """Factory function to get an initialized LangChain ChatModel instance."""
    if not settings.provider:
        raise ValueError("Provider could not be determined. Please check settings.")

    provider = settings.provider
    api_key = settings.api_key.get_secret_value() if settings.api_key else None

    init_params = {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p
    }
    
    if provider == "google":
        if provider == "google":
            return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=settings.model_name,
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            top_p=settings.top_p,
            top_k=settings.top_k
        )
    elif provider == "groq":
        return ChatGroq(groq_api_key=api_key, **init_params)
    elif provider == "ollama":
        return ChatOllama(base_url=settings.base_url, **init_params)
    elif provider == "llama_cpp":
        if not settings.model_path: raise ValueError("model_path is required for LlamaCpp provider.")
        return ChatLlamaCpp(
            model_path=settings.model_path, temperature=settings.temperature, max_tokens=settings.max_tokens,
            top_p=settings.top_p, top_k=settings.top_k, n_gpu_layers=settings.n_gpu_layers,
            n_batch=settings.n_batch, n_ctx=settings.n_ctx, n_threads=multiprocessing.cpu_count() - 1,
            verbose=False,
        )
    elif provider == "cerebras":
        return ChatCerebras(cerebras_api_key=api_key, **init_params)
    else:  
        return ChatOpenAI(api_key=api_key, base_url=settings.base_url, **init_params)


class AgentConfigSchema(BaseModel):
    base_url: Optional[str] = Field(default="https://api.mistral.ai/v1/")
    api_key: str
    model_name: str = "mistral-small-2506"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 5
    system_prompt: str = "You are a helpful AI assistant."


class InvokeRequestSchema(AgentSettings):
    message: str


class CortexResponseFormat(BaseModel):
    response: str

@tool
def search(query: str) -> str:
    """Search for information on a given topic."""
    log.info(f"Tool 'search' called with query: '{query}'")
    return f"You searched for: {query}. The answer is always 42."

@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation from an expression string."""
    log.info(f"Tool 'calculate' called with expression: '{expression}'")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        log.success(f"Calculation successful: {expression} = {result}")
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        log.error(f"Failed to evaluate expression '{expression}': {e}")
        return f"Failed to evaluate expression. Error: {e}"

tools = [search,calculate]

router = APIRouter(prefix="/chat", tags=["chat"])
@router.post("/invoke", response_model=CortexResponseFormat)
async def invoke_agent_sync(request: InvokeRequestSchema)->CortexResponseFormat:
    """
    Receives agent config and a message, runs the agent to completion,
    and returns a single, final response.
    """
    log.info(f"Received non-streaming request for model '{request.model_name}'")
    try:
        agent_settings = request
        llm = get_chat_model(agent_settings)
        
        agent = create_agent(
            model=llm, 
            tools=tools, 
            prompt=request.system_prompt,
            checkpointer=InMemorySaver(),
        )

        log.info(f"Invoking agent with input: '{request.message}'")
        thread_id = str(uuid4())
        response_data = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]}, {"configurable": {"thread_id": thread_id}}
        )
        
        log.debug(f"Full agent response object: {response_data}")

        final_output = "Agent did not return a final message."
        if response_data and "messages" in response_data and response_data["messages"]:
            last_message = response_data["messages"][-1]
            if hasattr(last_message,"content"):
                final_output = last_message.content

        log.success("Agent invocation finished successfully.")
        return CortexResponseFormat(response=final_output)

    except Exception:
        log.exception("An error occurred during agent invocation.")
        raise HTTPException(status_code=500, detail={"error": "An error occurred on the server."})

app.include_router(router)




