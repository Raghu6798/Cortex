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
from app.config.settings import AgentSettings, get_chat_model
app = FastAPI(title="Secure Agenta ADE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


base_urls = [
    "https://api.mistral.ai/v1/",
    "https://api.cerebras.ai/v1/",
    "https://openrouter.ai/api/v1",
    "https://api.groq.com/openai/v1",
    "https://api.sambanova.ai/v1",
    "https://api.together.xyz/v1",
    "http://localhost:11434/v1/",
    "http://localhost:8080/v1/",
    "https://integrate.api.nvidia.com/v1",
]


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
