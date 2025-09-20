import os
import sys
import httpx
from pydantic import BaseModel, Field
import traceback
from typing import Optional,AsyncIterator
from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


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
@router.post("/invoke")
async def invoke_agent_streaming(request: InvokeRequestSchema):
    """
    Streams the agent's response, now robustly handling both direct answers
    and tool-based outputs from the LangChain v1.0 agent.
    """
    log.info(f"Received streaming request for model '{request.model_name}'")
    
    async def stream_generator() -> AsyncIterator[str]:
        # Keep track of the last content streamed to avoid sending duplicates
        last_streamed_content = ""
        # Keep track if we have streamed any content at all
        has_streamed_content = False
        try:
            agent_settings = request
            llm = get_chat_model(agent_settings)
            
            agent = create_agent(
                model=llm, 
                tools=tools, 
                prompt=request.system_prompt
            )

            log.info(f"Invoking agent with input: '{request.message}'")

            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": request.message}]}
            ):
                log.debug(f"Agent Stream Chunk: {chunk}")


                if "messages" in chunk and chunk["messages"]:
                    latest_message = chunk["messages"][-1]
                    if latest_message.type == "ai" and latest_message.content:
                        new_content = latest_message.content[len(last_streamed_content):]
                        if new_content:
                            has_streamed_content = True
                            yield new_content
                            last_streamed_content = latest_message.content
                if "output" in chunk and chunk["output"]:
                    final_output = chunk["output"]
                    if final_output != last_streamed_content:
                
                        new_content = final_output[len(last_streamed_content):]
                        if new_content:
                            has_streamed_content = True
                            yield new_content

            if has_streamed_content:
                log.success("Agent stream finished successfully.")
            else:
                log.warning("Agent stream finished, but no streamable content or final output was found.")
                yield "Agent did not produce a streamable response."

        except Exception:
            log.exception("An error occurred during the agent stream.")
            yield f"An error occurred on the server."

    return StreamingResponse(stream_generator(), media_type="text/plain")

app.include_router(router)