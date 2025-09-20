import os
import httpx
from pydantic import BaseModel, Field
from typing import Optional
from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

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


class InvokeRequestSchema(BaseModel):
    config: AgentConfigSchema
    message: str


class CortexResponseFormat(BaseModel):
    response: dict

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Perform calculations."""
    return str(eval(expression))


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/invoke", response_model=CortexResponseFormat)
async def invoke_agent(
    request: InvokeRequestSchema,
):
    try:
        llm = ChatOpenAI(
            base_url=request.config.base_url
            or os.environ.get("OPENAI_BASE_URL", "https://api.mistral.ai/v1/"),
            api_key=request.config.api_key,
            model=request.config.model_name,
            temperature=request.config.temperature,
            top_p=request.config.top_p,
        )

        agent = create_agent(
            model=llm, tools=[search,calculate], prompt=request.config.system_prompt
        )

        response_data = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]}
        )

        return CortexResponseFormat(response={"messages": response_data.get("messages", [])})

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


app.include_router(router)
