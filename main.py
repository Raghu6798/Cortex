import sys
from pathlib import Path
from uuid import uuid4
import multiprocessing
import traceback
from typing import Optional, Literal

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from loguru import logger

# LangChain imports (adjust based on your actual installed packages)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_cerebras import ChatCerebras
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Clerk Auth
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials

from app.api.v1.sessions import router as sessions_router
from app.api.v1.frameworks import router as frameworks_router


# -------------------- Logging Setup --------------------
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

# -------------------- Environment --------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

# -------------------- Clerk Auth Setup --------------------
clerk_config = ClerkConfig(
    jwks_url="https://supreme-caribou-95.clerk.accounts.dev/.well-known/jwks.json",
    auto_error=True
)
clerk_auth_guard = ClerkHTTPBearer(config=clerk_config, add_state=True)

# Optional role-based access
def admin_required(credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)):
    roles = credentials.decoded.get("roles", [])
    if "admin" not in roles:
        raise HTTPException(status_code=403, detail="Admin permission required")
    return credentials

# -------------------- FastAPI App --------------------
app = FastAPI(title="Secure Agenta ADE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Pydantic Models --------------------
Provider = Literal[
    "openai", "google", "groq", "ollama", "mistral", "together",
    "openrouter", "nvidia", "cerebras", "sambanova", "llama_cpp",
    "custom_local"
]

class AgentSettings(BaseModel):
    model_name: str
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 512
    system_prompt: str = "You are a helpful AI assistant."
    model_path: Optional[str] = None
    n_gpu_layers: int = 8
    n_batch: int = 300
    n_ctx: int = 10000
    provider: Optional[Provider] = None

class InvokeRequestSchema(AgentSettings):
    message: str

class CortexResponseFormat(BaseModel):
    response: str

# -------------------- LangChain Tools --------------------
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


tools = [search, calculate]

# -------------------- Chat Model Factory --------------------
def get_chat_model(settings: AgentSettings) -> BaseChatModel:
    provider = settings.provider or "openai"
    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    init_params = {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p,
    }
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
        return ChatLlamaCpp(
            model_path=settings.model_path,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_p=settings.top_p,
            top_k=settings.top_k,
            n_gpu_layers=settings.n_gpu_layers,
            n_batch=settings.n_batch,
            n_ctx=settings.n_ctx,
            n_threads=multiprocessing.cpu_count() - 1,
            verbose=False,
        )
    elif provider == "cerebras":
        return ChatCerebras(cerebras_api_key=api_key, **init_params)
    else:
        return ChatOpenAI(api_key=api_key, base_url=settings.base_url, **init_params)

# -------------------- API Router --------------------
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/invoke", response_model=CortexResponseFormat)
async def invoke_agent_sync(
    request: InvokeRequestSchema,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
) -> CortexResponseFormat:
    """
    Receives agent config and a message, runs the agent to completion,
    and returns a single, final response.
    """
    user_id = credentials.decoded.get("sub")
    log.info(f"Request by user: {user_id}")

    try:
        llm = get_chat_model(request)
        agent = create_agent(
            model=llm,
            tools=tools,
            prompt=request.system_prompt,
            checkpointer=InMemorySaver(),
        )

        thread_id = str(uuid4())
        response_data = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            {"configurable": {"thread_id": thread_id}}
        )

        final_output = "Agent did not return a final message."
        if response_data and "messages" in response_data and response_data["messages"]:
            last_message = response_data["messages"][-1]
            if hasattr(last_message, "content"):
                final_output = last_message.content

        log.success("Agent invocation finished successfully.")
        return CortexResponseFormat(response=final_output)

    except Exception:
        log.exception("An error occurred during agent invocation.")
        raise HTTPException(status_code=500, detail="Internal server error")

# -------------------- Include Routers --------------------
app.include_router(router)
app.include_router(sessions_router, prefix="/chat")
app.include_router(frameworks_router, prefix="/chat")

# -------------------- Root Endpoint --------------------
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Agenta ADE Backend"}
