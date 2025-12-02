# C:\Users\Raghu\Downloads\Elisia_Decision_Tree\ADE\backend\app\config\settings.py

import os,sys
import multiprocessing
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

# Use pydantic's dotenv functionality directly, no need for separate load_dotenv
from pydantic import Field, SecretStr
from pydantic.networks import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama  # Import only when needed to avoid startup issues
from langchain_cerebras import ChatCerebras


from dotenv import load_dotenv 

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

load_dotenv()
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

Provider = Literal[
    "openai", "google", "groq", "ollama", "mistral", "together",
    "openrouter", "nvidia", "cerebras", "sambanova", "llama_cpp",
    "custom_local"
]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore' 
    )
    GROQ_API_KEY:str
    CEREBRAS_API_KEY:str
    MISTRAL_API_KEY:str
    LIVEKIT_API_KEY:str
    LIVEKIT_API_SECRET:str
    LIVEKIT_URL:str
    CLERK_SECRET_KEY:str
    ENCRYPTION_KEY:str
    SUPABASE_DB_URI:PostgresDsn
    SAMBANOVA_API_KEY:str
    GOOGLE_API_KEY:str
    E2B_API_KEY:str
    POSTGRES_CONNECTION_STRING:str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    LLAMACLOUD_API_KEY:str

    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = Field("cortex-production-storage", description="S3 Bucket Name")

    MINIO_ENDPOINT: str = "s3.amazonaws.com"  
    MINIO_ACCESS_KEY: str = Field(..., description="The access key for the MinIO client.")
    MINIO_SECRET_KEY: str = Field(..., description="The secret key for the MinIO client.")
    MINIO_BUCKET: str = "cortex-uploads"
    MINIO_SECURE:bool=False

class AgentSettings(BaseSettings):
    """
    A Pydantic settings model to hold the configuration for an AI agent.
    """
    # --- Model Configuration ---
    model_name: str = Field("meta-llama/llama-4-maverick-17b-128e-instruct", description="The name of the model to use.")
    api_key: Optional[SecretStr] = Field(None, description="The API key for the selected LLM provider.")
    base_url: Optional[str] = Field(None, description="The base URL for the API endpoint.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=512, description="Maximum number of tokens to generate.")
    system_prompt: str = Field(default="You are a helpful AI assistant.", description="The system prompt.")
    

    model_path: Optional[str] = Field(None, description="The local path to the model file (for LlamaCpp).")
    n_gpu_layers: int = Field(default=8, description="Number of GPU layers for LlamaCpp.")
    n_batch: int = Field(default=300, description="Batch size for LlamaCpp.")
    n_ctx: int = Field(default=10000, description="Context size for LlamaCpp.")
    provider: Optional[Provider] = Field(default=None, exclude=True)



settings = Settings()
agent_settings = AgentSettings()
