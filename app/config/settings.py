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

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore' 
    )
    GROQ_API_KEY:str
    CEREBRAS_API_KEY:str
    MISTRAL_API_KEY:str
    SAMBANOVA_API_KEY:str
    GOOGLE_API_KEY:str


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
    max_tokens: Optional[int] = Field(default=512, description="Maximum number of tokens to generate.")
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
        "top_p": settings.top_p, 
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


settings = Settings()
if __name__ == "__main__":
    print("--- Testing LLM-Agnostic Module ---")

    # Check if the .env file exists at the expected path
    if not DOTENV_PATH.exists():
        print(f"--- WARNING: .env file not found at {DOTENV_PATH} ---")
        print("Please ensure it exists and contains your API keys (e.g., GROQ_API_KEY=...).")
    else:
        print(f"Successfully located .env file at: {DOTENV_PATH}")

    # 1. Test with Groq configuration
    print("\n[1] Testing Groq...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_config_data = {
            "model_name": "openai/gpt-oss-20b", 
            "api_key": groq_api_key,
            "base_url": "https://api.groq.com/openai/v1"
        }
        groq_settings = AgentSettings.model_validate(groq_config_data)
        groq_llm = get_chat_model(groq_settings)
        # response = groq_llm.invoke("Hey what is up?")
        # print(response.content)
        print(f"  -> Determined Provider: {groq_settings.provider}")
        print(f"  -> Initialized Model: {type(groq_llm).__name__}")
        assert isinstance(groq_llm, ChatGroq), "Model should be ChatGroq"
    else:
        print("  -> SKIPPED: GROQ_API_KEY not found in environment/.env file.")

    print("\n[2] Testing Google Gemini...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        google_config_data = {
            "model_name": "gemini-2.5-flash", 
            "api_key": google_api_key,
        }
        google_settings = AgentSettings.model_validate(google_config_data)
        google_llm = get_chat_model(google_settings)
        response = google_llm.invoke("Hey what is up?")
        print(response.content)
        print()
        print(f"  -> Determined Provider: {google_settings.provider}")
        print(f"  -> Initialized Model: {type(google_llm).__name__}")
        assert isinstance(google_llm, ChatGoogleGenerativeAI), "Model should be ChatGoogleGenerativeAI"
    else:
        print("  -> SKIPPED: GOOGLE_API_KEY not found in environment/.env file.")


    print("\n--- Test execution finished. ---")