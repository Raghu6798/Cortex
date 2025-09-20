import os 
from dotenv import load_dotenv
from pydantic import BaseModel,Field, ConfigDict, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

import multiprocessing
from typing import Literal, Optional, Dict, Any, Type
from urllib.parse import urlparse
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_cerebras import ChatCerebras
from langchain_sambanova import ChatSambaNovaCloud


load_dotenv()

Provider = Literal[
    "openai", "google", "groq", "ollama", "mistral", "together",
    "openrouter", "nvidia", "cerebras", "sambanova", "llama_cpp"
]

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

class AgentSettings(BaseSettings):
    """
    A Pydantic settings model to hold the configuration for an AI agent.
    It is designed to be LLM-agnostic and can be instantiated from environment
    variables, .env files, or directly from a dictionary (e.g., frontend JSON).
    """
    # --- Model Configuration ---
    model_name: str = Field(
        ...,
        description="The name of the model to use (e.g., 'gpt-4o-mini', 'gemini-1.5-pro')."
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="The API key for the selected LLM provider."
    )
    base_url: Optional[str] = Field(
        default=None,
        description="The base URL for the API, required for custom/OpenAI-compatible endpoints."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(
        default=512,
        description="Maximum number of tokens to generate."
    )
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to guide the agent's behavior."
    )
    
    model_path: Optional[str] = Field(
        default=None,
        description="The local path to the model file (for LlamaCpp)."
    )
    n_gpu_layers: int = Field(default=8, description="Number of GPU layers for LlamaCpp.")
    n_batch: int = Field(default=300, description="Batch size for LlamaCpp.")
    n_ctx: int = Field(default=10000, description="Context size for LlamaCpp.")

    # --- Internal Fields ---
    # This field is not set by the user but determined by the logic below.
    provider: Optional[Provider] = Field(default=None, exclude=True)

    # Use the dot-env file if available
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-utf8', extra='ignore')

    @model_validator(mode='after')
    def determine_provider(self) -> 'AgentSettings':
        """
        Dynamically determine the provider based on model_name or base_url.
        This is the core of the agnostic logic.
        """
        if self.provider:
            return self

        # Rule 1: Local LlamaCpp model path has highest priority
        if self.model_path:
            self.provider = "llama_cpp"
            return self

        # Rule 2: Google models are identified by name
        if self.model_name.lower().startswith('gemini'):
            self.provider = "google"
            return self
            
        # Rule 3: Use base_url to identify other providers
        if self.base_url:
            hostname = urlparse(self.base_url).hostname
            if hostname:
                if "groq" in hostname:
                    self.provider = "groq"
                elif "mistral" in hostname:
                    self.provider = "mistral"
                elif "together" in hostname:
                    self.provider = "together"
                elif "openrouter" in hostname:
                    self.provider = "openrouter"
                elif "nvidia" in hostname:
                    self.provider = "nvidia"
                elif "cerebras" in hostname:
                    self.provider = "cerebras"
                elif "sambanova" in hostname:
                    self.provider = "sambanova"
                elif "localhost" in hostname or "127.0.0.1" in hostname:
                    self.provider = "ollama" # Default for local
                else:
                    self.provider = "openai" # Fallback for other OpenAI-compatible APIs
            return self

        # Rule 4: Default to OpenAI if no other rules match
        self.provider = "openai"
        return self

# --- The Agnostic Factory Function ---

def get_chat_model(settings: AgentSettings) -> BaseChatModel:
    """
    Factory function that takes AgentSettings and returns an initialized
    LangChain BaseChatModel instance.

    This is the central point of the LLM-agnostic architecture.
    """
    if not settings.provider:
        raise ValueError("Provider could not be determined. Please check settings.")

    provider = settings.provider
    api_key = settings.api_key.get_secret_value() if settings.api_key else None

    # Common parameters for most models
    init_params = {
        "model": settings.model_name,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "model_kwargs": {
            "top_p": settings.top_p,
            "top_k": settings.top_k,
        }
    }
    
    # Provider-specific logic
    if provider == "google":
        os.environ["GOOGLE_API_KEY"] = api_key or os.environ.get("GOOGLE_API_KEY", "")
        return ChatGoogleGenerativeAI(**init_params)

    elif provider == "groq":
        return ChatGroq(groq_api_key=api_key, **init_params)
        
    elif provider == "ollama":
        # Ollama doesn't use an API key
        return ChatOllama(base_url=settings.base_url, **init_params)
        
    elif provider == "llama_cpp":
        if not settings.model_path:
            raise ValueError("model_path is required for LlamaCpp provider.")
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

    elif provider == "sambanova":
        return ChatSambaNova(sambanova_api_key=api_key, **init_params)
    else:
        init_params["api_key"] = api_key
        if settings.base_url:
            init_params["base_url"] = settings.base_url
        return ChatOpenAI(**init_params)


if __name__ == "__main__":
    print("--- Testing LLM-Agnostic Module ---")

    # 1. Test with Groq configuration
    print("\n[1] Testing Groq...")
    groq_config_data = {
        "model_name": "qwen/qwen3-32b",
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1"
    }
    groq_settings = AgentSettings.model_validate(groq_config_data)
    groq_llm = get_chat_model(groq_settings)
    print(f"  -> Determined Provider: {groq_settings.provider}")
    print(f"  -> Initialized Model: {type(groq_llm).__name__}")
    assert isinstance(groq_llm, ChatGroq), "Check your groq api key"



    # 3. Test with Google Gemini
    print("\n[3] Testing Google Gemini...")
    google_config_data = {
        "model_name": "gemini-2.5-flash",
        "api_key": os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_KEY"),
    }
    google_settings = AgentSettings.model_validate(google_config_data)
    google_llm = get_chat_model(google_settings)
    print(f"  -> Determined Provider: {google_settings.provider}")
    print(f"  -> Initialized Model: {type(google_llm).__name__}")
    assert isinstance(google_llm, ChatGoogleGenerativeAI)


    print("\n--- All tests passed! ---")