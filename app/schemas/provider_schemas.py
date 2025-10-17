from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class ProviderType(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    MISTRAL = "mistral"
    CEREBRAS = "cerebras"
    SAMBANOVA = "sambanova"
    TOGETHER = "together"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"

@dataclass
class ModelInfo:
    id: str
    name: str
    description: Optional[str] = None
    context_length: int = 4096

@dataclass
class ProviderInfo:
    id: str
    name: str
    display_name: str
    base_url: str
    logo_url: Optional[str] = None
    description: Optional[str] = None
    requires_api_key: bool = True
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_embeddings: bool = False
    max_tokens: int = 4096
    models: List[ModelInfo] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []
