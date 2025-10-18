from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

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

class ProviderResponse(BaseModel):
    id: str
    name: str
    display_name: str
    base_url: str
    logo_url: str | None
    description: str | None
    requires_api_key: bool
    supports_streaming: bool
    supports_tools: bool
    supports_embeddings: bool
    max_tokens: int
    models: List[Dict[str, Any]] = []

class ModelResponse(BaseModel):
    id: str
    model_id: str
    display_name: str
    description: str | None
    context_length: int