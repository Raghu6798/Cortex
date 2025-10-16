import aiohttp
import asyncio 
import requests
import json
import uuid
from typing import Dict, List, Optional, Any, AsyncIterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_

from backend.app.config.settings import settings
from backend.app.db.models import LLMProviderDB, LLMModelDB
from backend.app.db.database import get_db
from backend.app.schemas.provider_schemas import ProviderInfo, ModelInfo

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, provider_info: ProviderInfo, api_key: Optional[str] = None):
        self.provider_info = provider_info
        self.api_key = api_key
        self.base_url = provider_info.base_url
        
    @abstractmethod
    async def get_models(self) -> List[ModelInfo]:
        """Fetch available models from the provider"""
        pass
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send chat completion request"""
        pass
    
    @abstractmethod
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completion"""
        pass

class OpenAIProvider(BaseLLMProvider):
    async def get_models(self) -> List[ModelInfo]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                data = await response.json()
                return [
                    ModelInfo(
                        id=model["id"],
                        name=model["id"],
                        description=f"OpenAI {model['id']}"
                    )
                    for model in data.get("data", [])
                ]
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class GroqProvider(BaseLLMProvider):
    async def get_models(self) -> List[ModelInfo]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                data = await response.json()
                return [
                    ModelInfo(
                        id=model["id"],
                        name=model["id"],
                        context_length=model["context_window"]
                    )
                    for model in data.get("data", [])
                ]
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class MistralProvider(BaseLLMProvider):
    async def get_models(self) -> List[ModelInfo]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                data = await response.json()
                return [
                    ModelInfo(
                        id=model["id"],
                        name=model["id"],
                        context_length=model["max_context_length"],
                        description=model["description"]
                    )
                    for model in data.get("data", [])
                ]
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class CerebrasProvider(BaseLLMProvider):
    async def get_models(self) -> List[ModelInfo]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                data = await response.json()
                return [
                    ModelInfo(
                        id=model["id"],
                        name=model["id"]
                    )
                    for model in data.get("data", [])
                ]
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class NvidiaNimProvider(BaseLLMProvider):

    async def get_models(self) -> List[ModelInfo]:
        response = requests.get(f"{self.base_url}/models")
        data = response.json()
        return [
            ModelInfo(id=model["id"],
            name=model["id"])
            for model in data.get("data", [])
        ]
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class SamanovaProvider(BaseLLMProvider):
    async def get_models(self) -> List[ModelInfo]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/models") as response:
                data = await response.json()
                return [
                    ModelInfo(id=model["id"],
                    name=model["id"],
                    context_length=model["context_length"])
                    for model in data.get("data", [])
                ]

    async def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                return await response.json()
    
    
    async def stream_chat(self, messages: List[Dict[str, str]], model: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                async for line in response.content:
                    if line:
                        yield json.loads(line.decode('utf-8'))

class LLMProviderRouter:
    """Main router for managing LLM providers and models"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        openai_info = ProviderInfo(
            id="openai",
            name="openai",
            display_name="OpenAI",
            base_url="https://api.openai.com/v1",
            logo_url="/logos/openai.png",
            description="OpenAI's GPT models",
            requires_api_key=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=True
        )
        self.providers["openai"] = OpenAIProvider(openai_info, settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else None)
        
        # Groq
        groq_info = ProviderInfo(
            id="groq",
            name="groq",
            display_name="Groq",
            base_url="https://api.groq.com/openai/v1",
            logo_url="https://cdn.brandfetch.io/idxygbEPCQ/w/201/h/201/theme/dark/icon.png?c=1bxid64Mup7aczewSAYMX&t=1668515712972",
            description="Fast inference with Groq",
            requires_api_key=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=False
        )
        self.providers["groq"] = GroqProvider(groq_info, settings.GROQ_API_KEY)
        
        # Mistral
        mistral_info = ProviderInfo(
            id="mistral",
            name="mistral",
            display_name="Mistral AI",
            base_url="https://api.mistral.ai/v1",
            logo_url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfBpAyt03guidOXPIaR3o28eNlVqemSOjQEg&s",
            description="Mistral's efficient models",
            requires_api_key=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=False
        )
        self.providers["mistral"] = MistralProvider(mistral_info, settings.MISTRAL_API_KEY)
        
        # Cerebras
        cerebras_info = ProviderInfo(
            id="cerebras",
            name="cerebras",
            display_name="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            logo_url="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/cerebras-color.png",
            description="Cerebras high-performance models",
            requires_api_key=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=False
        )
        self.providers["cerebras"] = CerebrasProvider(cerebras_info, settings.CEREBRAS_API_KEY)

        nvidia_nim_info = ProviderInfo(
            id="nvidia",
            name="nvidia",
            display_name="NVIDIA NIM",
            base_url="https://integrate.api.nvidia.com/v1",
            logo_url="https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/nim-inference-microservices-1024x576.png",
            description="NVIDIA NIM models",
            requires_api_key=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=False
        )
        self.providers["nvidia"] = NvidiaNimProvider(nvidia_nim_info)

        sambanova_info = ProviderInfo(
        id="sambanova",
        name="sambanova",
        display_name="Sambanova",
        base_url="https://api.sambanova.ai/v1",
        logo_url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBdHN5aZkfTgTr3F1CeJjgLG5jVHElmPatfA&s",
        description="Sambanova models",
        requires_api_key=True,
        supports_streaming=True,
        supports_tools=True,
        supports_embeddings=False
    )
        self.providers["sambanova"] = SamanovaProvider(sambanova_info, settings.SAMBANOVA_API_KEY)

    async def get_provider(self, provider_id: str) -> Optional[BaseLLMProvider]:
        """Get a provider by ID"""
        return self.providers.get(provider_id)
    
    async def get_all_providers(self) -> List[ProviderInfo]:
        """Get all available providers"""
        return [provider.provider_info for provider in self.providers.values()]
    
    async def get_models_for_provider(self, provider_id: str) -> List[ModelInfo]:
        """Get models for a specific provider"""
        provider = await self.get_provider(provider_id)
        if provider:
            return await provider.get_models()
        return []
    
    async def chat_completion(self, provider_id: str, model: str, messages: List[Dict[str, str]],**kwargs) -> Dict[str, Any]:
        """Send chat completion request"""
        provider = await self.get_provider(provider_id)
        if not provider:
            raise ValueError(f"Provider {provider_id} not found")
        
        return await provider.chat_completion(messages, model, **kwargs)
    
    async def stream_chat(
        self, 
        provider_id: str, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completion"""
        provider = await self.get_provider(provider_id)
        if not provider:
            raise ValueError(f"Provider {provider_id} not found")
        
        async for chunk in provider.stream_chat(messages, model, **kwargs):
            yield chunk
    
    async def sync_providers_to_db(self, db: Session):
        """Sync providers and models to database"""
        for provider_id, provider in self.providers.items():
            # Check if provider exists in DB
            existing_provider = db.query(LLMProviderDB).filter(LLMProviderDB.name == provider_id).first()
            
            if not existing_provider:
                # Create new provider
                provider_db = LLMProviderDB(
                    id=str(uuid.uuid4()),
                    name=provider_id,
                    display_name=provider.provider_info.display_name,
                    base_url=provider.provider_info.base_url,
                    logo_url=provider.provider_info.logo_url,
                    description=provider.provider_info.description,
                    requires_api_key=provider.provider_info.requires_api_key,
                    supports_streaming=provider.provider_info.supports_streaming,
                    supports_tools=provider.provider_info.supports_tools,
                    supports_embeddings=provider.provider_info.supports_embeddings,
                    max_tokens=provider.provider_info.max_tokens
                )
                db.add(provider_db)
                db.flush()  # Get the ID
            else:
                provider_db = existing_provider
            
            # Sync models
            try:
                models = await provider.get_models()
                for model_info in models:
                    existing_model = db.query(LLMModelDB).filter(
                        and_(
                            LLMModelDB.provider_id == provider_db.id,
                            LLMModelDB.model_id == model_info.id
                        )
                    ).first()
                    
                    if not existing_model:
                        model_db = LLMModelDB(
                            id=str(uuid.uuid4()),
                            provider_id=provider_db.id,
                            model_id=model_info.id,
                            display_name=model_info.name,
                            description=model_info.description,
                            context_length=model_info.context_length,
                            input_cost_per_token=model_info.input_cost,
                            output_cost_per_token=model_info.output_cost
                        )
                        db.add(model_db)
            except Exception as e:
                print(f"Error syncing models for {provider_id}: {e}")
        
        db.commit()

# Global router instance
llm_router = LLMProviderRouter()
if __name__ == "__main__":
    llm_router = LLMProviderRouter()
    mistral_models = asyncio.run(llm_router.get_models_for_provider("mistral"))
    print(mistral_models[0])
    groq_models = asyncio.run(llm_router.get_models_for_provider("groq"))
    print(groq_models[0])
    nvidia_nim_models = asyncio.run(llm_router.get_models_for_provider("nvidia"))
    print(nvidia_nim_models[0])
    sambanova_models = asyncio.run(llm_router.get_models_for_provider("sambanova"))
    print(sambanova_models[0])
    cerebras_models = asyncio.run(llm_router.get_models_for_provider("cerebras"))
    print(cerebras_models[0])

