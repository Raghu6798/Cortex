import requests
from typing import List, Any, Optional, Dict, Sequence, Union, Type, Callable
import json
from pydantic import BaseModel, SecretStr, Field

class LlamaIndexLLMFactory:
    """A factory class to create LlamaIndex LLM instances based on provider."""

    @staticmethod
    def create_llm(
        provider_id: str,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Creates and returns a LlamaIndex LLM instance configured for the specified provider.

        Args:
            provider_id: The identifier for the LLM provider (e.g., "groq", "nvidia", "openai").
            model_id: The specific model to use.
            api_key: The API key for the provider.
            base_url: The base URL for the API endpoint (crucial for non-OpenAI providers).
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            An instance of a LlamaIndex LLM (e.g., Groq, OpenAI).
        """
        logger.info(f"Creating LlamaIndex LLM for provider: '{provider_id}' with model: '{model_id}'")

        if provider_id == "groq":
            # Groq has a dedicated, optimized class
            return Groq(
                model=model_id,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        elif provider_id in ["nvidia", "openai", "mistral", "cerebras", "sambanova"]:
            # For NVIDIA and other OpenAI-compatible APIs, we use the generic OpenAI class
            # and point it to the correct base_url. This is the key to compatibility.
            if not base_url:
                raise ValueError(f"The '{provider_id}' provider requires a 'base_url'.")
            
            return OpenAI(
                model=model_id,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            # Default fallback for any other OpenAI-compatible provider
            logger.warning(f"Provider '{provider_id}' not explicitly handled. "
                           f"Falling back to generic OpenAI-compatible client. Ensure 'base_url' is correct.")
            if not base_url:
                raise ValueError(f"The generic fallback for provider '{provider_id}' requires a 'base_url'.")
            
            return OpenAI(
                model=model_id,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"The weather of {city} is 晴天"

if __name__ == "__main__":
    provider_id="cerebras"
    model_id="qwen-3-235b-a22b-instruct-2507"
    api_key=settings.CEREBRAS_API_KEY
    base_url="https://api.cerebras.ai/v1"
    temperature=0.7
    max_tokens=1000
    llm = LlamaIndexLLMFactory.create_llm(provider_id, model_id, api_key, base_url, temperature, max_tokens)
    response = llm.invoke("What is the weather of Beijing?")
    print(response)