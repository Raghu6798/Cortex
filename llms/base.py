import sys
import os
from loguru import logger
from dotenv import load_dotenv
from typing import Optional, Dict, Any

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq

from llama_index.core.base.llms.types import ChatMessage, MessageRole

load_dotenv()

os.makedirs("log", exist_ok=True)
logger.remove()
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(sys.stderr, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add("log/app.log", level="DEBUG", format=log_format, rotation="10 MB", retention="10 days", compression="zip", enqueue=True)


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
        """
        logger.info(f"Creating LlamaIndex LLM for provider: '{provider_id}' with model: '{model_id}'")

        # Your factory logic here is excellent and does not need to change.
        if provider_id == "groq":
            return Groq(
                model=model_id,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        elif provider_id in ["nvidia", "openai", "mistral", "cerebras", "sambanova"]:
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
            logger.warning(f"Provider '{provider_id}' not explicitly handled. Falling back to generic OpenAI client.")
            if not base_url:
                raise ValueError(f"A 'base_url' is required for the generic fallback.")
            
            return OpenAI(
                model=model_id,
                api_key=api_key,
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"The weather of {city} is sunny"

if __name__ == "__main__":
    # --- Configuration ---
    provider_id="cerebras"
    model_id="qwen-3-235b-a22b-instruct-2507"
    api_key=os.getenv("CEREBRAS_API_KEY")
    base_url="https://api.cerebras.ai/v1"
    temperature=0.7
    max_tokens=1000

    # 1. Create the LLM using your factory (this part was already correct)
    logger.info("Creating LLM with the factory...")
    llm = LlamaIndexLLMFactory.create_llm(
        provider_id, 
        model_id, 
        api_key, 
        base_url, 
        temperature, 
        max_tokens
    )
    logger.success("LLM created successfully.")

    # 2. Prepare the messages in the correct LlamaIndex format
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM, content="You are a helpful assistant."
        ),
        ChatMessage(
            role=MessageRole.USER, content="What is the weather in Beijing?"
        ),
    ]
    
    # 3. Call the correct method: .chat() instead of .invoke()
    logger.info("Sending request to the LLM using .chat()...")
    response = llm.chat(messages)
    
    # 4. Print the response content
    print("\n--- LLM Response ---")
    print(response.message.content)
    print("--------------------\n")