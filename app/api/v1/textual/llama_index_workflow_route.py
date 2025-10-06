# from pathlib import Path

# from workflows.events import StartEvent
# from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
# from pathlib import Path

# from workflows.events import StartEvent
# from pathlib import Path
# from typing import AsyncIterator, Any, List

# from fastapi import APIRouter, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel

# from llama_index.core.agent.workflow import ReActAgent,CodeActAgent
# from llama_index.core.llms.llm import LLM
# from llama_index.core.workflow import Context
# from llama_index.llms.groq import Groq

# from app.schemas.llamaindex_schema import PrepEvent, InputEvent, StreamEvent, ToolCallEvent
# from app.config.settings import Settings


# settings = Settings()

# llm = Groq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)


# def add(x: int, y: int) -> int:
#     """Useful function to add two numbers."""
#     return x + y

# def multiply(x: int, y: int) -> int:
#     """Useful function to multiply two numbers."""
#     return x * y

# handler = llm.complete("Hello")
# print(handler)