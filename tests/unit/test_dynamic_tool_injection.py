from typing import Any, Dict
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import sys
import aiohttp
import urllib.parse
import os
import asyncio
from loguru import logger
import pytest

# ------------------------- Logging Setup -------------------------
os.makedirs("log", exist_ok=True)
logger.remove()
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.add(sys.stderr, level="INFO", format=log_format, colorize=True)
logger.add("log/app.log", level="DEBUG", format=log_format, rotation="10 MB", retention="10 days", compression="zip")
logger.info("Logger initialized successfully.")

load_dotenv()


# ------------------------- Schema Definition -------------------------
class ToolConfigSchema(BaseModel):
    name: str
    description: str
    api_url: str
    api_method: str
    api_headers: Dict[str, str] = {}
    api_query_params: Dict[str, str] = {}
    api_path_params: Dict[str, str] = {}
    dynamic_boolean: bool = False
    dynamic_variables: Dict[str, str] = {}
    request_payload: str = ""


# ------------------------- Helper Functions -------------------------
async def execute_api_call(input_params: Dict[str, Any]):
    """Executes an HTTP API call."""
    method = input_params.get("api_method", "GET").upper()
    base_url = input_params.get("api_url")
    headers = input_params.get("api_headers", {})
    query_params = input_params.get("api_query_params", {})
    path_params = input_params.get("api_path_params", {})
    body = input_params.get("api_body", {})

    # Substitute path parameters
    for k, v in path_params.items():
        base_url = base_url.replace(f"{{{k}}}", str(v))
    if query_params:
        base_url += "?" + urllib.parse.urlencode(query_params)

    async with aiohttp.ClientSession() as session:
        async with session.request(method, base_url, headers=headers, json=body) as response:
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                return await response.text()


def substitute_placeholders(template: Any, values: Dict[str, Any]) -> Any:
    """Recursively replaces {{key}} placeholders."""
    if isinstance(template, str):
        for k, v in values.items():
            template = template.replace(f"{{{{{k}}}}}", str(v))
        return template
    if isinstance(template, dict):
        return {k: substitute_placeholders(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [substitute_placeholders(i, values) for i in template]
    return template


def create_tool_function(schema: ToolConfigSchema):
    """Generates async tool function for given schema."""
    async def tool_func(input_params: Dict[str, Any] = None):
        dynamic_values = input_params or {}
        use_dynamic = schema.dynamic_boolean

        substituted_url = substitute_placeholders(schema.api_url, dynamic_values) if use_dynamic else schema.api_url
        substituted_headers = substitute_placeholders(schema.api_headers, dynamic_values)
        substituted_query_params = substitute_placeholders(schema.api_query_params, dynamic_values)
        substituted_path_params = substitute_placeholders(schema.api_path_params, dynamic_values)

        combined_params = {
            "api_url": substituted_url,
            "api_method": schema.api_method,
            "api_headers": substituted_headers,
            "api_query_params": substituted_query_params,
            "api_path_params": substituted_path_params,
        }

        return await execute_api_call(combined_params)
    return tool_func


# ------------------------- The Actual Test -------------------------
@pytest.mark.asyncio
async def test_dynamic_tool_injection_weather(monkeypatch):
    """Test dynamic placeholder injection + tool execution end-to-end."""

    nvidia_key = os.getenv("NVIDIANIM_API_KEY")
    weather_key = os.getenv("WEATHER_API_KEY")

    assert nvidia_key, "❌ NVIDIANIM_API_KEY not set"
    assert weather_key, "❌ WEATHER_API_KEY not set"

    schema = ToolConfigSchema(
        name="get_weather",
        description="Fetches weather using OpenWeather API",
        api_url="https://api.openweathermap.org/data/2.5/weather",
        api_method="GET",
        api_headers={"Content-Type": "application/json"},
        api_query_params={
            "lat": "{{lat}}",
            "lon": "{{lon}}",
            "appid": weather_key,
            "units": "metric",
            "lang": "en"
        },
        dynamic_boolean=True,
        request_payload=""
    )

    tool_func = create_tool_function(schema)
    structured_tool = StructuredTool.from_function(
        func=None,
        name=schema.name,
        description=schema.description,
        coroutine=tool_func
    )

    llm = ChatNVIDIA(api_key=nvidia_key, model="meta/llama-3.1-8b-instruct", temperature=0.7)
    agent = create_agent(
        model=llm,
        tools=[structured_tool],
        prompt="You are a helpful assistant that fetches weather using latitude and longitude."
    )

    response = await asyncio.wait_for(
        agent.ainvoke({"messages": [{"role": "user", "content": "Get weather for Hyderabad"}]}),
        timeout=60.0
    )

    # ✅ Validation
    assert response is not None
    assert isinstance(response, dict)
    assert "messages" in response or "content" in response
    logger.success("✅ Dynamic tool injection test passed successfully.")
