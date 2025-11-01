from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import json
import sys
import aiohttp
import urllib.parse
import os
import asyncio
from typing import Dict, Any
import sys
import os
from loguru import logger

# Create 'log' directory if it doesn't exist
os.makedirs("log", exist_ok=True)

# Remove default handler
logger.remove()

# Define custom log format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Console sink (colored output)
logger.add(
    sys.stderr,
    level="INFO",
    format=log_format,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# File sink (persistent logs)
logger.add(
    "log/app.log",
    level="DEBUG",  # Save detailed logs
    format=log_format,
    rotation="10 MB",  # Rotate after 10 MB
    retention="10 days",  # Keep logs for 10 days
    compression="zip",  # Compress old logs
    enqueue=True,  # Thread/process safe
)

# Example usage
logger.info("Logger initialized successfully.")
load_dotenv()

@tool
async def execute_api_call(input_params: Dict[str, Any]):
    """
    Executes an HTTP API call (GET, POST, PUT, DELETE, etc.) with the specified parameters.

    This tool makes HTTP requests to external APIs. It handles URL construction, headers,
    query parameters, path parameters, and request bodies.

    Args:
        input_params: A dictionary with the following keys:
            - api_url (str, required): The base URL for the API endpoint
            - api_method (str, optional): HTTP method (GET, POST, PUT, DELETE). Defaults to GET
            - api_headers (dict, optional): HTTP headers as key-value pairs (e.g., {"Authorization": "Bearer token"})
            - api_query_params (dict, optional): URL query parameters as key-value pairs
            - api_path_params (dict, optional): Path parameters to replace in URL (e.g., {id})
            - api_body (dict, optional): Request body for POST/PUT requests

    Example:
        {
            "api_url": "https://api.example.com/v1/models",
            "api_method": "GET",
            "api_headers": {"Authorization": "Bearer abc123"}
        }

    Returns:
        dict: JSON response from the API, or error details if the request fails
    """
    # 1. Extract essential API information from the input
    method = input_params.get("api_method") or input_params.get("method", "GET")
    method = method.upper()
    
    # Accept both 'api_url' and 'url' for backwards compatibility
    base_url = input_params.get("api_url") or input_params.get("url")
    
    path_params_def = input_params.get("api_path_params", {})
    query_params_def = input_params.get("api_query_params", {})
    
    # Accept both 'api_headers' and 'headers'
    headers_def = input_params.get("api_headers") or input_params.get("headers", {})
    
    body_def = input_params.get("api_body") or input_params.get("body", {})
    dynamic_variables = input_params.get("dynamic_variables",{})
    
    logger.info(f"{method},{base_url},{path_params_def},{query_params_def},{headers_def},{body_def},{dynamic_variables}")
    
    if not base_url:
        print("[function_handler] Error: 'api_url' not specified in input.")
        return {"error": "API URL (api_url or url) was not specified."}

    request_kwargs = {}

    try:
        # 2. Construct URL with Path and Query Parameters (Method-Agnostic)
        for param,value in path_params_def.items():
            logger.info(f"{param}---->{value}")
            if value is not None:
                base_url = base_url.replace(f"{{{param}}}", str(value))

        query_params = {}
        for query,value in query_params_def.items():
            logger.info(f"{query}---->{value}")
            if value is not None:
                query_params[query] = value
        
        final_url = base_url
        if query_params:
            final_url += "?" + urllib.parse.urlencode(query_params)
        print(final_url)

        # 3. Construct Headers (Method-Agnostic)
        headers = {}
        for header,value in headers_def.items():
            logger.info(f"{header}----->{value}")
            if value is not None:
                headers[header] = str(value)
        if headers:
            request_kwargs["headers"] = headers

        # 4. Construct Request Body (Method-Agnostic)
        # This block runs if 'api_body' is defined, regardless of the HTTP method.
        if body_def:
            payload = {}
            for prop,value in body_def.items():
                logger.info(f"{prop} ---> {value}")
                if prop:
                    payload[prop] = value
            if payload:
                logger.info(payload)
                request_kwargs["json"] = payload
                logger.info(f"[function_handler] Constructed request payload: {payload}")

    except ValueError as e:
        # Catches missing required parameter errors from resolve_value
        logger.error(f"[function_handler] Validation Error: {e}")
        return {"error": str(e)}

    # 5. Execute the HTTP Request
    logger.debug(f"[function_handler] Executing API call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            # The 'method' variable determines the type of HTTP request dynamically.
            async with session.request(method, final_url,**request_kwargs) as response:
                response_data = None
                # Gracefully handle non-JSON responses
                try:
                    response_data = await response.json()
                except (aiohttp.ContentTypeError, aiohttp.client_exceptions.ContentTypeError):
                    response_data = await response.text()

                if response.status >= 400:
                    print(f"[function_handler] API call failed with status {response.status}: {response_data}")
                    return {
                        "error": "API request failed.",
                        "status_code": response.status,
                        "details": response_data,
                    }

                logger.success(f"[function_handler] API call successful. Status: {response_data}")
                return response_data

    except aiohttp.ClientConnectorError as e:
        logger.error(f"[function_handler] Connection Error: {e}")
        return {"error": f"Could not connect to the server at {final_url}."}
    except Exception as e:
        logger.error(f"[function_handler] An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def dynamic_tool_call_injection(agent_response:str) -> ToolMessage:
    tool_call_args = json.loads(agent_response["messages"][1].tool_calls[0]["args"])
    tool_call_name = agent_response["messages"][1].tool_calls[0]["name"]
    tool_call_id = agent_response["messages"][1].tool_calls[0]["id"]
    tool_call_result = tool_call_args
    return ToolMessage(content=tool_call_result, tool_call_id=tool_call_id)

llm = ChatNVIDIA(api_key=os.getenv("NVIDIANIM_API_KEY"), model="meta/llama-3.1-8b-instruct", temperature=0.7)
agent = create_agent(
    model=llm,
    tools=[execute_api_call],
    prompt="You are a helpful assistant that can use tools to help the user.",
)
response = asyncio.run(agent.ainvoke({"messages":[{"role":"user","content":"Call the API to fetch all models from https://integrate.api.nvidia.com/v1/models , requires no headers and do a GET request"}]}))
print(response["messages"][1].tool_calls[0]["args"])