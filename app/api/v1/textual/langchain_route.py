import os
import sys
from uuid import uuid4
import asyncio
import httpx
from pydantic import BaseModel, Field
import traceback
from typing import Optional,AsyncIterator
from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

from loguru import logger

logger.remove()


logger.add(
    sys.stderr,  
    level="INFO", 
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True
)
log = logger


@tool
async def execute_api_call(input_params: Dict[str, Any]):
    """
    Dynamically executes any HTTP request (GET, POST, PUT, DELETE, etc.) based on a
    structured API definition.

    This function is designed to be method-agnostic. It constructs the URL, headers,
    and request body based entirely on the provided input parameters, allowing it to
    handle any type of HTTP request.

    Args:
        input_params: A dictionary containing the API call definition and runtime values
                      sourced from the LLM or system variables.

    Returns:
        A dictionary with the JSON response from the API or a detailed error message.
    """
    # 1. Extract essential API information from the input
    method = input_params.get("api_method", "GET").upper()
    base_url = input_params.get("api_url")
    path_params_def = input_params.get("api_path_params", [])
    query_params_def = input_params.get("api_query_params", [])
    headers_def = input_params.get("api_headers", [])
    body_def = input_params.get("api_body", []) # Key for generic body handling
    dynamic_variables = input_params.get("dynamic_variables", {})

    if not base_url:
        print("[function_handler] Error: 'api_url' not specified in input.")
        return {"error": "API URL (api_url) was not specified."}

    # Dictionary to hold keyword arguments for the aiohttp request
    request_kwargs = {}

    try:
        # 2. Construct URL with Path and Query Parameters (Method-Agnostic)
        for param in path_params_def:
            param_name = param["name"]
            param_value = resolve_value(param)
            if param_value is not None:
                base_url = base_url.replace(f"{{{param_name}}}", urllib.parse.quote(str(param_value)))

        query_params = {}
        for param in query_params_def:
            param_name = param["name"]
            param_value = resolve_value(param)
            if param_value is not None:
                query_params[param_name] = param_value
        
        final_url = base_url
        if query_params:
            final_url += "?" + urllib.parse.urlencode(query_params)

        # 3. Construct Headers (Method-Agnostic)
        headers = {}
        for header in headers_def:
            header_name = header["name"]
            header_value = resolve_value(header)
            if header_value is not None:
                headers[header_name] = str(header_value)
        if headers:
            request_kwargs["headers"] = headers

        # 4. Construct Request Body (Method-Agnostic)
        # This block runs if 'api_body' is defined, regardless of the HTTP method.
        if body_def:
            payload = {}
        
            if payload:
                request_kwargs["json"] = payload
                print(f"[function_handler] Constructed request payload: {payload}")

    except ValueError as e:
        # Catches missing required parameter errors from resolve_value
        print(f"[function_handler] Validation Error: {e}")
        return {"error": str(e)}

    # 5. Execute the HTTP Request
    print(f"[function_handler] Executing API call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            # The 'method' variable determines the type of HTTP request dynamically.
            async with session.request(method, final_url, **request_kwargs) as response:
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

                print(f"[function_handler] API call successful. Status: {response.status}")
                return response_data

    except aiohttp.ClientConnectorError as e:
        print(f"[function_handler] Connection Error: {e}")
        return {"error": f"Could not connect to the server at {final_url}."}
    except Exception as e:
        print(f"[function_handler] An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


tools = [execute_api_call]

router = APIRouter(prefix="/chat", tags=["chat"])
