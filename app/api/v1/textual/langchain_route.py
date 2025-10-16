import os
import sys
from uuid import uuid4
import asyncio
import httpx
from pydantic import BaseModel, Field
import traceback
from typing import Optional,AsyncIterator,Dict,Any
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

from sqlalchemy.orm import Session

from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

from backend.app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from backend.app.utils.logger import logger
from backend.app.auth.clerk_auth import get_current_user
from backend.app.integrations.llm_router import llm_router
from backend.app.db.database import get_db
from backend.app.db.models import ChatSessionDB


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
    path_params_def = input_params.get("api_path_params", {})
    query_params_def = input_params.get("api_query_params", {})
    headers_def = input_params.get("api_headers", {})
    body_def = input_params.get("api_body", {}) 
    dynamic_variables = input_params.get("dynamic_variables",{})
    logger.info(f"{method},{base_url},{path_params_def},{query_params_def},{headers_def},{body_def},{dynamic_variables}")
    if not base_url:
        print("[function_handler] Error: 'api_url' not specified in input.")
        return {"error": "API URL (api_url) was not specified."}

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

router = APIRouter(prefix="/api/v1", tags=["textual"])

@router.post("/chat/invoke", response_model=CortexResponseFormat, tags=["Chat"])
async def invoke_react_agent(request: CortexInvokeRequestSchema, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> CortexResponseFormat:
    """
    Invokes a ReAct agent with the given request.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    session = db.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the provider configuration from the request
    provider_id = getattr(request, 'provider_id', 'openai')  # Default to openai
    model_id = getattr(request, 'model_id', request.model_name)
    
    # Get provider info from llm_router
    provider = await llm_router.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")
    
    # Initialize ChatOpenAI with provider configuration
    llm = ChatOpenAI(
        base_url=provider.provider_info.base_url,
        api_key=provider.api_key,
        model=model_id
    )
    tools = [execute_api_call]
    agent = create_agent(
        model=llm,
        tools=tools,
        prompt=request.system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    response = await agent.ainvoke({"messages": [{"role": "user", "content": request.message}]}, config={"configurable": {"session_id": session.id}})
    return CortexResponseFormat(response=response["messages"][-1].content)

 