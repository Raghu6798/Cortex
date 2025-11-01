# In a file like: app/api/v1/textual/agno_route.py

import os
import aiohttp
import urllib.parse
from typing import Dict, Any, List, Optional
from uuid import uuid4
from dotenv import load_dotenv

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput, RunEvent
from agno.tools.e2b import E2BTools

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from app.config.settings import settings
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB
from app.utils.logger import logger

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
    """
    Executes an HTTP API call with specified parameters. Use this tool to interact with any external API.

    Args:
        api_url (str): The base URL for the API endpoint, may contain placeholders like {id}.
        api_method (str): HTTP method (GET, POST, PUT, DELETE). Defaults to GET.
        api_headers (dict): HTTP headers (e.g., {"Authorization": "Bearer token"}).
        api_query_params (dict): URL query parameters.
        api_path_params (dict): Path parameters to replace in the URL.
        api_body (dict): Request body for POST/PUT requests.
    """
    method = api_method.upper()
    base_url = api_url
    path_params = api_path_params or {}
    query_params = api_query_params or {}
    headers = api_headers or {}
    body = api_body or {}

    if not base_url:
        return {"error": "API URL (api_url) was not specified."}

    request_kwargs = {}

    try:
        # Construct URL with Path and Query Parameters
        for param, value in path_params.items():
            if value is not None:
                base_url = base_url.replace(f"{{{param}}}", str(value))

        final_url = base_url
        if query_params:
            final_url += "?" + urllib.parse.urlencode({k: v for k, v in query_params.items() if v is not None})
        
        logger.info(f"Final URL: {final_url}")

        if headers:
            request_kwargs["headers"] = headers

        if body:
            request_kwargs["json"] = body
            logger.info(f"Request Body: {body}")

    except Exception as e:
        logger.error(f"Error preparing request: {e}")
        return {"error": f"Error preparing request: {str(e)}"}

    logger.debug(f"Executing API call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, final_url, **request_kwargs) as response:
                response_data = None
                try:
                    response_data = await response.json()
                except (aiohttp.ContentTypeError, aiohttp.client_exceptions.ContentTypeError):
                    response_data = await response.text()

                if response.status >= 400:
                    logger.warning(f"API call failed with status {response.status}: {response_data}")
                    return {"error": "API request failed", "status_code": response.status, "details": response_data}

                logger.success("API call successful.")
                return response_data

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Connection Error: {e}")
        return {"error": f"Could not connect to the server at {final_url}."}
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

db_url = settings.SUPABASE_DB_URI
db = PostgresDb(db_url=str(db_url))

router = APIRouter(prefix="/api/v1/agno", tags=["Agno Multi-Agent"])

@router.post("/invoke", response_model=CortexResponseFormat, tags=["Chat"])
async def invoke_agno_agent(
    request: CortexInvokeRequestSchema,
    current_user: dict = Depends(get_current_user),
    database_session: Session = Depends(get_db)
) -> CortexResponseFormat:
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = database_session.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session_id = session.id

    provider_id = request.provider_id
    model_id = request.model_id
    provider = await llm_router.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")

    api_key = request.api_key.get_secret_value() if request.api_key else provider.api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required.")

    llm = OpenAIChat(id=model_id, base_url=provider.base_url, api_key=api_key)

    agent = Agent(
        model=llm,
        db=db,
        tools=[execute_api_call], 
        enable_user_memories=True,
    )
    logger.info(f"Agent configured with model '{model_id}' and the 'execute_api_call' tool.")

    try:
      
        run_output: RunOutput = await agent.arun(
            input=request.message,
            stream=False,  
            user_id=user_id,
            session_id=session_id,
        )

        logger.info(f"Agent run completed. Response content: {run_output.content}")
        
  
        final_response_string = run_output.content

        if not isinstance(final_response_string, str):
            import json
            final_response_string = json.dumps(final_response_string, indent=2)

        return CortexResponseFormat(response=final_response_string)

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")