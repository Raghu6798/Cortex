import asyncio
import os
import sys
from pathlib import Path
from typing import Dict,Any,List

from fastapi import APIRouter, Depends, HTTPException
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.planners import PlanReActPlanner
from google.genai.types import ThinkingConfig
from google.genai.types import GenerateContentConfig

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from app.auth.clerk_auth import get_current_user
from app.db.database import get_db       
from sqlalchemy.orm import Session
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["textual"])

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

thinking_config = ThinkingConfig(include_thoughts=True,thinking_budget=256)
planner_react = PlanReActPlanner()

capital_agent = LlmAgent(model="gemini-2.0-flash",name="capital_agent",description="Executes an HTTP request to the given URL with the given method, headers, query parameters, and body.",instruction="""You are an agent that executes an HTTP request to the given URL with the given method, headers, path or query parameters, and body.""",planner=planner_react,tools = [execute_api_call])

@router.post("/ReActAgent/adk",response_model=CortexResponseFormat,tags=["Chat"])
async def invoke_adk_react_agent(request: CortexInvokeRequestSchema,current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> CortexResponseFormat:
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_input = types.Content(role="user",parts=[types.Part(text="Call the API to fetch all models from https://api.openai.com/v1/models")])
    session_service = InMemorySessionService()
    runner = Runner(session_service=session_service)

    session_id = session_service.get_user_sessions(db=db, user_id=user_id).id
    session = session_service.create_session(pp_name="adk_agent", user_id=user_id, session_id=session_id)
    events = runner.run(user_id=user_id, session_id=session_id, new_message=user_input)
    for event in events:
        print(f"\nDEBUG EVENT: {event}\n")
        if event.is_final_response() and event.content:
            final_answer = event.content.parts[0].text.strip()
            print("\nFINAL ANSWER\n", final_answer, "\n")


