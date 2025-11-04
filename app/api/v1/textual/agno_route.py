# # app/api/v1/textual/agno_route.py

import os
import aiohttp
import urllib.parse
import json
from uuid import uuid4
import requests

from typing import Any, Callable, Dict, Optional

from agno.tools import tool
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from loguru import logger
from dotenv import load_dotenv
import sys
import asyncio

from app.utils.placeholder_args_sub import substitute_placeholders,remove_unresolved_placeholders
from app.utils.logger import logger
from app.config.config import settings
from app.auth.clerk_auth import get_current_user
from app.db.database import get_db
from app.db.models import ChatSessionDB, ChatMetricsDB

@tool(stop_after_tool_call=True)
async def execute_api_call(
    api_url: str,
    api_method: str = "GET",
    api_headers: Optional[Dict[str, str]] = None,
    api_query_params: Optional[Dict[str, str]] = None,
    api_path_params: Optional[Dict[str, str]] = None,
    api_body: Optional[Dict[str, Any]] = None,
    dynamic_boolean: bool = False,
    dynamic_variables: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Executes an HTTP API call (GET, POST, PUT, DELETE, etc.) with the specified parameters.

    This tool makes HTTP requests to external APIs. It handles URL construction, headers,
    query parameters, path parameters, and request bodies. Supports dynamic variable injection
    using {{placeholder}} syntax when dynamic_boolean is True.

    Args:
        api_url (str, required): The base URL for the API endpoint
        api_method (str, optional): HTTP method (GET, POST, PUT, DELETE). Defaults to GET
        api_headers (dict, optional): HTTP headers as key-value pairs (e.g., {"Authorization": "Bearer token"})
        api_query_params (dict, optional): URL query parameters as key-value pairs
        api_path_params (dict, optional): Path parameters to replace in URL (e.g., {id})
        api_body (dict, optional): Request body for POST/PUT requests
        dynamic_boolean (bool, optional): Enable dynamic variable injection. Defaults to False
        dynamic_variables (dict, optional): Variables to substitute in placeholders when dynamic_boolean is True

    Example:
        execute_api_call(
            api_url="https://api.example.com/v1/models",
            api_method="GET",
            api_headers={"Authorization": "Bearer abc123"}
        )

    Returns:
        dict: JSON response from the API, or error details if the request fails

    But you need to structure the final response in a way that is easy to understand for the user about the response data.
    """
    method = api_method.upper() if api_method else "GET"

    dynamic_values = dynamic_variables or {}

    use_dynamic_injection = dynamic_boolean

    logger.debug(f"Tool 'execute_api_call' called with dynamic_injection={use_dynamic_injection}")

    if use_dynamic_injection:
        substituted_url = substitute_placeholders(api_url, dynamic_values) if api_url else None
        substituted_headers = substitute_placeholders(api_headers, dynamic_values) if api_headers else {}
        substituted_query_params = substitute_placeholders(api_query_params, dynamic_values) if api_query_params else {}
        substituted_path_params = substitute_placeholders(api_path_params, dynamic_values) if api_path_params else {}
        substituted_body = None
        if api_body:
            substituted_body = substitute_placeholders(api_body, dynamic_values)
    
        substituted_query_params = remove_unresolved_placeholders(substituted_query_params)
        substituted_path_params = remove_unresolved_placeholders(substituted_path_params)
        substituted_headers = remove_unresolved_placeholders(substituted_headers)
        
     
        base_url = substituted_url
        headers_def = substituted_headers.copy() if substituted_headers else {}
        query_params_def = substituted_query_params.copy() if substituted_query_params else {}
        path_params_def = substituted_path_params.copy() if substituted_path_params else {}
        body_def = substituted_body
    else:
        base_url = api_url
        headers_def = api_headers.copy() if api_headers else {}
        query_params_def = api_query_params.copy() if api_query_params else {}
        path_params_def = api_path_params.copy() if api_path_params else {}
        body_def = api_body

    logger.info(f"{method},{base_url},{path_params_def},{query_params_def},{headers_def},{body_def}")

    if not base_url:
        logger.error("[function_handler] Error: 'api_url' not specified in input.")
        return {"error": "API URL was not specified."}

    request_kwargs = {}

    try:
        for param, value in path_params_def.items():
            logger.info(f"{param}---->{value}")
            if value is not None:
                base_url = base_url.replace(f"{{{param}}}", str(value))

        query_params_final = {}
        for query, value in query_params_def.items():
            logger.info(f"{query}---->{value}")
            if value is not None:
                query_params_final[query] = str(value)
        
        final_url = base_url
        if query_params_final:
            final_url += "?" + urllib.parse.urlencode(query_params_final)
        logger.info(f"Final URL: {final_url}")

        headers_final = {}
        for header, value in headers_def.items():
            logger.info(f"{header}----->{value}")
            if value is not None:
                headers_final[header] = str(value)
        if headers_final:
            request_kwargs["headers"] = headers_final

        if body_def:
            payload = {}
            for prop, value in body_def.items():
                logger.info(f"{prop} ---> {value}")
                if prop:
                    payload[prop] = value
            if payload:
                logger.info(f"Request payload: {payload}")
                request_kwargs["json"] = payload
                logger.info(f"[function_handler] Constructed request payload: {payload}")

    except ValueError as e:
        logger.error(f"[function_handler] Validation Error: {e}")
        return {"error": str(e)}


    logger.debug(f"[function_handler] Executing API call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, final_url, **request_kwargs) as response:
                response_data = None
                
                try:
                    response_data = await response.json()
                except (aiohttp.ContentTypeError, aiohttp.client_exceptions.ContentTypeError):
                    response_data = await response.text()

                if response.status >= 400:
                    logger.error(f"[function_handler] API call failed with status {response.status}: {response_data}")
                    return {
                        "error": "API request failed.",
                        "status_code": response.status,
                        "details": response_data,
                    }

                logger.success(f"[function_handler] API call successful. Status: {response.status}")
                return response_data

    except aiohttp.ClientConnectorError as e:
        logger.error(f"[function_handler] Connection Error: {e}")
        return {"error": f"Could not connect to the server at {final_url}."}
    except Exception as e:
        logger.error(f"[function_handler] An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

correct_role_map = {
    "system": "system", "user": "user", "assistant": "assistant",
    "tool": "tool", "model": "assistant",
}

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
    base_url = request.base_url
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")

    api_key = request.api_key.get_secret_value() if request.api_key else provider.api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
        
    llm = OpenAIChat(
    provider=provider,
    id=model_id,
    base_url=base_url,
    api_key=api_key,
    role_map=correct_role_map
)
    if request.tools:
        tools = [execute_api_call(**tool) for tool in request.tools]
    else:
        tools = []

    tool_agent = Agent(
    model=llm,
    tools=[execute_api_call],
    markdown=True
)

    summarizer_agent = Agent(
    model=llm,
    tools=[], 
    markdown=True
)  
    
    tool_run_response = await tool_agent.arun(request.message)

    if tool_run_response.tools and tool_run_response.tools[0].result:
        api_result = tool_run_response.tools[0].result
        
        print("\n--- Intermediate: Tool call was successful. Raw data received. ---")
        
        summarizer_prompt = (
            f"Here is the data that was fetched:\n\n"
            f"```json\n{json.dumps(api_result, indent=2)}\n```\n\n"
            f"Now, please answer my original question: '{original_prompt}'"
        )
        
        print("\n--- Step 2: Running Summarizer Agent to generate final response ---")
        final_response = await summarizer_agent.arun(summarizer_prompt)
        return final_response.content
    else:
        logger.error("Agent failed invocation ")
       