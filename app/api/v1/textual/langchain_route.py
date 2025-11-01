# app/api/v1/textual/langchain_route.py

import os
import sys
import urllib.parse
import json
import re
from uuid import uuid4
import asyncio
import re
import aiohttp
import httpx
from pydantic import BaseModel, Field, SecretStr
import traceback
from typing import Optional, AsyncIterator, Dict, Any, List, Callable

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_sambanova import ChatSambaNovaCloud

from langchain_core.tools import tool,StructuredTool
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat, ToolConfigSchema
from app.utils.logger import logger
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB, ChatMetricsDB

from app.utils.placeholder_args_sub import substitute_placeholders

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

router = APIRouter(prefix="/api/v1", tags=["textual"])

def create_tool_function(schema: ToolConfigSchema) -> Callable:
    """Creates a unique function for a tool schema to be used by FunctionTool.
    
    This function integrates placeholder substitution so that {{ }} placeholders in the
    tool schema (api_url, api_headers, api_query_params, api_path_params, request_payload)
    are replaced with values from the agent's tool call arguments during execution.
    """
    async def tool_func(input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        # Extract dynamic values from agent's tool call arguments
        dynamic_values = input_params or {}
        
        # Check if dynamic variable injection is enabled
        use_dynamic_injection = getattr(schema, 'dynamic_boolean', False)
        
        logger.debug(f"Tool '{schema.name}' called with input_params: {dynamic_values}, dynamic_injection={use_dynamic_injection}")
        
        if use_dynamic_injection:
            # Apply placeholder substitution to all schema fields
            substituted_url = substitute_placeholders(schema.api_url, dynamic_values)
            substituted_headers = substitute_placeholders(schema.api_headers, dynamic_values)
            substituted_query_params = substitute_placeholders(schema.api_query_params, dynamic_values)
            substituted_path_params = substitute_placeholders(schema.api_path_params, dynamic_values)
            substituted_body = None
            if schema.request_payload:
                try:
                    parsed_payload = json.loads(schema.request_payload)
                    substituted_body = substitute_placeholders(parsed_payload, dynamic_values)
                except (json.JSONDecodeError, ValueError):
                    substituted_body = substitute_placeholders(schema.request_payload, dynamic_values)
            
            # Filter out any placeholders that weren't substituted (still contain {{}})
            # This prevents sending malformed requests with unresolved placeholders
            def remove_unresolved_placeholders(params_dict: Dict[str, Any]) -> Dict[str, Any]:
                """Remove any values that still contain {{placeholder}} patterns."""
                cleaned = {}
                for k, v in params_dict.items():
                    if isinstance(v, str) and '{{' in v and '}}' in v:
                        logger.warning(f"Skipping unresolved placeholder in {k}: {v}")
                        continue
                    cleaned[k] = v
                return cleaned
            
            substituted_query_params = remove_unresolved_placeholders(substituted_query_params)
            substituted_path_params = remove_unresolved_placeholders(substituted_path_params)
            substituted_headers = remove_unresolved_placeholders(substituted_headers)
            
        else:
            substituted_url = schema.api_url
            substituted_headers = schema.api_headers.copy() if schema.api_headers else {}
            substituted_query_params = schema.api_query_params.copy() if schema.api_query_params else {}
            substituted_path_params = schema.api_path_params.copy() if schema.api_path_params else {}
            substituted_body = None
            if schema.request_payload:
                try:
                    substituted_body = json.loads(schema.request_payload)
                except (json.JSONDecodeError, ValueError):
                    substituted_body = schema.request_payload

        # Build combined params, prioritizing substituted values
        # Only merge agent-provided api_headers/query_params/path_params if they don't conflict
        combined_params = {
            "api_url": substituted_url,
            "api_method": schema.api_method,
        }
        
        # Merge headers: substituted first, then agent-provided (agent can override)
        final_headers = substituted_headers.copy() if substituted_headers else {}
        agent_headers = dynamic_values.get("api_headers", {}) or {}
        final_headers.update(agent_headers)
        if final_headers:
            combined_params["api_headers"] = final_headers
        
        # Merge query params: substituted first, then agent-provided direct params
        final_query_params = substituted_query_params.copy() if substituted_query_params else {}
        # Allow agent to provide additional query params directly
        agent_query_params = dynamic_values.get("api_query_params", {}) or {}
        final_query_params.update(agent_query_params)
        
        # Also check for direct parameters that might be query params (e.g., lat, lon)
        excluded_keys = {"api_url", "api_method", "api_headers", "api_query_params", 
                        "api_path_params", "api_body", "method", "url", "headers", "body", "location"}
        # Add any remaining dynamic_values that might be query parameters
        for key, value in dynamic_values.items():
            if key not in excluded_keys and key not in final_query_params:
                # If there's a placeholder in schema that matches this key, use the value
                if isinstance(value, (str, int, float)):
                    final_query_params[key] = str(value)
        
        if final_query_params:
            combined_params["api_query_params"] = final_query_params
        
        # Merge path params
        final_path_params = substituted_path_params.copy() if substituted_path_params else {}
        agent_path_params = dynamic_values.get("api_path_params", {}) or {}
        final_path_params.update(agent_path_params)
        if final_path_params:
            combined_params["api_path_params"] = final_path_params
        
        if substituted_body is not None:
            combined_params["api_body"] = substituted_body
        
        logger.info(f"Executing tool '{schema.name}' with dynamic_injection={use_dynamic_injection}")
        logger.debug(f"Original schema - URL: {schema.api_url}, Query: {schema.api_query_params}")
        logger.debug(f"Dynamic values from agent: {dynamic_values}")
        logger.debug(f"Final combined params - Query: {combined_params.get('api_query_params')}, Headers: {combined_params.get('api_headers')}")
        
        return await execute_api_call(input_params=combined_params)
    return tool_func


@router.post("/ReActAgent/langchain", response_model=CortexResponseFormat, tags=["Chat"])
async def invoke_react_agent(request: CortexInvokeRequestSchema, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> CortexResponseFormat:
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    session = db.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Request payload received with {len(request.tools)} tool schemas.")


    provider_id = request.provider_id
    model_id = request.model_id
    provider = await llm_router.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")
    api_key = request.api_key.get_secret_value() if request.api_key else provider.api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
    
    if provider_id == "groq":
        llm = ChatGroq(api_key=api_key, model_name=model_id, temperature=request.temperature)
    elif provider_id == "mistral":
        llm = ChatMistralAI(api_key=api_key, model=model_id, temperature=request.temperature)
    elif provider_id == "google":
        model_id = re.sub(r'models/', '', model_id)
        llm = ChatGoogleGenerativeAI(api_key=api_key, model=model_id, temperature=request.temperature)
    elif provider_id == "nvidia":
        llm = ChatNVIDIA(api_key=api_key, model=model_id, temperature=request.temperature)
    elif provider_id == "cerebras":
        llm = ChatCerebras(api_key=api_key, model=model_id, temperature=request.temperature)
    elif provider_id == "sambanova":
        llm = ChatSambaNovaCloud(api_key=api_key, model=model_id, temperature=request.temperature)
    else:
        llm = ChatOpenAI(api_key=api_key, model=model_id, temperature=request.temperature)

    executable_tools: List[StructuredTool] = []
    for tool_schema in request.tools:
        tool_function = create_tool_function(tool_schema)
        logger.info(f"Created tool function for {tool_schema.name}")
        
        # Enhance description with dynamic parameter information if enabled
        description = tool_schema.description or "No description provided."
        if getattr(tool_schema, 'dynamic_boolean', False):
            # Extract placeholder variables from schema
            all_placeholders = set()
            
            # Check query params for placeholders
            if tool_schema.api_query_params:
                for v in tool_schema.api_query_params.values():
                    if isinstance(v, str) and '{{' in v and '}}' in v:
                        placeholders = re.findall(r'\{\{(\w+)\}\}', v)
                        all_placeholders.update(placeholders)
            
            # Check path params
            if tool_schema.api_path_params:
                for v in tool_schema.api_path_params.values():
                    if isinstance(v, str) and '{{' in v and '}}' in v:
                        placeholders = re.findall(r'\{\{(\w+)\}\}', v)
                        all_placeholders.update(placeholders)
            
            # Check headers
            if tool_schema.api_headers:
                for v in tool_schema.api_headers.values():
                    if isinstance(v, str) and '{{' in v and '}}' in v:
                        placeholders = re.findall(r'\{\{(\w+)\}\}', v)
                        all_placeholders.update(placeholders)
            
            # Check URL
            if tool_schema.api_url and '{{' in tool_schema.api_url:
                placeholders = re.findall(r'\{\{(\w+)\}\}', tool_schema.api_url)
                all_placeholders.update(placeholders)
            
            # Check request payload
            if tool_schema.request_payload:
                placeholders = re.findall(r'\{\{(\w+)\}\}', tool_schema.request_payload)
                all_placeholders.update(placeholders)
            
            if all_placeholders:
                required_params = ", ".join(sorted(all_placeholders))
                description += f"\n\nRequired parameters when calling this tool: {required_params}"
        
        tool = StructuredTool.from_function(
            func=None,
            name=tool_schema.name,
            description=description,
            coroutine=tool_function
        )
        executable_tools.append(tool)
        logger.info(f"Created structured tool for {executable_tools}")
    
    logger.info(f"Successfully created {len(executable_tools)} structured tools for the agent.")

    try:
        # Pass the list of *executable* tools to the agent
        agent = create_agent(
            model=llm,
            tools=executable_tools, # <-- Use the corrected list of FunctionTool objects
            prompt=request.system_prompt,
            checkpointer=InMemorySaver(),
        )
    except Exception as agent_error:
        logger.error(f"Error creating agent: {agent_error}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {agent_error}")
    
    try:
        response_raw = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"configurable": {"thread_id": session.id}}
        )
        
        # --- Response and Metrics processing (unchanged, but included for completeness) ---
        ai_message = None
        if isinstance(response_raw, dict) and "messages" in response_raw:
            final_ai_message = next((msg for msg in reversed(response_raw["messages"]) if msg.type == "ai"), None)
            if final_ai_message:
                ai_message = final_ai_message.content
                # Process metrics
                if hasattr(final_ai_message, 'usage_metadata') and final_ai_message.usage_metadata:
                    usage = final_ai_message.usage_metadata
                    metrics = ChatMetricsDB(
                        id=str(uuid4()), session_id=session.id, user_id=user_id,
                        provider_id=provider_id, model_id=model_id,
                        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                        total_tokens=usage.get("total_tokens")
                    )
                    db.add(metrics)
                    db.commit()
            else:
                ai_message = "Agent did not produce a final answer."
        else:
             raise HTTPException(status_code=500, detail="Invalid agent response format.")

        if ai_message is None:
            raise HTTPException(status_code=500, detail="Could not extract AI response from agent output.")
            
        return CortexResponseFormat(response=ai_message)

    except Exception as e:
        logger.error(f"Error during agent invocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")