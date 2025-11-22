#app/api/v1/textual/llama_index_workflow_route.py
import asyncio
import aiohttp
import urllib.parse
from typing import Dict, Any

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool

from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from mistralai import Mistral
from mistralai import Tool, Function, UserMessage, AssistantMessage, ToolMessage

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from app.config.settings import settings
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB, LLMProviderDB, LLMModelDB, ChatMetricsDB

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

tool_parameters = {
    "type": "object",
    "properties": {
        "api_url": {"type": "string", "description": "The API endpoint URL."},
        "api_method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
            "default": "GET",
            "description": "HTTP method to use."
        },
        "api_headers": {"type": "object", "additionalProperties": {"type": "string"}},
        "api_query_params": {"type": "object", "additionalProperties": True},
        "api_path_params": {"type": "object", "additionalProperties": True},
        "api_body": {"type": "object", "additionalProperties": True},
    },
    "required": ["api_url"]
}

execute_api_call_tool = {
    "type": "function",
    "function": {
        "name": "execute_api_call",
        "description": """
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
""",
        "parameters": tool_parameters
    }
}

tools = [execute_api_call_tool]
names_to_functions = {"execute_api_call": execute_api_call}


router = APIRouter(prefix="/api/v1", tags=["textual"])



@router.post("/ReActAgent/llama_index",response_model=CortexResponseFormat,tags=["Chat"])
async def invoke_llama_index(request: CortexInvokeRequestSchema,current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> CortexResponseFormat:
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    provider_id = request.provider_id
    model_id = request.model_id
    provider = await llm_router.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")
    api_key = request.api_key.get_secret_value() if request.api_key else provider.api_key
    if provider_id == "mistral":
        mistral = Mistral(api_key=api_key)
        llm = mistral.chat.completions(
            model=model_id,
            messages=[
                UserMessage(content=request.message)
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
            temperature=request.temperature
        )
        assistant_response_message = response_from_model.choices[0].message
        messages.append(assistant_response_message)

        if assistant_response_message.tool_calls is None:
            print("\n--- MODEL DID NOT USE A TOOL ---")
            print(assistant_response_message.content)
            
        tool_call = assistant_response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)
        logger.info(f" Model decided to call function: {function_name}")
        logger.info(f" With parameters: {function_params}")

        tool_result = await names_to_functions[function_name](**function_params)
        tool_msg_content = json.dumps(tool_result, indent=2)
        messages.append(ToolMessage(content=tool_msg_content, tool_call_id=tool_call.id))
        final_response = mistral.chat.complete(
            model=model_id,
            messages=messages 
        )
        return final_response.choices[0].message.content
    
    elif provider_id == "groq":
        llm = Groq(model=model_id, api_key=api_key)
        agent = ReActAgent(tools=tools, llm=llm, verbose=True)
        ctx = Context(agent)
        handler = agent.run(request.message, ctx=ctx)
        return handler.choices[0].message.content

    elif provider_id == "cerebras":
        cerebras = Cerebras(api_key=api_key)






