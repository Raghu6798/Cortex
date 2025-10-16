# REVISED FILE: app/api/v1/textual/llama_index_workflow_route.py

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

from backend.app.config.settings import settings


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


# --- Tool Definition ---
def get_math_result(expression: str) -> Dict[str, Any]:
    """
    Calculates a mathematical expression using the math.js API.
    """
    return asyncio.run(execute_api_call(
        method="GET",
        url="http://api.mathjs.org/v4/",
        params={"expr": expression}  # only single encode!
    ))


print("--- Setting up LlamaIndex ReActAgent with a custom API tool ---")

# 1. Initialize the LLM
llm = Groq(model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=settings.GROQ_API_KEY)

# 2. Wrap math function as a proper LlamaIndex tool
math_tool = FunctionTool.from_defaults(fn=get_math_result, name="MathTool", description="Evaluate math expressions.")

# 3. Set up the agent
agent = ReActAgent(tools=[math_tool], llm=llm, verbose=True)

# 4. Run the agent
ctx = Context(agent)
query = "What is 20 + (2 * 4)?"
print(f"\n--- Running agent with query: '{query}' ---")
async def main():
    print(f"\n--- Running agent with query: '{query}' ---")
    handler = await agent.run(query, ctx=ctx)
    print(handler)

if __name__ == "__main__":
    asyncio.run(main())


router = APIRouter(prefix="/chat", tags=["chat"])
