# NEW FILE: app/api/v1/chat.py

import aiohttp
import urllib.parse
from uuid import uuid4
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import create_model, Field

from langchain.agents import create_agent
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from app.config.settings import get_chat_model
from app.schemas.api_schemas import InvokeRequestSchema, CortexResponseFormat

log = logger
router = APIRouter()


@tool
async def execute_api_call(input_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically executes any HTTP request based on a structured API definition.
    This is a generic tool executor.
    """
    method = input_params.get("api_method", "GET").upper()
    base_url = input_params.get("api_url")
    path_params_def = input_params.get("api_path_params", [])
    query_params_def = input_params.get("api_query_params", [])
    headers_def = input_params.get("api_headers", [])

    if not base_url:
        return {"error": "API URL (api_url) was not specified."}

    request_kwargs = {}
    try:
        final_url = base_url
        query_params = {param["key"]: param["value"] for param in query_params_def}
        if query_params:
            final_url += "?" + urllib.parse.urlencode(query_params)

        headers = {header["key"]: header["value"] for header in headers_def}
        if headers:
            request_kwargs["headers"] = headers

    except Exception as e:
        return {"error": f"Error constructing request: {e}"}

    log.info(f"Executing dynamic tool API call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, final_url, **request_kwargs) as response:
                response_data = await response.text()
                if response.status >= 400:
                    return {"error": "API request failed.", "status": response.status, "details": response_data}
                return {"status": response.status, "response": response_data}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

# --- Helper to create tools on the fly ---
def create_dynamic_tool(tool_config: dict) -> BaseTool:
    """Creates a LangChain tool dynamically from a configuration dictionary."""
    args_fields = {
        param['key']: (str, Field(..., description=f"Value for the '{param['key']}' parameter."))
        for param in tool_config.get("api_query_params", [])
    }
    DynamicArgsModel = create_model(f"{tool_config['name']}Args", **args_fields)

    async def dynamic_tool_func(**kwargs):
        input_params = {
            "api_url": tool_config["api_url"],
            "api_method": tool_config["api_method"],
            "api_headers": tool_config.get("api_headers", []),
            "api_query_params": [{"key": k, "value": v} for k, v in kwargs.items()]
        }
        return await execute_api_call.ainvoke({"input_params": input_params})

    return tool(
        name=tool_config["name"],
        description=tool_config["description"],
        args_schema=DynamicArgsModel,
        func=dynamic_tool_func,
        coroutine=True,
    )

# --- Main Agent Invocation Endpoint ---
@router.post("/chat/invoke", response_model=CortexResponseFormat, tags=["Chat"])
async def invoke_agent_sync(request: InvokeRequestSchema) -> CortexResponseFormat:
    """
    Receives agent config and a message, runs the agent to completion,
    and returns a single, final response.
    """
    log.info(f"Received request for model '{request.model_name}' with {len(request.tools)} tools.")
    try:
        llm = get_chat_model(request)
        
        # Dynamically create tools from the request
        dynamic_tools = [create_dynamic_tool(t.model_dump()) for t in request.tools]
        
        agent = create_agent(
            model=llm,
            tools=dynamic_tools, # Use the dynamically created tools
            prompt=request.system_prompt,
            checkpointer=InMemorySaver(),
        )

        log.info(f"Invoking agent with input: '{request.message}'")
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        response_data = await agent.ainvoke({"messages": [{"role": "user", "content": request.message}]}, config)
        
        final_output = "Agent did not return a final message."
        if response_data and "messages" in response_data and response_data["messages"]:
            last_message = response_data["messages"][-1]
            if hasattr(last_message, "content"):
                final_output = last_message.content

        log.success("Agent invocation finished successfully.")
        return CortexResponseFormat(response=final_output)

    except Exception as e:
        log.exception("An error occurred during agent invocation.")
        raise HTTPException(status_code=500, detail={"error": f"An error occurred: {e}"})