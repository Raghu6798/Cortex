# app/api/v1/textual/langchain_route.py

import os
import sys
import urllib.parse
from uuid import uuid4
import asyncio
import aiohttp
import httpx
from pydantic import BaseModel, Field, SecretStr
import traceback
from typing import Optional, AsyncIterator, Dict, Any, List, Callable

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerai
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_sambanova import ChatSambaNovaCloud

from langchain_core.tools import tool, FunctionTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat, ToolConfigSchema
from app.utils.logger import logger
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB, ChatMetricsDB

# This function is the universal tool executor. It's unchanged.
@tool
async def execute_api_call(input_params: Dict[str, Any]):
    """
    Executes an HTTP API call (GET, POST, PUT, DELETE, etc.) with the specified parameters.
    """
    # ... (rest of the function is the same as before, no changes needed here)
    pass # Placeholder for brevity

router = APIRouter(prefix="/api/v1", tags=["textual"])

# Helper function to prevent Python's closure issue in loops
def create_tool_function(schema: ToolConfigSchema) -> Callable:
    """Creates a unique function for a tool schema to be used by FunctionTool."""
    async def tool_func(input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        # Combines static params from schema with dynamic ones from LLM
        combined_params = {
            "api_url": schema.api_url,
            "api_method": schema.api_method,
            "api_headers": schema.api_headers,
            "api_query_params": schema.api_query_params,
            "api_path_params": schema.api_path_params,
            "api_body": schema.request_payload, # Map request_payload to api_body
            **(input_params or {})
        }
        logger.info(f"Executing tool '{schema.name}' with params: {combined_params}")
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

    # --- LLM Initialization (this part of your code is correct) ---
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
    # ... add other providers here
    else:
        llm = ChatOpenAI(api_key=api_key, model=model_id, temperature=request.temperature)

    # =================== THIS IS THE FIX ===================
    # Dynamically create executable LangChain tools from the incoming schemas.
    executable_tools: List[FunctionTool] = []
    for tool_schema in request.tools:
        # Use our helper to create a unique function for this tool
        tool_function = create_tool_function(tool_schema)
        
        # Create the LangChain FunctionTool object
        dynamic_tool = FunctionTool(
            name=tool_schema.name,
            description=tool_schema.description,
            func=tool_function,
        )
        executable_tools.append(dynamic_tool)
    
    logger.info(f"Successfully created {len(executable_tools)} executable LangChain tools.")
    # ======================================================

    try:
        # Pass the list of *executable* tools to the agent
        agent = create_agent(
            llm=llm,
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