import os
import sys
from uuid import uuid4
import asyncio
import aiohttp
import httpx
from pydantic import BaseModel, Field
import traceback
from typing import Optional,AsyncIterator,Dict,Any
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.tools import tool
from langchain_sambanova import ChatSambaNovaCloud
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

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from app.utils.logger import logger
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB, LLMProviderDB, LLMModelDB, ChatMetricsDB


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

@router.post("/chat/invoke",response_model=CortexResponseFormat,tags=["Chat"])
async def invoke_react_agent(request: CortexInvokeRequestSchema,current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> CortexResponseFormat:

    """
    Invokes a ReAct agent with the given request.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    session = db.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info(f"request payload : {request}")
    provider_id = request.provider_id
    model_id = request.model_id
    
    logger.info(f"Request provider_id: {provider_id}")
    logger.info(f"Request model_id: {model_id}")
    
    # Get provider info from llm_router
    provider = await llm_router.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} not found")

    api_key = request.api_key.get_secret_value() if request.api_key else provider.api_key
    
    logger.info(f"Provider ID: {provider_id} and {provider.provider_info}")
    logger.info(f"Base URL: {request.base_url or provider.provider_info.base_url}")
    logger.info(f"The temperature is: {request.temperature}")
    logger.info(f"The max tokens is: {request.max_tokens}")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required for this provider. Please configure your API key in the agent settings.")
    
    # Dynamic LLM initialization based on provider
    if provider_id == "sambanova":
        llm = ChatSambaNovaCloud(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    elif provider_id == "cerebras":
        llm = ChatCerebras(
            base_url=provider.provider_info.base_url,
            model=model_id,
            temperature=request.temperature,
            api_key=SecretStr(api_key),
        )
    elif provider_id == "groq":
        llm = ChatGroq(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    elif provider_id == "google":
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    elif provider_id == "mistral":
        llm = ChatMistralAI(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    elif provider_id == "nvidia":
    
        llm = ChatNVIDIA(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    else:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_id,
            temperature=request.temperature
        )
    tools = [execute_api_call]
    agent = create_agent(
        model=llm,
        tools=tools,
        prompt=request.system_prompt,
        checkpointer=InMemorySaver(),
    )
    
    try:
        response_raw = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config={"configurable": {"thread_id": session.id}}
        )
        logger.info(f"Response type: {type(response_raw)}")
        logger.info(f"Response content: {response_raw}")
        
        # Extract the AI message content for frontend display
        ai_message = None
        if isinstance(response_raw, dict) and "messages" in response_raw:
            logger.info(f"Processing {len(response_raw['messages'])} messages")
            for i, message in enumerate(response_raw["messages"]):
                logger.info(f"Message {i}: type={message.type}, content_length={len(str(message.content)) if hasattr(message, 'content') else 0}")
                if message.type == "ai" or message.type == "AIMessage":
                    ai_message = message.content
                    logger.info(f"Found AI message: {ai_message[:100]}...")
                    break
        
        if not ai_message:
            # Try to get the last message if no AI message found
            if isinstance(response_raw, dict) and "messages" in response_raw and response_raw["messages"]:
                last_message = response_raw["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    ai_message = last_message.content
                else:
                    raise HTTPException(status_code=500, detail="No AI response generated")
            else:
                raise HTTPException(status_code=500, detail="No AI response generated")
        
        # Store the full response object in database for metrics tracking
        try:
            # Extract metrics from the response
            if isinstance(response_raw, dict) and "messages" in response_raw:
                ai_message_obj = None
                for message in response_raw["messages"]:
                    if message.type == "ai":
                        ai_message_obj = message
                        break
                
                if ai_message_obj and hasattr(ai_message_obj, 'response_metadata') and ai_message_obj.response_metadata:
                    metadata = ai_message_obj.response_metadata
                    token_usage = metadata.get("token_usage", {})
                    usage_metadata = getattr(ai_message_obj, 'usage_metadata', {})
                    
                    # Create metrics record
                    metrics = ChatMetricsDB(
                        id=str(uuid4()),
                        session_id=session.id,
                        user_id=user_id,
                        provider_id=provider_id,
                        model_id=model_id,
                        input_tokens=usage_metadata.get("input_tokens") or token_usage.get("prompt_tokens"),
                        output_tokens=usage_metadata.get("output_tokens") or token_usage.get("completion_tokens"),
                        total_tokens=usage_metadata.get("total_tokens") or token_usage.get("total_tokens"),
                        completion_time=token_usage.get("completion_time"),
                        prompt_time=token_usage.get("prompt_time"),
                        queue_time=token_usage.get("queue_time"),
                        total_time=token_usage.get("total_time"),
                        model_name=metadata.get("model_name"),
                        system_fingerprint=metadata.get("system_fingerprint"),
                        service_tier=metadata.get("service_tier"),
                        finish_reason=metadata.get("finish_reason"),
                        response_metadata=metadata
                    )
                    
                    db.add(metrics)
                    db.commit()
                    
                    total_tokens = usage_metadata.get("total_tokens") or token_usage.get("total_tokens")
                    total_time = token_usage.get("total_time")
                    logger.info(f"Stored metrics for session {session.id}: {total_tokens} tokens, {total_time}s")
                else:
                    logger.warning("No response metadata found for metrics tracking")
                    
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
            # Don't fail the request if metrics storage fails
        
        # Return only the AI message content to frontend
        return CortexResponseFormat(response=ai_message)
        
    except Exception as e:
        logger.error(f"Error during agent invocation: {str(e)}")
        if "invalid_api_key" in str(e).lower() or "authentication" in str(e).lower():
            raise HTTPException(status_code=401, detail=f"Invalid API key for {provider_id}. Please check your API key in the agent settings. Error: {str(e)}")
        elif "rate_limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
 