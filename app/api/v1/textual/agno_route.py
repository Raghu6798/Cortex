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

from app.schemas.api_schemas import CortexInvokeRequestSchema, CortexResponseFormat
from app.config.settings import settings
from app.auth.clerk_auth import get_current_user
from app.integrations.llm_router import llm_router
from app.db.database import get_db
from app.db.models import ChatSessionDB
from app.utils.logger import logger

load_dotenv()

async def execute_api_call(api_url: str,api_method: str = "GET",api_headers: Optional[Dict[str, str]] = None,api_query_params: Optional[Dict[str, Any]] = None,api_path_params: Optional[Dict[str, Any]] = None,api_body: Optional[Dict[str, Any]] = None,)->Dict[str, Any]:
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
    db: Session = Depends(get_db)
) -> CortexResponseFormat:
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = db.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).first()
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