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

from app.config.settings import settings


# --- Generic Async API Executor ---
async def execute_api_call(
    method: str,
    url: str,
    params: Dict[str, Any] = None,
    headers: Dict[str, Any] = None,
    json_body: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generic async function to make an HTTP request to any API.
    """
    request_kwargs = {}
    if headers:
        request_kwargs["headers"] = headers
    if json_body:
        request_kwargs["json"] = json_body

    final_url = url
    if params:
        final_url += "?" + urllib.parse.urlencode(params)

    print(f"▶️  Executing API Call: {method} {final_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, final_url, **request_kwargs) as response:
                try:
                    response_data = await response.json()
                except aiohttp.ContentTypeError:
                    response_data = await response.text()

                if response.status >= 400:
                    return {"error": "API request failed", "status": response.status, "details": response_data}
                
                print(f"✅ API Call Successful. Status: {response.status}")
                return {"result": response_data}

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


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
