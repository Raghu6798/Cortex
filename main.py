# import os
# from uuid import uuid4
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver

# from dotenv import load_dotenv
# import asyncio
# load_dotenv()
# @tool
# def search(query: str) -> str:
#     """Search for information."""
#     return f"Results for: {query}"

# @tool
# def calculate(expression: str) -> str:
#     """Perform calculations."""
#     return str(eval(expression))


# llm = ChatOpenAI(
#     model = "magistral-small-2507",
#     base_url = "https://api.mistral.ai/v1/",
#     api_key = os.getenv("MISTRAL_API_KEY")
# )
# tools = [search,calculate]
# checkpointer = InMemorySaver()
# agent = create_agent(model=llm,tools = tools,prompt = "You are a helpful AI assistant",checkpointer=checkpointer,)

# thread_id = str(uuid4())

# # --- Async runner ---
# async def run_stream():
#     async for event in agent.astream(
#         {"messages": [{"role": "user", "content": "What are your capabilities in detail"}]},
#         stream_mode="messages",
#         config={"configurable": {"thread_id": thread_id}},
#     ):   
#         chunk, metadata = event
#         print(chunk.content,end = "",flush=True)


# if __name__ == "__main__":
#     asyncio.run(run_stream())

# This is for backend scripts, NOT for authenticating incoming user requests
from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
clerk_secret_key = "sk_test_HOg06gde6cb3jgODmUNRDG58vknmZ4FYxOmFpVfGAU"


clerk_sdk = Clerk(bearer_auth=clerk_secret_key)
bearer_scheme = HTTPBearer()



async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    try:
        # The authenticate_request method is designed for this exact purpose.
        # It takes the full request and extracts/verifies the token.
        # NOTE: This requires passing the raw request, which is more complex in FastAPI.
        # A simpler, direct token verification is often easier. Let's use that.
        session_claims = clerk_sdk.sessions.verify_token(creds.credentials)
        return session_claims
    except Exception as e:
        log.error(f"Clerk authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")