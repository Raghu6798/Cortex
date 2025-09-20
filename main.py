import os
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
import asyncio
load_dotenv()
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Perform calculations."""
    return str(eval(expression))


llm = ChatOpenAI(
    model = "magistral-small-2507",
    base_url = "https://api.mistral.ai/v1/",
    api_key = os.getenv("MISTRAL_API_KEY")
)
tools = [search,calculate]
checkpointer = InMemorySaver()
agent = create_agent(model=llm,tools = tools,prompt = "You are a helpful AI assistant",checkpointer=checkpointer,)

thread_id = str(uuid4())

# --- Async runner ---
async def run_stream():
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": "What are your capabilities in detail"}]},
        stream_mode="messages",
        config={"configurable": {"thread_id": thread_id}},
    ):   
        chunk, metadata = event
        print(chunk.content,end = "",flush=True)


if __name__ == "__main__":
    asyncio.run(run_stream())

