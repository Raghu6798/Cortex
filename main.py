import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# --- END KEY ---

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import agent_route

# --- Main FastAPI App Initialization ---
app = FastAPI(title="Secure Agenta ADE Backend")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_route.router)

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the server is running."""
    return {"message": "Welcome to the Agenta ADE Backend"}

