# Cortex/main.py

import sys
from pathlib import Path

# --- THIS IS THE KEY FIX FOR IMPORTS ---
# Add the project's root directory (the one containing this 'main.py' file)
# to the Python path. This allows absolute imports to work from any context.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# --- END FIX ---

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# This import will now work because the root directory is on the path
from app.api.v1 import agent_route

# --- Main FastAPI App Initialization ---
app = FastAPI(title="Secure Agenta ADE Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router from your agent_route file
app.include_router(agent_route.router)

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the server is running."""
    return {"message": "Welcome to the Agenta ADE Backend"}
