# main.py
import sys
import uvicorn
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base
from app.db import models
from app.schemas.api_schemas import HealthStatus

# routers
from app.api.v1.sessions import router as sessions_router
from app.api.v1.frameworks import router as frameworks_router
from app.api.v1.agents import router as agents_router
from app.api.v1.textual.langchain_route import router as langchain_router
from app.api.v1.textual.llama_index_workflow_route import router as llama_index_workflow_router
from app.api.v1.textual.adk_route import router as adk_agent_router 
from app.api.v1.providers import router as providers_router

from app.utils.logger import logger 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    # The application now assumes the database schema is already up-to-date.
    yield
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cortexagents.netlify.app",
        "http://localhost:3000"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoints ---
@app.get("/", response_model=HealthStatus, tags=["Health"])
async def greet():
    return {"status": "ok", "message": "Welcome to the Cortex Root Endpoint"}

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Service is healthy"}

logger.info("Including API routers...")
app.include_router(sessions_router)
app.include_router(frameworks_router)
app.include_router(langchain_router)
app.include_router(providers_router)
app.include_router(agents_router)
app.include_router(llama_index_workflow_router)
app.include_router(adk_agent_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)