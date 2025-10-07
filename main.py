


import sys
import uvicorn
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Project-specific imports ---
from app.db.database import engine
from app.db import models
from app.schemas.api_schemas import HealthStatus
# Import the routers from their new locations
from app.api.v1.sessions import router as sessions_router
from app.api.v1.frameworks import router as frameworks_router
from app.api.v1.chat import router as chat_router

# --- Logger Configuration ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>", colorize=True)
log = logger

# --- FastAPI App Instance ---
app = FastAPI(
    title="Cortex Backend",
    description="A Backend-as-a-Service for AI Agents.",
    version="1.0.0"
)

# --- Startup Event ---
@app.on_event("startup")
def on_startup():
    """Actions to perform on application startup."""
    log.info("Application starting up...")
    log.info("Attempting to create database tables...")
    try:
        models.Base.metadata.create_all(bind=engine)
        log.success("Database tables created or already exist.")
    except Exception as e:
        log.error(f"Could not create database tables: {e}")

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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

log.info("Including API routers...")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(frameworks_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1") 

# --- Main Entrypoint ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)