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
from app.api.v1.textual.langchain_route import router as langchain_router
from app.api.v1.sessions import router as sessions_router
from app.api.v1.frameworks import router as frameworks_router
from app.api.v1.providers import router as providers_router
from app.api.v1.agents import router as agents_router
from app.utils.logger import logger 


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    logger.info("Setting up database schema...")
    
    try:
        # Use Alembic for database migrations instead of manual table creation
        logger.info("Running database migrations with Alembic...")
        
        import subprocess
        import sys
        
        # Run alembic upgrade to apply all migrations
        result = subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.success("Database migrations completed successfully!")
        else:
            logger.error(f"Alembic migration failed: {result.stderr}")
            # Fallback to manual creation if migrations fail
            logger.warning("Falling back to manual table creation...")
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.success("Database schema ready with fallback!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        # Final fallback - try manual creation
        try:
            logger.warning("Attempting final fallback...")
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.success("Database schema ready with final fallback!")
        except Exception as e2:
            logger.error(f"Could not create database: {e2}")
            raise e2
    
    yield
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan)

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

logger.info("Including API routers...")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(frameworks_router)
app.include_router(langchain_router)
app.include_router(providers_router)
app.include_router(agents_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)