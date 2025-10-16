import sys
import uvicorn
from loguru import logger
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db.database import engine
from backend.app.db import models
from backend.app.schemas.api_schemas import HealthStatus

from backend.app.db.database import engine, Base
from backend.app.api.v1.textual.langchain_route import router as langchain_router
from backend.app.api.v1.sessions import router as sessions_router
from backend.app.api.v1.frameworks import router as frameworks_router
from backend.app.api.v1.providers import router as providers_router

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>", colorize=True)
log = logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Application starting up...")
    log.info("Setting up database schema...")
    
    try:
        # Use Alembic for database migrations instead of manual table creation
        log.info("Running database migrations with Alembic...")
        
        import subprocess
        import sys
        
        # Run alembic upgrade to apply all migrations
        result = subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            log.success("Database migrations completed successfully!")
        else:
            log.error(f"Alembic migration failed: {result.stderr}")
            # Fallback to manual creation if migrations fail
            log.warning("Falling back to manual table creation...")
            Base.metadata.create_all(bind=engine, checkfirst=True)
            log.success("Database schema ready with fallback!")
        
    except Exception as e:
        log.error(f"Database setup failed: {e}")
        # Final fallback - try manual creation
        try:
            log.warning("Attempting final fallback...")
            Base.metadata.create_all(bind=engine, checkfirst=True)
            log.success("Database schema ready with final fallback!")
        except Exception as e2:
            log.error(f"Could not create database: {e2}")
            raise e2
    
    yield
    log.info("Application shutting down...")


app = FastAPI(lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
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
app.include_router(langchain_router, prefix="/api/v1")
app.include_router(providers_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)