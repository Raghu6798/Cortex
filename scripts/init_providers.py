#!/usr/bin/env python3
"""
Script to initialize providers and models in the database
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import SessionLocal
from app.integrations.llm_router import llm_router
from app.db.models import LLMProviderDB, LLMModelDB
from app.utils.logger import logger 

async def init_providers():
    """Initialize providers and models in the database"""
    db = SessionLocal()
    try:
        logger.info("Syncing providers and models to database...")
        await llm_router.sync_providers_to_db(db)
        logger.info("Providers and models synced successfully!")
        
        # Print summary
        providers_count = db.query(LLMProviderDB).count()
        models_count = db.query(LLMModelDB).count()
        logger.info(f"Database now contains {providers_count} providers and {models_count} models")
        
    except Exception as e:
        logger.error(f"Error syncing providers: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Initializing providers and models...")
    asyncio.run(init_providers())
