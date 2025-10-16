#!/usr/bin/env python3
"""
Script to initialize providers and models in the database
"""

import asyncio
import sys
from pathlib import Path

from backend.app.db.database import SessionLocal
from backend.app.integrations.llm_router import llm_router
from backend.app.db.models import LLMProviderDB, LLMModelDB

async def init_providers():
    """Initialize providers and models in the database"""
    db = SessionLocal()
    try:
        print("Syncing providers and models to database...")
        await llm_router.sync_providers_to_db(db)
        print("Providers and models synced successfully!")
        
        # Print summary
        providers_count = db.query(LLMProviderDB).count()
        models_count = db.query(LLMModelDB).count()
        print(f"Database now contains {providers_count} providers and {models_count} models")
        
    except Exception as e:
        print(f"Error syncing providers: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(init_providers())
