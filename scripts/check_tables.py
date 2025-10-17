#app/scripts/check_tables.py
"""
Script to check database tables.
"""

import sys
import os

import transformers
transformers.logging.set_verbosity_error()

from sqlalchemy import text
from app.db.database import engine
from app.utils.logger import logger as log

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['PYTORCH_NO_WARN_DEPRECATED_API'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_tables():
    """Check what tables exist in the database."""
    try:
        log.info("🔍 Checking database tables...")
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result]
            
            log.success(f"✅ Database connection successful!")
            log.info(f"📋 Found {len(tables)} tables: {tables}")
            
            if not tables:
                log.warning("⚠️  No tables found in database")
            else:
                for table in tables:
                    log.info(f"  - {table}")
                    
    except Exception as e:
        log.error(f"❌ Database check failed: {e}")
        raise e

if __name__ == "__main__":
    check_tables()
