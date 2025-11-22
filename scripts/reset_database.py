"""
Script to reset and recreate all database tables.
Run this after updating your .env file with a new PostgreSQL connection string.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import engine, Base
from app.db import models  # Import all models to register them
from app.utils.logger import logger

def reset_database():
    """Drop all tables and recreate them from models."""
    logger.info("Starting database reset...")
    
    try:
        # Drop all tables
        logger.info("Dropping all existing tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("✓ All tables dropped successfully")
        
        # Create all tables
        logger.info("Creating all tables from models...")
        Base.metadata.create_all(bind=engine)
        logger.info("✓ All tables created successfully")
        
        logger.info("✅ Database reset completed successfully!")
        logger.info("You can now start your server with: uv run main.py")
        
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        raise

if __name__ == "__main__":
    reset_database()

