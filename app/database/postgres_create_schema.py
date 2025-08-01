#!/usr/bin/env python3
"""
Database table creation script for MultiDB Chatbot
Run this to create all PostgreSQL tables before using the application.

Usage:
  From project root: python -m app.database.postgres_create_schema
  Or from anywhere: python app/database/postgres_create_schema.py
"""

import sys
import os
import asyncio
import logging

# Add the project root to Python path so imports work from anywhere
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from app
from app.database.postgres_connection import postgres_manager
from app.database.postgres_models import DatabaseBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def postgres_create_schema():
    """Create all database tables"""
    try:
        # Initialize PostgreSQL connection
        await postgres_manager.initialize()
        logger.info("✅ Connected to PostgreSQL")

        # Get the engine
        engine = postgres_manager.engine

        # Create all tables
        async with engine.begin() as conn:
            # Drop all tables first (optional - remove if you want to keep existing data)
            # await conn.run_sync(DatabaseBase.metadata.drop_all)
            # logger.info("🗑️  Dropped existing tables")

            # Create all tables
            await conn.run_sync(DatabaseBase.metadata.create_all)
            logger.info("✅ Created all database tables")

        # List created tables
        async with postgres_manager.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))

            tables = [row[0] for row in result.fetchall()]
            logger.info(f"📋 Created tables: {', '.join(tables)}")

        logger.info("🎉 Database setup complete!")
        return True

    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    finally:
        await postgres_manager.close()

if __name__ == "__main__":
    print(f"🚀 Starting PostgreSQL schema creation...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Current directory: {os.getcwd()}")

    success = asyncio.run(postgres_create_schema())
    if success:
        print("\n✅ Database tables created successfully!")
        print("You can now run user registration.")
    else:
        print("\n❌ Database setup failed. Check the logs above.")
        sys.exit(1)