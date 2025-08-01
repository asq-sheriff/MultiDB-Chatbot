import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import text

from app.config import config

logger = logging.getLogger(__name__)

# Import DatabaseBase from models
from app.database.postgres_models import DatabaseBase

class PostgreSQLConnectionManager:
    """
    PostgreSQL connection manager with async support.
    Compatible with modern SQLAlchemy and asyncpg versions.

    Used by: services layer for business logic operations
    Integration: Called from multi_db_service.py and individual services
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize PostgreSQL connection pool.
        Called by: main.py during application startup
        """
        if self._initialized:
            logger.info("PostgreSQL already initialized")
            return

        try:
            # Determine if we should use connection pooling
            use_pooling = config.postgresql.host not in ["localhost", "127.0.0.1"]

            # Create engine parameters based on SQLAlchemy 2.0.41 and asyncpg 0.30.0
            engine_kwargs = {
                "echo": False,  # Set to True for SQL debugging
                "pool_pre_ping": True,  # Validate connections
            }

            # Add pooling parameters only if using QueuePool
            if use_pooling:
                engine_kwargs.update({
                    "poolclass": QueuePool,
                    "pool_size": config.postgresql.pool_size,
                    "max_overflow": config.postgresql.max_overflow,
                    "pool_timeout": config.postgresql.pool_timeout,
                    "pool_recycle": config.postgresql.pool_recycle,
                })
            else:
                # For localhost, use NullPool with simplified parameters
                engine_kwargs.update({
                    "poolclass": NullPool,
                })

            # Create async engine
            self._engine = create_async_engine(
                config.postgresql.url,
                **engine_kwargs
            )

            # Create session factory (SQLAlchemy 2.0 syntax)
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )

            # Test connection
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                if test_value != 1:
                    raise ConnectionError("PostgreSQL connection test failed")

            self._initialized = True
            pool_info = "with connection pooling" if use_pooling else "with NullPool"
            logger.info(f"✅ PostgreSQL connected {pool_info}: {config.postgresql.host}:{config.postgresql.port}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            raise ConnectionError(f"Cannot connect to PostgreSQL: {e}")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get PostgreSQL session with automatic transaction management.

        Used by: All services that need PostgreSQL database access
        Usage pattern:
            async with postgres_manager.get_session() as session:
                # Database operations here
                pass
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get_session_sync(self) -> AsyncSession:
        """
        Get PostgreSQL session for dependency injection.
        Used by: FastAPI dependencies in app/api/dependencies.py
        """
        if not self._initialized:
            await self.initialize()
        return self._session_factory()

    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine for migrations"""
        if not self._engine:
            raise RuntimeError("PostgreSQL not initialized")
        return self._engine

    async def test_connection(self) -> bool:
        """Test PostgreSQL connection"""
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"PostgreSQL version: {version}")
                return True
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {e}")
            return False

    async def close(self) -> None:
        """Close PostgreSQL connections"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("PostgreSQL connections closed")

# Global PostgreSQL manager instance
postgres_manager = PostgreSQLConnectionManager()

# Helper functions for easy access
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL session - used in dependency injection"""
    async with postgres_manager.get_session() as session:
        yield session