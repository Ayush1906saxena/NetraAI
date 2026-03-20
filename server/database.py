"""
Async SQLAlchemy database layer with connection pooling.

Provides the async engine, session factory, and dependency injection
for FastAPI endpoints. All models share the Base from server.models.user.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from server.config import settings
from server.models.user import Base

logger = logging.getLogger(__name__)

# ── Async engine with connection pooling ──────────────────────────────────
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# ── Session factory ───────────────────────────────────────────────────────
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables defined by the ORM models.

    Should be called once during application startup.
    """
    # Import all models so they register with Base.metadata
    import server.models.user  # noqa: F401
    import server.models.patient  # noqa: F401
    import server.models.store  # noqa: F401
    import server.models.screening  # noqa: F401
    import server.models.image  # noqa: F401
    import server.models.report  # noqa: F401
    import server.models.audit_log  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")


async def close_db() -> None:
    """Dispose of the connection pool.

    Should be called during application shutdown.
    """
    await engine.dispose()
    logger.info("Database connection pool closed.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    Commits on success, rolls back on error, and always closes.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
