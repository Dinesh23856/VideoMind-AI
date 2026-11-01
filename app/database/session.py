"""Database engine, declarative base, and session management."""

from __future__ import annotations

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

db_url = settings.database_url
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

if db_url.startswith("sqlite"):
    import sqlite3
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _load_sqlite_vec_extension(dbapi_connection, connection_record):
        if isinstance(dbapi_connection, sqlite3.Connection):
            try:
                import sqlite_vec
                dbapi_connection.enable_load_extension(True)
                sqlite_vec.load(dbapi_connection)
                dbapi_connection.enable_load_extension(False)
            except Exception:
                pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the configured database."""
    try:
        import app.database.models  # noqa: F401 - ensure models are registered on Base.metadata
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
