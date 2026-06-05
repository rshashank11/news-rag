from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

load_dotenv()

_async_engine = None
_async_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def to_asyncpg_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url

    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return database_url


def to_sync_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    return database_url


def get_async_engine():
    global _async_engine

    if _async_engine is None:
        _async_engine = create_async_engine(
            to_asyncpg_url(settings.postgresql_url),
            pool_pre_ping=True,
        )

    return _async_engine


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _async_sessionmaker

    if _async_sessionmaker is None:
        _async_sessionmaker = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )

    return _async_sessionmaker


def AsyncSessionLocal() -> AsyncSession:
    return get_async_sessionmaker()()


engine = create_engine(
    to_sync_url(settings.postgresql_url),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5,
    pool_timeout=10,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass
