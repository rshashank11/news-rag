import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

POSTGRESQL_URL = os.environ.get("POSTGRESQL_URL")

if not POSTGRESQL_URL:
    raise RuntimeError("POSTGRESQL_URL is missing from .env")

engine = create_engine(
    POSTGRESQL_URL,
    # checks whether the database connection is alive before using it
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

class Base(DeclarativeBase):
    pass