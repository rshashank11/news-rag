from sqlalchemy import create_engine  # Creates the main database engine/connection factory.
from sqlalchemy import text  # Lets us run plain SQL safely when we need a small schema change.
from sqlalchemy.orm import sessionmaker, DeclarativeBase  # Sessionmaker creates DB sessions; DeclarativeBase is the ORM base class.

import os  # Lets us read database URLs from environment variables.
from dotenv import load_dotenv  # Loads local .env values while running locally.

load_dotenv()  # Makes .env variables available through os.environ.

POSTGRESQL_URL = (  # Stores the Postgres connection string.
    os.environ.get("POSTGRESQL_URL")  # Preferred variable name in this project.
    or os.environ.get("DATABASE_URL", "")  # Fallback name used by many cloud hosts.
)
engine = create_engine(POSTGRESQL_URL)  # Creates the object SQLAlchemy uses to talk to Postgres.

SessionLocal = sessionmaker(  # Creates new database sessions whenever the app needs to query Postgres.
    autoflush=False,  # Prevents SQLAlchemy from automatically writing pending changes before every query.
    bind=engine,  # Connects every session produced here to our Postgres engine.
)

class Base(DeclarativeBase):  # Parent class for all SQLAlchemy table models.
    pass  # No extra behavior is needed; child models inherit SQLAlchemy mapping behavior.

def get_session():  # FastAPI-style helper that gives one DB session to a request.
    db = SessionLocal()  # Opens a new database session.
    try:
        yield db  # Gives the session to the caller.
    finally:
        db.close()  # Always closes the session so connections are not leaked.


def ensure_database_schema() -> None:  # Small startup migration so old databases still work.
    with engine.begin() as connection:  # Opens a transaction and commits automatically if there is no error.
        connection.execute(  # Sends one SQL command to Postgres.
            text(  # Wraps raw SQL in SQLAlchemy's text object.
                "ALTER TABLE IF EXISTS stories "  # Only changes the table if it already exists.
                "ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ"  # Adds publish date column without breaking existing DBs.
            )
        )
