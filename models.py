from database import Base  # Imports the SQLAlchemy base class used by all database models.
import uuid  # Generates unique IDs when a story does not already provide one.
from sqlalchemy import Column, DateTime, Text, String  # Column types used in the stories table.
from sqlalchemy.dialects.postgresql import UUID, ARRAY  # Postgres-specific types for UUIDs and string arrays.

class StoryMetadata(Base):  # Python class that maps to the Postgres stories table.
    __tablename__ = "stories"  # Exact table name in Postgres.

    id = Column(  # Unique story ID column.
        UUID(as_uuid=True),  # Stores UUID values as real Python uuid.UUID objects.
        primary_key=True,  # Marks this column as the table's unique identifier.
        default=uuid.uuid4,  # Generates a random UUID only if no ID is provided.
    )
    headline = Column(String, nullable=False)  # Story headline; required because every source needs a title.
    summary = Column(Text)  # Optional short summary/meta description.
    published_at = Column(DateTime(timezone=True))  # Optional publish date with timezone awareness.
    topics = Column(ARRAY(String))  # Optional list of tag/topic names from the raw story.
    categories = Column(ARRAY(String))  # Optional list of article sections/categories.
    full_content = Column(Text, nullable=False)  # Full cleaned article text; required for answer generation.
