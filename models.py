import uuid

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from database import Base


class StoryMetaData(Base):
    __tablename__ = "stories"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    headline = Column(
        String,
        nullable=False,
    )

    summary = Column(
        Text,
        nullable=True,
    )

    published_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    topics = Column(
        ARRAY(String),
        nullable=True,
    )

    categories = Column(
        ARRAY(String),
        nullable=True,
    )

    full_content = Column(
        Text,
        nullable=False,
    )
