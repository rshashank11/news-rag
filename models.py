from database import Base
import uuid
from sqlalchemy import Column, Text, String
from sqlalchemy.dialects.postgresql import UUID, ARRAY

class StoryMetadata(Base):
    __tablename__="stories"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    headline = Column(String, nullable=False)
    summary = Column(Text)
    topics = Column(ARRAY(String))
    categories = Column(ARRAY(String))
    full_content = Column(Text, nullable=False)
