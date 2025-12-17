import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
    JSON,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# -------------------------------------------------------------------
# SINGLE Base FOR THE WHOLE PROJECT
# -------------------------------------------------------------------
Base = declarative_base()


# ------------------ Document Table ------------------ #
class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    original_path = Column(String, nullable=True)
    file_size = Column(Integer)
    checksum = Column(String)
    pages = Column(Integer)
    status = Column(String, default="uploaded")  # uploaded / processing / completed / failed
    priority = Column(String, default="Low")
    approved = Column(Boolean, default=False)   # overall approval status
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    doc_metadata = Column(JSON, default=dict)

    # Relationships
    extracted_texts = relationship(
        "ExtractedText",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    department_contents = relationship(
        "DepartmentContent",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    department_tables = relationship(
        "DepartmentTable",
        back_populates="document",
        cascade="all, delete-orphan",
    )


# ------------------ Department Content Table ------------------ #
class DepartmentContent(Base):
    __tablename__ = "department_contents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    department = Column(String, nullable=False)
    content = Column(Text)  # summary/paragraph text
    page_start = Column(Integer)
    page_end = Column(Integer)
    confidence = Column(Float, default=1.0)
    keywords_matched = Column(JSON, default=dict)
    pdf_path = Column(String)
    doc_priority = Column(String, nullable=False, default="Low")
    approved = Column(Boolean, default=False)   # department approval
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    document = relationship("Document", back_populates="department_contents")


# ------------------ Extracted Text Table ------------------ #
class ExtractedText(Base):
    __tablename__ = "extracted_texts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number = Column(Integer, nullable=False)
    text_content = Column(Text)  # full-page cleaned text
    language = Column(String, default="unknown")
    confidence = Column(Float, default=0.0)
    ocr_engine = Column(String, default="tesseract")
    bbox = Column(JSON, default=dict)
    tables = Column(JSON, default=dict)  # raw tables per page
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="extracted_texts")


# ------------------ Department Tables Table ------------------ #
class DepartmentTable(Base):
    __tablename__ = "department_tables"

    id = Column(Integer, primary_key=True, index=True)

    # IMPORTANT: use UUID here to match Document.id type
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    department = Column(String, index=True, nullable=False)

    page_number = Column(Integer, nullable=False)

    # store normalized 2D table as JSON: [["col1","col2"], ["v1","v2"], ...]
    rows = Column(JSON, nullable=False)

    confidence = Column(Float, default=1.0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="department_tables")
