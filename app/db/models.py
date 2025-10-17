# app/db/models.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON, Boolean, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func 
from app.db.database import Base

class ChatSessionDB(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    
    # Store the complex AgentConfig Pydantic model as JSON
    agent_config = Column(JSON, nullable=False)
    
    memory_usage = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Establish the one-to-many relationship to messages
    messages = relationship("MessageDB", back_populates="session", cascade="all, delete-orphan")

class MessageDB(Base):
    __tablename__ = "messages"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    sender = Column(String, nullable=False)
    text = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Store the list of files as JSON
    files = Column(JSON, nullable=True)
    
    # Establish the many-to-one relationship back to the session
    session = relationship("ChatSessionDB", back_populates="messages")

class LLMProviderDB(Base):
    __tablename__ = "llm_providers"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String, nullable=False)
    base_url = Column(String, nullable=False)
    logo_url = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    requires_api_key = Column(Boolean, default=True)
    supports_streaming = Column(Boolean, default=True)
    supports_tools = Column(Boolean, default=True)
    supports_embeddings = Column(Boolean, default=False)
    max_tokens = Column(Integer, default=4096)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to models
    models = relationship("LLMModelDB", back_populates="provider", cascade="all, delete-orphan")

class LLMModelDB(Base):
    __tablename__ = "llm_models"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    provider_id = Column(String, ForeignKey("llm_providers.id"), nullable=False)
    model_id = Column(String, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    context_length = Column(Integer, default=4096)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to provider
    provider = relationship("LLMProviderDB", back_populates="models")

class ChatMetricsDB(Base):
    __tablename__ = "chat_metrics"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    provider_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    
    # Token usage metrics
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    # Timing metrics
    completion_time = Column(Float, nullable=True)
    prompt_time = Column(Float, nullable=True)
    queue_time = Column(Float, nullable=True)
    total_time = Column(Float, nullable=True)
    
    # Model and service info
    model_name = Column(String, nullable=True)
    system_fingerprint = Column(String, nullable=True)
    service_tier = Column(String, nullable=True)
    finish_reason = Column(String, nullable=True)
    
    # Full response metadata as JSON for detailed tracking
    response_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to session
    session = relationship("ChatSessionDB")