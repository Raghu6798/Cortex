# app/db/models.py
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON, Boolean, Text, Integer, BigInteger,Table
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func 
from typing import List, Optional

from app.db.database import Base
from app.utils.encryption import encrypt_value, decrypt_value

agent_tool_association = Table(
    'agent_tool_association',
    Base.metadata,
    Column('agent_id', String, ForeignKey('agents.AgentId'), primary_key=True),
    Column('tool_id', String, ForeignKey('configured_tools.id'), primary_key=True)
)


class ConfiguredToolDB(Base):
    __tablename__ = "configured_tools"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True) 
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    tool_type = Column(String, nullable=False)
    

    config_data = Column(JSON, nullable=False) 
    

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


    agents: Mapped[List["AgentDB"]] = relationship(
        secondary=agent_tool_association,
        back_populates="configured_tools"
    )


class AgentDB(Base):
    __tablename__ = "agents"
    __table_args__ = {'extend_existing': True}

    AgentId = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    architecture = Column(String, nullable=False) 
    framework = Column(String, nullable=False)     
    settings = Column(JSON, nullable=False)        

    configured_tools: Mapped[List["ConfiguredToolDB"]] = relationship(
        secondary=agent_tool_association,
        back_populates="agents"
    )
    
 
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to chat sessions
    sessions = relationship("ChatSessionDB", back_populates="agent", cascade="all, delete-orphan")
    
    # Relationship to sandboxes
    sandboxes = relationship("SandboxDB", back_populates="agent", cascade="all, delete-orphan")



class KnowledgeBaseDB(Base):
    __tablename__ = "knowledge_bases"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    vector_db = Column(String, nullable=False) 
    connection_config = Column(JSON, nullable=False) 
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, default=50)
    embedding_model = Column(String, nullable=False)
    use_ocr = Column(Boolean, default=False)
    
    status = Column(String, default="Draft") 
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
class ChatSessionDB(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    agent_id = Column(String, ForeignKey("agents.AgentId"), nullable=True)
    
    # Store the complex AgentConfig Pydantic model as JSON
    agent_config = Column(JSON, nullable=False)
    
    memory_usage = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    # Establish the one-to-many relationship to messages
    messages = relationship("MessageDB", back_populates="session", cascade="all, delete-orphan")
    
    # Relationship to agent
    agent = relationship("AgentDB", back_populates="sessions")

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

class UserSecretDB(Base):
    __tablename__ = "user_secrets"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False) 
    
    _secret_value = Column("secret_value", String, nullable=False)

    @property
    def secret_value(self):
        # This will now use your robust decryption function
        return decrypt_value(self._secret_value)

    @secret_value.setter
    def secret_value(self, value: str):
        # This will now use your robust encryption function
        self._secret_value = encrypt_value(value)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SandboxDB(Base):
    __tablename__ = "sandboxes"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    agent_id = Column(String, ForeignKey("agents.AgentId"), nullable=True, index=True)
    
    e2b_sandbox_id = Column(String, nullable=False, unique=True, index=True)
    template_id = Column(String, nullable=False)
    state = Column(String, nullable=False, default='running', index=True)  # e.g., 'running', 'paused', 'killed'
    
    meta_info = Column(JSON, nullable=True)
    timeout_seconds = Column(Integer, nullable=False)
    
    started_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    is_active = Column(Boolean, default=True) # For soft deletes
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    agent = relationship("AgentDB", back_populates="sandboxes")
    metrics = relationship("SandboxMetricDB", back_populates="sandbox", cascade="all, delete-orphan")
    events = relationship("SandboxEventDB", back_populates="sandbox", cascade="all, delete-orphan")

class SandboxMetricDB(Base):
    __tablename__ = "sandbox_metrics"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, index=True)
    sandbox_id = Column(String, ForeignKey("sandboxes.id"), nullable=False, index=True)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    cpu_used_pct = Column(Float, nullable=False)
    mem_used_bytes = Column(BigInteger, nullable=False)
    mem_total_bytes = Column(BigInteger, nullable=False)
    disk_used_bytes = Column(BigInteger, nullable=False)
    disk_total_bytes = Column(BigInteger, nullable=False)

    sandbox = relationship("SandboxDB", back_populates="metrics")

class SandboxEventDB(Base):
    __tablename__ = "sandbox_events"
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True, index=True)
    sandbox_id = Column(String, ForeignKey("sandboxes.id"), nullable=False, index=True)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String, nullable=False) # e.g., 'sandbox.lifecycle.created', 'sandbox.lifecycle.killed'
    event_data = Column(JSON, nullable=True)

    sandbox = relationship("SandboxDB", back_populates="events")