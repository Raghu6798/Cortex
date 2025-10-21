"""
Session models for storing chat sessions and agent configurations.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class AgentFramework(str, Enum):
    """Supported agent frameworks."""
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    PYDANTIC_AI = "pydantic_ai"
    LANGGRAPH = "langgraph"
    ADK = "adk"

class Message(BaseModel):
    """Individual chat message."""
    id: str
    sender: str = Field(..., pattern="^(user|agent)$")
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    files: Optional[List[Dict[str, Any]]] = None

class AgentConfig(BaseModel):
    """Agent configuration settings."""
    api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    system_prompt: str = "You are a helpful AI assistant."
    base_url: Optional[str] = None
    provider: Optional[str] = None
    max_tokens: Optional[int] = Field(default=512, ge=1)

class ChatSession(BaseModel):
    """Complete chat session with agent configuration."""
    id: str
    user_id: str
    title: str
    framework: AgentFramework
    agent_config: AgentConfig
    messages: List[Message] = []
    memory_usage: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class SessionCreateRequest(BaseModel):
    """Request model for creating a new session."""
    framework: AgentFramework
    title: Optional[str] = None
    agent_config: Optional[AgentConfig] = None
    agent_id: Optional[str] = None

class SessionUpdateRequest(BaseModel):
    """Request model for updating a session."""
    title: Optional[str] = None
    agent_config: Optional[AgentConfig] = None
    messages: Optional[List[Message]] = None
    agent_id: Optional[str] = None
    
class SessionResponse(BaseModel):
    """Response model for session data."""
    session: ChatSession

class SessionsListResponse(BaseModel):
    """Response model for list of sessions."""
    sessions: List[ChatSession]
    total: int
