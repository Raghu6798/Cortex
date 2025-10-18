from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from typing import Any

class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    architecture: str  # 'mono' or 'multi'
    framework: str
    settings: Dict[str, Any]
    tools: Optional[List[Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str]
    architecture: str
    framework: str
    settings: dict
    tools: Optional[List[Dict[str, Any]]]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    session_count: int = 0

    class Config:
        from_attributes = True

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None