
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Base & Create Schemas ---

class SandboxBase(BaseModel):
    template_id: str = Field("base", description="The E2B template to use for the sandbox.")
    timeout_seconds: int = Field(300, description="The default lifetime of the sandbox in seconds.")
    metadata: Optional[Dict[str, str]] = Field(None, description="User-defined metadata for the sandbox.")

class SandboxCreate(SandboxBase):
    agent_id: Optional[str] = Field(None, description="Optional agent to associate with this sandbox.")

class SandboxTimeoutUpdate(BaseModel):
    timeout_seconds: int = Field(..., gt=0, description="The new timeout for the sandbox in seconds from now.")

# --- Response Schemas ---

class SandboxMetricResponse(BaseModel):
    timestamp: datetime
    cpu_used_pct: float
    mem_used_bytes: int
    mem_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int

    class Config:
        from_attributes = True

class SandboxEventResponse(BaseModel):
    timestamp: datetime
    event_type: str
    event_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class SandboxResponse(BaseModel):
    id: str
    user_id: str
    agent_id: Optional[str]
    e2b_sandbox_id: str
    template_id: str
    state: str
    metadata: Optional[Dict[str, str]]
    started_at: datetime
    expires_at: datetime
    timeout_seconds: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    metrics: List[SandboxMetricResponse] = []
    events: List[SandboxEventResponse] = []

    class Config:
        from_attributes = True