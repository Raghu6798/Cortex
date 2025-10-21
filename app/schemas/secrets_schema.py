from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SecretCreate(BaseModel):
    """Schema for creating a new secret. The value is write-only."""
    name: str = Field(..., description="A unique, user-friendly name for the secret.", example="My NVIDIA API Key")
    value: str = Field(..., description="The sensitive secret value, e.g., an API key or token.")

class SecretResponse(BaseModel):
    """
    Schema for returning secret metadata.
    Crucially, it DOES NOT include the secret value.
    """
    id: str
    user_id: str
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # Allows creating this schema from a SQLAlchemy model