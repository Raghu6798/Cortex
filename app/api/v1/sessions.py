"""
API routes for session management.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_clerk_auth import HTTPAuthorizationCredentials

from app.models.session import (
    ChatSession, AgentFramework, SessionCreateRequest, 
    SessionUpdateRequest, SessionsListResponse, Message
)
from app.services.session_service import session_service
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer

# -------------------- Clerk Auth Setup --------------------
clerk_config = ClerkConfig(
    jwks_url="https://supreme-caribou-95.clerk.accounts.dev/.well-known/jwks.json",
    auto_error=True
)
clerk_auth_guard = ClerkHTTPBearer(config=clerk_config, add_state=True)

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.get("/", response_model=SessionsListResponse)
async def get_user_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Get all chat sessions for the authenticated user."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    sessions = session_service.get_user_sessions(user_id)
    return SessionsListResponse(sessions=sessions, total=len(sessions))

@router.get("/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Get a specific chat session by ID."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    session = session_service.get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@router.post("/", response_model=ChatSession)
async def create_session(
    request: SessionCreateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Create a new chat session."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    session = session_service.create_session(
        user_id=user_id,
        framework=request.framework,
        title=request.title
    )
    
    return session

@router.put("/{session_id}", response_model=ChatSession)
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Update an existing chat session."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    # Prepare update data
    update_data = {}
    if request.title is not None:
        update_data["title"] = request.title
    if request.agent_config is not None:
        update_data["agent_config"] = request.agent_config
    if request.messages is not None:
        update_data["messages"] = request.messages
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    session = session_service.update_session(user_id, session_id, **update_data)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@router.post("/{session_id}/messages", response_model=ChatSession)
async def add_message_to_session(
    session_id: str,
    message: Message,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Add a message to a chat session."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    session = session_service.add_message_to_session(user_id, session_id, message)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Delete a chat session."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    success = session_service.delete_session(user_id, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}

@router.get("/user/stats")
async def get_user_stats(
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
):
    """Get statistics for the authenticated user."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    sessions = session_service.get_user_sessions(user_id)
    total_sessions = len(sessions)
    total_messages = sum(len(session.messages) for session in sessions)
    
    # Count by framework
    framework_counts = {}
    for session in sessions:
        framework = session.framework.value
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "framework_counts": framework_counts
    }
