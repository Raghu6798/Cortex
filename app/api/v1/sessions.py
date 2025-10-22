"""
API routes for session management, protected by a custom Clerk JWT dependency.
"""
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session
from app.db.database import get_db

from app.models.session import (
    ChatSession, AgentFramework, SessionCreateRequest, 
    SessionUpdateRequest, SessionsListResponse, Message
)
from app.services.session_service import session_service

# --- Import our verified custom dependency ---
from app.auth.clerk_auth import get_current_user

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])

@router.get("/", response_model=SessionsListResponse)
async def get_user_sessions(
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Get all chat sessions for the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    sessions = session_service.get_user_sessions(db=db, user_id=user_id)
    for session in sessions:
        print(f"DEBUG BACKEND: Sending session '{session.id}' with agent_config tools: {session.agent_config.tools}")
    return SessionsListResponse(sessions=sessions, total=len(sessions))

@router.get("/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Get a specific chat session by ID, ensuring it belongs to the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    session = session_service.get_session(db=db, user_id=user_id, session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or you do not have permission to access it.")
    
    return session

@router.post("/", response_model=ChatSession)
async def create_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Create a new chat session for the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    print(f"--- SESSION API DEBUG ---")
    print(f"Request received: {request}")
    print(f"Agent config in request: {request.agent_config}")
    if request.agent_config:
        print(f"Tools in agent config: {getattr(request.agent_config, 'tools', 'NO TOOLS FIELD')}")
    print("-------------------------")
    
    session = session_service.create_session(
        db=db,
        user_id=user_id,
        framework=request.framework,
        title=request.title,
        agent_config=request.agent_config,
        agent_id=request.agent_id,
    )
    return session

@router.put("/{session_id}", response_model=ChatSession)
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Update an existing chat session, ensuring it belongs to the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided.")
    
    session = session_service.update_session(db=db, user_id=user_id, session_id=session_id, **update_data)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or you do not have permission to modify it.")
    
    return session

@router.post("/{session_id}/messages", response_model=ChatSession)
async def add_message_to_session(
    session_id: str,
    message: Message,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Add a message to a chat session, ensuring it belongs to the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    # First, verify the user has access to this session
    session_check = session_service.get_session(db=db, user_id=user_id, session_id=session_id)
    if not session_check:
        raise HTTPException(status_code=404, detail="Session not found or you do not have permission to add messages to it.")

    updated_session = session_service.add_message_to_session(db=db, user_id=user_id, session_id=session_id, message=message)
    return updated_session

@router.delete("/{session_id}", response_model=Dict)
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Delete a chat session, ensuring it belongs to the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    success = session_service.delete_session(db=db, user_id=user_id, session_id=session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or you do not have permission to delete it.")
    
    return {"message": "Session deleted successfully"}

@router.get("/user/stats", response_model=Dict)
async def get_user_stats(
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user)
):
    """Get session statistics for the authenticated user."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    
    sessions = session_service.get_user_sessions(db=db, user_id=user_id)
    total_sessions = len(sessions)
    total_messages = sum(len(s.messages) for s in sessions)
    
    framework_counts = {}
    for session in sessions:
        framework_val = session.framework.value if isinstance(session.framework, AgentFramework) else session.framework
        framework_counts[framework_val] = framework_counts.get(framework_val, 0) + 1
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "framework_counts": framework_counts
    }