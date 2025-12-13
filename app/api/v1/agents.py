# app/api/v1/agents.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import AgentDB, ChatSessionDB
from app.schemas.api_schemas import HealthStatus
from app.auth.clerk_auth import get_current_user
from app.schemas.agents_schema import AgentCreate, AgentResponse, AgentUpdate

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

@router.post("", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate,token_payload: str = Depends(get_current_user),db: Session = Depends(get_db)):
    """Create a new agent for the user"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        agent = AgentDB(
            AgentId=str(uuid.uuid4()),
            user_id=user_id,
            name=agent_data.name,
            description=agent_data.description,
            architecture=agent_data.architecture,
            framework=agent_data.framework,
            settings=agent_data.settings,
            tools=agent_data.tools
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)
        
        # Get session count
        session_count = db.query(ChatSessionDB).filter(ChatSessionDB.agent_id == agent.AgentId).count()
        
        return AgentResponse(
            id=agent.AgentId,
            user_id=agent.user_id,
            name=agent.name,
            description=agent.description,
            architecture=agent.architecture,
            framework=agent.framework,
            settings=agent.settings,
            tools=agent.tools,
            is_active=agent.is_active,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            session_count=session_count
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@router.get("", response_model=List[AgentResponse])
async def get_user_agents(token_payload: str = Depends(get_current_user),db: Session = Depends(get_db),page: int = 1,limit: int = 10):
    """Get all agents for a user with pagination"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Get total count for pagination info
        total_count = db.query(AgentDB).filter(
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).count()
        
        # Get paginated agents
        agents = db.query(AgentDB).filter(
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).offset(offset).limit(limit).all()
        
        result = []
        for agent in agents:
            session_count = db.query(ChatSessionDB).filter(ChatSessionDB.agent_id == agent.AgentId).count()
            result.append(AgentResponse(
                id=agent.AgentId,
                user_id=agent.user_id,
                name=agent.name,
                description=agent.description,
                architecture=agent.architecture,
                framework=agent.framework,
                settings=agent.settings,
                tools=agent.tools,
                is_active=agent.is_active,
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                session_count=session_count
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str,token_payload: str = Depends(get_current_user),db: Session = Depends(get_db)):
    """Get a specific agent by ID"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        agent = db.query(AgentDB).filter(
            AgentDB.AgentId == agent_id,
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        session_count = db.query(ChatSessionDB).filter(ChatSessionDB.agent_id == agent.AgentId).count()
        
        return AgentResponse(
            id=agent.AgentId,
            user_id=agent.user_id,
            name=agent.name,
            description=agent.description,
            architecture=agent.architecture,
            framework=agent.framework,
            settings=agent.settings,
            tools=agent.tools,
            is_active=agent.is_active,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            session_count=session_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str,agent_update: AgentUpdate,token_payload: str = Depends(get_current_user),db: Session = Depends(get_db)):
    """Update an agent"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        agent = db.query(AgentDB).filter(
            AgentDB.AgentId == agent_id,
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields if provided
        if agent_update.name is not None:
            agent.name = agent_update.name
        if agent_update.description is not None:
            agent.description = agent_update.description
        if agent_update.settings is not None:
            agent.settings = agent_update.settings
        if agent_update.tools is not None:
            agent.tools = agent_update.tools
        
        agent.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(agent)
        
        session_count = db.query(ChatSessionDB).filter(ChatSessionDB.agent_id == agent.AgentId).count()
        
        return AgentResponse(
            id=agent.AgentId,
            user_id=agent.user_id,
            name=agent.name,
            description=agent.description,
            architecture=agent.architecture,
            framework=agent.framework,
            settings=agent.settings,
            tools=agent.tools,
            is_active=agent.is_active,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            session_count=session_count
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str,token_payload: str = Depends(get_current_user),db: Session = Depends(get_db)):
    """Delete an agent (soft delete)"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        agent = db.query(AgentDB).filter(
            AgentDB.AgentId == agent_id,
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Soft delete
        agent.is_active = False
        agent.updated_at = datetime.utcnow()
        
        db.commit()
        
        return {"message": "Agent deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@router.get("/{agent_id}/sessions")
async def get_agent_sessions(agent_id: str,token_payload: str = Depends(get_current_user),db: Session = Depends(get_db)):
    """Get all chat sessions for a specific agent"""
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=403, detail="User ID not found in token.")
        
        # Verify agent belongs to user
        agent = db.query(AgentDB).filter(
            AgentDB.AgentId == agent_id,
            AgentDB.user_id == user_id,
            AgentDB.is_active == True
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        sessions = db.query(ChatSessionDB).filter(
            ChatSessionDB.agent_id == agent_id,
            ChatSessionDB.is_active == True
        ).order_by(ChatSessionDB.updated_at.desc()).all()
        
        return [
            {
                "id": session.id,
                "title": session.title,
                "framework": session.framework,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages)
            }
            for session in sessions
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent sessions: {str(e)}")
