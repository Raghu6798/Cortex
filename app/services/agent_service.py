# app/services/agent_service.py

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from fastapi import HTTPException, status


from app.db.models import AgentDB, ChatSessionDB
from app.schemas.agents_schema import AgentCreate, AgentResponse, AgentUpdate

class AgentService:
    """Service class for handling all business logic related to Agents."""

    def _map_agent_to_response(self, db: Session, agent: AgentDB) -> AgentResponse:
        """Helper to map AgentDB object to AgentResponse schema and calculate session count."""
        # Calculate session count (which was done in every handler)
        session_count = db.query(ChatSessionDB).filter(
            ChatSessionDB.agent_id == agent.AgentId,
            ChatSessionDB.is_active == True # Assuming only active sessions count
        ).count()
        
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

    def create_agent(self, db: Session, user_id: str, agent_data: AgentCreate) -> AgentResponse:
        """Create a new agent for the user."""
        try:
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
            
            return self._map_agent_to_response(db, agent)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent: {str(e)}")

    def get_user_agents(self, db: Session, user_id: str, page: int = 1, limit: int = 10) -> List[AgentResponse]:
        """Get all active agents for a user with pagination."""
        try:
            offset = (page - 1) * limit
            
            agents = db.query(AgentDB).filter(
                AgentDB.user_id == user_id,
                AgentDB.is_active == True
            ).offset(offset).limit(limit).all()
            
            return [self._map_agent_to_response(db, agent) for agent in agents]
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get agents: {str(e)}")

    def get_agent(self, db: Session, user_id: str, agent_id: str) -> AgentResponse:
        """Get a specific active agent by ID for the user."""
        try:
            agent = db.query(AgentDB).filter(
                AgentDB.AgentId == agent_id,
                AgentDB.user_id == user_id,
                AgentDB.is_active == True
            ).first()
            
            if not agent:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
            
            return self._map_agent_to_response(db, agent)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get agent: {str(e)}")


    def update_agent(self, db: Session, user_id: str, agent_id: str, agent_update: AgentUpdate) -> AgentResponse:
        """Update an agent for the user."""
        try:
            agent = db.query(AgentDB).filter(
                AgentDB.AgentId == agent_id,
                AgentDB.user_id == user_id,
                AgentDB.is_active == True
            ).with_for_update().first() # Use with_for_update for explicit locking on update
            
            if not agent:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
            
            # Update fields if provided using Pydantic's model_dump/dict and exclude_unset
            update_data = agent_update.model_dump(exclude_unset=True) if hasattr(agent_update, 'model_dump') else agent_update.dict(exclude_unset=True)
            
            for key, value in update_data.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            agent.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(agent)
            
            return self._map_agent_to_response(db, agent)
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update agent: {str(e)}")

    def delete_agent(self, db: Session, user_id: str, agent_id: str) -> Dict[str, str]:
        """Soft delete an agent for the user."""
        try:
            agent = db.query(AgentDB).filter(
                AgentDB.AgentId == agent_id,
                AgentDB.user_id == user_id,
                AgentDB.is_active == True
            ).with_for_update().first()
            
            if not agent:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
            
            # Soft delete
            agent.is_active = False
            agent.updated_at = datetime.utcnow()
            
            db.commit()
            
            return {"message": "Agent deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete agent: {str(e)}")

    def get_agent_sessions(self, db: Session, user_id: str, agent_id: str) -> List[Dict[str, Any]]:
        """Get all active chat sessions for a specific agent belonging to the user."""
        try:
            # 1. Verify agent existence and ownership
            agent = db.query(AgentDB).filter(
                AgentDB.AgentId == agent_id,
                AgentDB.user_id == user_id,
                AgentDB.is_active == True
            ).first()
            
            if not agent:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
            
            # 2. Fetch sessions
            sessions = db.query(ChatSessionDB).filter(
                ChatSessionDB.agent_id == agent_id,
                ChatSessionDB.is_active == True
            ).order_by(ChatSessionDB.updated_at.desc()).all()
            
            # 3. Transform data (assuming 'messages' is a relationship on ChatSessionDB)
            return [
                {
                    "id": session.id,
                    "title": session.title,
                    "framework": session.framework,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "message_count": len(getattr(session, 'messages', []))
                }
                for session in sessions
            ]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get agent sessions: {str(e)}")

agent_service = AgentService()