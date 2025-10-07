from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from uuid import uuid4

from app.models.session import ChatSession, AgentFramework, AgentConfig, Message
from app.db.models import ChatSessionDB, MessageDB

class SessionService:
    """Service for managing chat sessions with a PostgreSQL database."""

    def create_session(
        self,
        db: Session,
        user_id: str,
        framework: AgentFramework,
        title: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session in the database."""
        session_id = f"session-{uuid4()}"
        
        if not title:
            title = f"New {framework.value.replace('_', ' ').title()} Chat"

        # Default agent config
        default_config = AgentConfig(
            api_key="", model_name="gpt-4o-mini", temperature=0.7,
            top_p=0.9, system_prompt="You are a helpful AI assistant."
        )

        db_session = ChatSessionDB(
            id=session_id,
            user_id=user_id,
            title=title,
            framework=framework.value,
            agent_config=default_config.model_dump()
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        return ChatSession.model_validate(db_session.__dict__)

    def get_session(self, db: Session, user_id: str, session_id: str) -> Optional[ChatSession]:
        """Get a specific session by ID from the database."""
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id,
            ChatSessionDB.user_id == user_id
        ).first()
        
        if db_session:
            return ChatSession.model_validate(db_session, from_attributes=True)
        return None

    def get_user_sessions(self, db: Session, user_id: str) -> List[ChatSession]:
        """Get all sessions for a user from the database."""
        db_sessions = db.query(ChatSessionDB).filter(
            ChatSessionDB.user_id == user_id
        ).order_by(ChatSessionDB.updated_at.desc()).all()
        
        return [ChatSession.model_validate(s, from_attributes=True) for s in db_sessions]

    def update_session(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        **updates
    ) -> Optional[ChatSession]:
        """Update a session in the database."""
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id,
            ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return None

        for key, value in updates.items():
            if key == "agent_config" and isinstance(value, AgentConfig):
                setattr(db_session, key, value.model_dump())
            elif key == "messages":
                # This is more complex; usually handled by add_message
                pass
            elif hasattr(db_session, key):
                setattr(db_session, key, value)
        
        db_session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_session)

        return ChatSession.model_validate(db_session, from_attributes=True)

    def add_message_to_session(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        message: Message
    ) -> Optional[ChatSession]:
        """Add a message to a session."""
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id,
            ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return None

        db_message = MessageDB(
            id=message.id,
            session_id=session_id,
            sender=message.sender,
            text=message.text,
            files=message.files, # Pydantic model dump happens automatically for dict
            timestamp=message.timestamp
        )
        
        db.add(db_message)
        db_session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_session)
        
        return ChatSession.model_validate(db_session, from_attributes=True)

    def delete_session(self, db: Session, user_id: str, session_id: str) -> bool:
        """Delete a session from the database."""
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id,
            ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return False
            
        db.delete(db_session)
        db.commit()
        return True

# Global session service instance
session_service = SessionService()