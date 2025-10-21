# FINAL CORRECTED FILE: app/services/session_service.py

from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime, timezone 

from app.models.session import ChatSession, AgentFramework, AgentConfig, Message
from app.db.models import ChatSessionDB, MessageDB

class SessionService:
    """Service for managing chat sessions with a PostgreSQL database."""

    def create_session(
        self, db: Session, user_id: str, framework: AgentFramework, title: Optional[str] = None, agent_config: Optional[AgentConfig] = None, agent_id: Optional[str] = None
    ) -> ChatSession:
        session_id = f"session-{uuid4()}"
        
        if not title:
            title = f"New {framework.value.replace('_', ' ').title()} Chat"

        # Use provided agent config or default
        config_to_store = agent_config if agent_config else AgentConfig(api_key="", model_name="gpt-4o-mini")
        current_utc_time = datetime.now(timezone.utc)
        db_session = ChatSessionDB(
            id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            title=title,
            framework=framework.value,
            agent_config=config_to_store.model_dump(),
            created_at=current_utc_time, 
            updated_at=current_utc_time  
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        print("--- AFTER DB REFRESH ---")
        print(f"ID: {db_session.id}")
        print(f"Created At: {db_session.created_at}")
        print(f"Updated At: {db_session.updated_at}")
        print("------------------------")
        return ChatSession.model_validate(db_session, from_attributes=True)

    def get_session(self, db: Session, user_id: str, session_id: str) -> Optional[ChatSession]:
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id, ChatSessionDB.user_id == user_id
        ).first()
        
        # CORRECTED USAGE: Also use from_attributes here
        return ChatSession.model_validate(db_session, from_attributes=True) if db_session else None

    def get_user_sessions(self, db: Session, user_id: str) -> List[ChatSession]:
        db_sessions = db.query(ChatSessionDB).filter(
            ChatSessionDB.user_id == user_id
        ).order_by(ChatSessionDB.updated_at.desc()).all()
        
        # CORRECTED USAGE: And here for lists
        return [ChatSession.model_validate(s, from_attributes=True) for s in db_sessions]

    def update_session(
        self, db: Session, user_id: str, session_id: str, **updates
    ) -> Optional[ChatSession]:
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id, ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return None
        
        # ... (update logic is fine)
        for key, value in updates.items():
            setattr(db_session, key, value)
            
        db_session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(db_session)

        # CORRECTED USAGE: And here
        return ChatSession.model_validate(db_session, from_attributes=True)

    def add_message_to_session(
        self, db: Session, user_id: str, session_id: str, message: Message
    ) -> Optional[ChatSession]:
        # First, ensure the session exists and belongs to the user
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id, ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return None

        db_message = MessageDB(**message.model_dump(), session_id=session_id)
        
        db.add(db_message)
        db.commit()
        db.refresh(db_session)
        
        # CORRECTED USAGE: And finally, here
        return ChatSession.model_validate(db_session, from_attributes=True)

    def delete_session(self, db: Session, user_id: str, session_id: str) -> bool:
        db_session = db.query(ChatSessionDB).filter(
            ChatSessionDB.id == session_id, ChatSessionDB.user_id == user_id
        ).first()

        if not db_session:
            return False
            
        db.delete(db_session)
        db.commit()
        return True

session_service = SessionService()