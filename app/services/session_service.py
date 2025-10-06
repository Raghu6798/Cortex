"""
Session service for managing chat sessions.
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.models.session import ChatSession, AgentFramework, AgentConfig, Message

class SessionService:
    """Service for managing chat sessions with file-based storage."""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        """Initialize the session service with storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_sessions_file(self, user_id: str) -> Path:
        """Get the file path for a user's sessions."""
        return self.storage_dir / f"{user_id}_sessions.json"
    
    def _load_user_sessions(self, user_id: str) -> Dict[str, ChatSession]:
        """Load all sessions for a user from storage."""
        sessions_file = self._get_user_sessions_file(user_id)
        if not sessions_file.exists():
            return {}
        
        try:
            with open(sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    session_id: ChatSession(**session_data) 
                    for session_id, session_data in data.items()
                }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading sessions for user {user_id}: {e}")
            return {}
    
    def _save_user_sessions(self, user_id: str, sessions: Dict[str, ChatSession]) -> None:
        """Save all sessions for a user to storage."""
        sessions_file = self._get_user_sessions_file(user_id)
        try:
            # Convert sessions to dict for JSON serialization
            sessions_data = {
                session_id: session.model_dump() 
                for session_id, session in sessions.items()
            }
            
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving sessions for user {user_id}: {e}")
            raise
    
    def create_session(
        self, 
        user_id: str, 
        framework: AgentFramework, 
        title: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None
    ) -> ChatSession:
        """Create a new chat session."""
        session_id = f"session-{datetime.utcnow().timestamp()}"
        
        if not title:
            title = f"New {framework.value.title()} Chat"
        
        if not agent_config:
            # Default configuration
            agent_config = AgentConfig(
                api_key="",
                model_name="gpt-4o-mini",
                temperature=0.7,
                top_p=0.9,
                system_prompt="You are a helpful AI assistant."
            )
        
        new_session = ChatSession(
            id=session_id,
            user_id=user_id,
            title=title,
            framework=framework,
            agent_config=agent_config,
            messages=[],
            memory_usage=0.0
        )
        
        # Load existing sessions and add new one
        sessions = self._load_user_sessions(user_id)
        sessions[session_id] = new_session
        self._save_user_sessions(user_id, sessions)
        
        return new_session
    
    def get_session(self, user_id: str, session_id: str) -> Optional[ChatSession]:
        """Get a specific session by ID."""
        sessions = self._load_user_sessions(user_id)
        return sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """Get all sessions for a user."""
        sessions = self._load_user_sessions(user_id)
        # Return sessions sorted by updated_at (newest first)
        return sorted(
            sessions.values(), 
            key=lambda s: s.updated_at, 
            reverse=True
        )
    
    def update_session(
        self, 
        user_id: str, 
        session_id: str, 
        **updates
    ) -> Optional[ChatSession]:
        """Update a session with new data."""
        sessions = self._load_user_sessions(user_id)
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        # Always update the timestamp
        session.updated_at = datetime.utcnow()
        
        # Save back to storage
        self._save_user_sessions(user_id, sessions)
        
        return session
    
    def add_message_to_session(
        self, 
        user_id: str, 
        session_id: str, 
        message: Message
    ) -> Optional[ChatSession]:
        """Add a message to a session."""
        sessions = self._load_user_sessions(user_id)
        if session_id not in sessions:
            return None
        
        session = sessions[session_id]
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        # Update memory usage (simple estimation)
        session.memory_usage = min(100.0, len(session.messages) * 2.5)
        
        self._save_user_sessions(user_id, sessions)
        return session
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a session."""
        sessions = self._load_user_sessions(user_id)
        if session_id not in sessions:
            return False
        
        del sessions[session_id]
        self._save_user_sessions(user_id, sessions)
        return True
    
    def get_session_count(self, user_id: str) -> int:
        """Get the total number of sessions for a user."""
        sessions = self._load_user_sessions(user_id)
        return len(sessions)

# Global session service instance
session_service = SessionService()
