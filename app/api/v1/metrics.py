from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session
from app.db.database import get_db
from app.auth.clerk_auth import get_current_user
from app.services.session_service import session_service


router = APIRouter(prefix="/metrics", tags=["Metrics"])


