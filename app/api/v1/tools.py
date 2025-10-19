from fastapi import APIRouter,Depends,HTTPException,status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ToolConfigDB
from app.auth.clerk_auth import get_current_user

from 