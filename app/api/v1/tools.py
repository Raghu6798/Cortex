from fastapi import APIRouter,Depends,HTTPException,status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ToolConfigDB
from app.auth.clerk_auth import get_current_user

from app.schemas.ToolCallSchema import ToolConfigSchema
from app.schemas.ToolCallSchema import ToolCallSchema
from app.schemas.ToolCallSchema import ToolCallEvent
from app.schemas.ToolCallSchema import ToolCallResponse
from app.utils.logger import logger

router = APIRouter(
    prefix="/api/v1/tools",
    tags=["tools"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=ToolCallResponse)
async def create_tool_config(
    tool_config: ToolConfigSchema,
    agent_id: str,
    token_payload: dict = Depends(get_current_user),    
    db: Session = Depends(get_db),
):
    """Create a new tool config"""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    tool_config = ToolConfigDB(**tool_config.model_dump(), user_id=user_id, agent_id=agent_id)
    db.add(tool_config)
    db.commit()
    db.refresh(tool_config)
    return ToolCallResponse(tool_calls=[ToolCallSchema(tool_name=tool_config.name, tool_args=tool_config.request_payload, tool_result=tool_config.response_payload)])

@router.get("/", response_model=List[ToolConfigSchema])
async def get_tool_configs(
    agent_id: str,
    token_payload: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all tool configs for an agent"""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="User ID not found in token.")
    tool_configs = db.query(ToolConfigDB).filter(ToolConfigDB.user_id == user_id, ToolConfigDB.agent_id == agent_id).all()
    return tool_configs

    