import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from e2b_code_interpreter import Sandbox as E2BSandbox # E2B SDK

from app.db.database import get_db
from app.db.models import SandboxDB, AgentDB
from app.auth.clerk_auth import get_current_user
from app.schemas.sandbox_schema import SandboxCreate, SandboxResponse, SandboxTimeoutUpdate
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1/sandboxes", tags=["Sandboxes"])

@router.post("/", response_model=SandboxResponse, status_code=status.HTTP_201_CREATED)
async def create_sandbox(
    sandbox_data: SandboxCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create and persist a new E2B code sandbox."""
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")

    # If agent_id is provided, verify it belongs to the user
    if sandbox_data.agent_id:
        agent = db.query(AgentDB).filter(AgentDB.AgentId == sandbox_data.agent_id, AgentDB.user_id == user_id).first()
        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found.")

    try:
        # TODO: Implement actual E2B SDK call here
        logger.info(f"Creating E2B sandbox for user {user_id} with timeout {sandbox_data.timeout_seconds}s")
        # e2b_instance = await E2BSandbox.create(
        #     template=sandbox_data.template_id,
        #     timeout=sandbox_data.timeout_seconds,
        #     metadata=sandbox_data.metadata
        # )
        # MOCKING the E2B response for now
        e2b_instance = type('obj', (object,), {
            'sandbox_id': f'mock_e2b_{uuid.uuid4()}',
            'template_id': sandbox_data.template_id,
            'metadata': sandbox_data.metadata
        })()
        logger.info(f"E2B sandbox created with ID: {e2b_instance.sandbox_id}")
        
        start_time = datetime.now(timezone.utc)
        expire_time = start_time + timedelta(seconds=sandbox_data.timeout_seconds)

        db_sandbox = SandboxDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            agent_id=sandbox_data.agent_id,
            e2b_sandbox_id=e2b_instance.sandbox_id,
            template_id=e2b_instance.template_id,
            state='running',
            meta_info=e2b_instance.metadata,
            timeout_seconds=sandbox_data.timeout_seconds,
            started_at=start_time,
            expires_at=expire_time
        )

        db.add(db_sandbox)
        db.commit()
        db.refresh(db_sandbox)

        return db_sandbox
    except Exception as e:
        logger.error(f"Failed to create sandbox for user {user_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not create sandbox: {str(e)}")


@router.get("/", response_model=List[SandboxResponse])
async def list_sandboxes(
    agent_id: Optional[str] = None,
    state: Optional[str] = None, # 'running', 'paused', 'killed'
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all active sandboxes for the current user, with optional filters."""
    user_id = current_user.get("sub")
    query = db.query(SandboxDB).filter(SandboxDB.user_id == user_id, SandboxDB.is_active == True)

    if agent_id:
        query = query.filter(SandboxDB.agent_id == agent_id)
    if state:
        query = query.filter(SandboxDB.state == state)
    
    sandboxes = query.order_by(SandboxDB.created_at.desc()).all()
    return sandboxes


@router.get("/{sandbox_id}", response_model=SandboxResponse)
async def get_sandbox(
    sandbox_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information for a specific sandbox, including metrics and events."""
    user_id = current_user.get("sub")
    
    sandbox = db.query(SandboxDB).options(
        joinedload(SandboxDB.metrics),
        joinedload(SandboxDB.events)
    ).filter(
        SandboxDB.id == sandbox_id, 
        SandboxDB.user_id == user_id
    ).first()

    if not sandbox:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sandbox not found.")
    
    # TODO: Here you could also make a live call to e2b_instance.get_info() or .get_metrics()
    # to supplement the stored data with real-time information if needed.
    
    return sandbox


@router.patch("/{sandbox_id}/timeout", response_model=SandboxResponse)
async def set_sandbox_timeout(
    sandbox_id: str,
    timeout_update: SandboxTimeoutUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update the timeout for a running sandbox."""
    user_id = current_user.get("sub")
    db_sandbox = db.query(SandboxDB).filter(SandboxDB.id == sandbox_id, SandboxDB.user_id == user_id).first()

    if not db_sandbox:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sandbox not found.")
    if db_sandbox.state != 'running':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Can only update timeout for running sandboxes.")

    try:
        # TODO: Implement actual E2B SDK call
        # e2b_instance = await E2BSandbox.connect(db_sandbox.e2b_sandbox_id)
        # await e2b_instance.set_timeout(timeout_update.timeout_seconds)
        
        db_sandbox.timeout_seconds = timeout_update.timeout_seconds
        db_sandbox.expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_update.timeout_seconds)
        db.commit()
        db.refresh(db_sandbox)

        return db_sandbox
    except Exception as e:
        logger.error(f"Failed to update timeout for sandbox {db_sandbox.e2b_sandbox_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update sandbox timeout: {str(e)}")

@router.delete("/{sandbox_id}", status_code=status.HTTP_204_NO_CONTENT)
async def terminate_sandbox(
    sandbox_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Terminate (kill) a sandbox and mark it as inactive."""
    user_id = current_user.get("sub")
    db_sandbox = db.query(SandboxDB).filter(SandboxDB.id == sandbox_id, SandboxDB.user_id == user_id).first()

    if not db_sandbox:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sandbox not found.")

    try:
        # TODO: Implement actual E2B SDK call to kill the instance
        # if db_sandbox.state == 'running':
        #     e2b_instance = await E2BSandbox.connect(db_sandbox.e2b_sandbox_id)
        #     await e2b_instance.kill()

        # Soft delete
        db_sandbox.is_active = False
        db_sandbox.state = 'killed'
        db.commit()

        return None
    except Exception as e:
        logger.error(f"Failed to terminate sandbox {db_sandbox.e2b_sandbox_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to terminate sandbox: {str(e)}")