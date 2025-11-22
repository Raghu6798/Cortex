# from datetime import datetime

# def sandbox_to_dict(sb):
#     return {
#         "sandbox_id": sb.sandbox_id,
#         "template_id": sb.template_id,
#         "name": sb.name,
#         "metadata": sb.metadata,
#         "started_at": sb.started_at.isoformat() if isinstance(sb.started_at, datetime) else sb.started_at,
#         "end_at": sb.end_at.isoformat() if isinstance(sb.end_at, datetime) else sb.end_at,
#         "state": sb.state.value if hasattr(sb.state, "value") else sb.state,
#         "cpu_count": sb.cpu_count,
#         "memory_mb": sb.memory_mb,
#         "envd_version": sb.envd_version,
#     }

# def extract_all_sandbox_metadata(paginator):
#     sandboxes = paginator.next_items()
#     return [sandbox_to_dict(sb) for sb in sandboxes]

# def print_paginator_details(pg):
#     print("\n--- Paginator Metadata ---")
#     for attr in dir(pg):
#         if attr.startswith("_"):
#             continue
#         try:
#             print(f"{attr}: {getattr(pg, attr)}")
#         except:
#             print(f"{attr}: <unreadable>")

#     print("\n--- Sandbox Items Metadata ---")
#     items = pg.next_items()
#     for i, item in enumerate(items):
#         print(f"\nSandbox #{i+1}")
#         try:
#             print(item.__dict__)
#         except:
#             import inspect
#             print(inspect.getmembers(item))


# def extract_metadata(obj):
#     metadata = {}
#     for attr in dir(obj):
#         if attr.startswith("__"):
#             continue
#         try:
#             value = getattr(obj, attr)
#             metadata[attr] = value
#         except Exception:
#             metadata[attr] = "<unreadable>"
#     return metadata

# paginator_metadata = extract_metadata(paginator)

# def extract_all_sandbox_metadata(paginator):
#     sandboxes = paginator.next_items()
#     return [sandbox_to_dict(sb) for sb in sandboxes]

# paginator = Sandbox.list()
# metadata = extract_all_sandbox_metadata(paginator)
# print(json.dumps(metadata, indent=4))

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session, joinedload
from e2b_code_interpreter import Sandbox as E2BSandbox, SandboxInfo

from app.db.models import SandboxDB
from app.schemas.sandbox_schema import SandboxCreate
from app.utils.logger import logger

class SandboxService:
    """Service layer for all sandbox-related business logic, interacting with E2B SDK and the database."""

    def _get_db_sandbox(self, db: Session, user_id: str, sandbox_id: str) -> Optional[SandboxDB]:
        """Helper to fetch a sandbox by its internal ID, ensuring user ownership."""
        return db.query(SandboxDB).filter(
            SandboxDB.id == sandbox_id,
            SandboxDB.user_id == user_id,
            SandboxDB.is_active == True
        ).first()

    async def create_sandbox(
        self, db: Session, user_id: str, sandbox_data: SandboxCreate
    ) -> SandboxDB:
        """Calls the E2B SDK to create a sandbox and persists its metadata to the database."""
        logger.info(f"Initiating E2B sandbox creation for user '{user_id}' with template '{sandbox_data.template_id}'.")
        
        try:
            # --- REAL E2B SDK Integration ---
            e2b_instance = await E2BSandbox.create(
                template=sandbox_data.template_id,
                timeout=sandbox_data.timeout_seconds,
                metadata=sandbox_data.metadata
            )
            logger.info(f"Successfully created E2B sandbox with ID: {e2b_instance.sandbox_id}")
            
            # Use get_info() to get accurate start/end times from the E2B service
            info = await e2b_instance.get_info()
            
        except Exception as e:
            logger.error(f"E2B SDK sandbox creation failed: {e}", exc_info=True)
            raise  # Re-raise the exception to be caught by the API layer

        db_sandbox = SandboxDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            agent_id=sandbox_data.agent_id,
            e2b_sandbox_id=info.sandbox_id,
            template_id=info.template_id,
            state=info.state,
            metadata=info.metadata,
            timeout_seconds=sandbox_data.timeout_seconds,
            started_at=info.started_at,
            expires_at=info.end_at
        )

        db.add(db_sandbox)
        # The API route that calls this will be responsible for the commit.
        return db_sandbox

    async def list_live_sandboxes(self) -> List[SandboxInfo]:
        """
        Lists all currently running or paused sandboxes directly from the E2B SDK.
        Note: This lists sandboxes for the entire team, not just a single user.
        The API layer should filter this based on what's in our DB if user-scoping is needed.
        """
        try:
            paginator = E2BSandbox.list()
            # For simplicity in V1, we fetch all items. For production, implement full pagination.
            sandboxes: List[SandboxInfo] = []
            while paginator.has_next:
                items = await paginator.next_items()
                sandboxes.extend(items)
            return sandboxes
        except Exception as e:
            logger.error(f"Failed to list live sandboxes from E2B SDK: {e}", exc_info=True)
            return []

    def get_sandbox_details(self, db: Session, user_id: str, sandbox_id: str) -> Optional[SandboxDB]:
        """Retrieves a single persisted sandbox with its relations."""
        return db.query(SandboxDB).options(
            joinedload(SandboxDB.metrics),
            joinedload(SandboxDB.events)
        ).filter(
            SandboxDB.id == sandbox_id, 
            SandboxDB.user_id == user_id
        ).first()

    async def update_sandbox_timeout(
        self, db: Session, user_id: str, sandbox_id: str, new_timeout_seconds: int
    ) -> Optional[SandboxDB]:
        """Updates the timeout for a running sandbox in E2B and the local DB."""
        db_sandbox = self._get_db_sandbox(db, user_id, sandbox_id)

        if not db_sandbox:
            return None
        if db_sandbox.state != 'running':
            raise ValueError("Can only update timeout for a 'running' sandbox.")

        try:
            # --- REAL E2B SDK Integration ---
            e2b_instance = await E2BSandbox.connect(db_sandbox.e2b_sandbox_id)
            await e2b_instance.set_timeout(new_timeout_seconds)
            logger.info(f"Successfully updated E2B sandbox '{db_sandbox.e2b_sandbox_id}' timeout to {new_timeout_seconds}s.")
        except Exception as e:
            logger.error(f"E2B SDK set_timeout failed for sandbox '{db_sandbox.e2b_sandbox_id}': {e}", exc_info=True)
            raise
        
        db_sandbox.timeout_seconds = new_timeout_seconds
        db_sandbox.expires_at = datetime.now(timezone.utc) + timedelta(seconds=new_timeout_seconds)
        
        return db_sandbox

    async def terminate_sandbox(self, db: Session, user_id: str, sandbox_id: str) -> bool:
        """Terminates a sandbox via the E2B SDK and marks it as inactive in the DB."""
        db_sandbox = self._get_db_sandbox(db, user_id, sandbox_id)

        if not db_sandbox:
            return False

        try:
            # --- REAL E2B SDK Integration ---
            if db_sandbox.state == 'running':
                logger.info(f"Attempting to kill E2B sandbox '{db_sandbox.e2b_sandbox_id}'...")
                e2b_instance = await E2BSandbox.connect(db_sandbox.e2b_sandbox_id)
                await e2b_instance.kill()
                logger.info(f"Successfully killed E2B sandbox '{db_sandbox.e2b_sandbox_id}'.")
        except Exception as e:
            logger.error(f"E2B SDK kill failed for sandbox '{db_sandbox.e2b_sandbox_id}': {e}", exc_info=True)
            # Decide if you want to proceed with DB update even if SDK fails. It's often safer to do so.
        
        db_sandbox.is_active = False
        db_sandbox.state = 'killed'
        db_sandbox.expires_at = datetime.now(timezone.utc)
        
        return True

# Create a singleton instance of the service
sandbox_service = SandboxService()