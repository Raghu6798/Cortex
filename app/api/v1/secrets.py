import uuid
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import UserSecretDB
from app.auth.clerk_auth import get_current_user
from app.schemas.secrets_schema import SecretCreate, SecretResponse
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1/secrets", tags=["Secrets Management"])

@router.post("/", response_model=SecretResponse, status_code=status.HTTP_201_CREATED)
def create_secret(
    secret_data: SecretCreate, 
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new secret for the authenticated user.
    The secret value is encrypted before being stored in the database.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")

    # Check if a secret with the same name already exists for this user
    existing_secret = db.query(UserSecretDB).filter(
        UserSecretDB.user_id == user_id,
        UserSecretDB.name == secret_data.name
    ).first()
    
    if existing_secret:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A secret with the name '{secret_data.name}' already exists."
        )

    try:
        db_secret = UserSecretDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=secret_data.name,
        )
        # The setter on the model handles encryption automatically
        db_secret.secret_value = secret_data.value
        
        db.add(db_secret)
        db.commit()
        db.refresh(db_secret)
        
        logger.info(f"Secret '{secret_data.name}' created for user '{user_id}'.")
        return db_secret

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create secret for user '{user_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not create secret.")


@router.get("/", response_model=List[SecretResponse])
def list_secrets(
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user)
):
    """
    List all secrets (metadata only) for the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    
    secrets = db.query(UserSecretDB).filter(UserSecretDB.user_id == user_id).order_by(UserSecretDB.name).all()
    return secrets


@router.delete("/{secret_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_secret(
    secret_id: str,
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a specific secret owned by the authenticated user.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    
    secret_to_delete = db.query(UserSecretDB).filter(
        UserSecretDB.id == secret_id,
        UserSecretDB.user_id == user_id
    ).first()
    
    if not secret_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Secret with ID '{secret_id}' not found."
        )
        
    try:
        secret_name = secret_to_delete.name
        db.delete(secret_to_delete)
        db.commit()
        logger.info(f"Secret '{secret_name}' (ID: {secret_id}) deleted for user '{user_id}'.")
        return None # Return None for 204 No Content status
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete secret '{secret_id}' for user '{user_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not delete secret.")