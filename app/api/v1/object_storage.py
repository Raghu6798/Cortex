# app/api/v1/object_storage.py

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status,Query
from fastapi.responses import RedirectResponse
from typing import Dict, Any, List

from app.auth.clerk_auth import get_current_user
from app.services.file_service import FileService, file_service 

router = APIRouter(prefix="/api/v1/object-storage", tags=["Object Storage"])

# --- 1. UPLOAD ENDPOINT (Multipart Upload) ---
@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(..., description="The file to upload (e.g., .pdf, .docx, .png)."),
    use_s3: bool = Query(False, description="Set to true to upload to AWS S3, false for MinIO"),
    token_payload: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Uploads a file to the configured object storage (MinIO).
    The file is stored securely under a unique path indexed by the user ID.
    """
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Failed to get user ID: {str(e)}")
    try:
        # Delegate core logic to the FileService
        return await file_service.upload_file(user_id=user_id, file=file, use_s3=use_s3)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}")

# --- 2. LIST ENDPOINT ---
@router.get("/list", response_model=List[Dict[str, Any]])
async def list_files(
    token_payload: dict = Depends(get_current_user),
):
    """
    Lists all files uploaded by the authenticated user.
    Returns metadata (name, size, object_name) but NOT the file content.
    """
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Failed to get user ID: {str(e)}")

    try:
        # Delegate listing to the service
        return await file_service.list_user_files(user_id=user_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="File listing service is not yet fully implemented."
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list files: {str(e)}")


# --- 3. DOWNLOAD ENDPOINT (Redirect to Presigned URL) ---
@router.get("/download")
async def download_file(
    object_name: str,
    token_payload: dict = Depends(get_current_user),
):
    """
    Generates a secure, temporary pre-signed URL to download a file.
    
    Args:
        object_name: The full path of the object in storage 
                     (e.g., user-id/files/uuid/filename.ext).
    """
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Failed to get user ID: {str(e)}")

    # Basic security check: ensure the object_name belongs to the user
    if not object_name.startswith(f"{user_id}/files/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found or access denied.")

    try:
        # Delegate secure URL generation to the service
        presigned_url = await file_service.create_presigned_download_url(
            user_id=user_id, 
            object_name=object_name
        )
        
        # Redirect the client to the MinIO/S3 pre-signed URL
        return RedirectResponse(url=presigned_url, status_code=status.HTTP_302_FOUND)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Presigned URL generation is not yet fully implemented."
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create download link: {str(e)}")


# --- 4. DELETE ENDPOINT ---
@router.delete("/delete")
async def delete_file(
    object_name: str,
    token_payload: dict = Depends(get_current_user),
) -> Dict[str, str]:
    """
    Deletes a file from the object storage.
    
    Args:
        object_name: The full path of the object in storage 
                     (e.g., user-id/files/uuid/filename.ext).
    """
    try:
        user_id = token_payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User ID not found in token.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Failed to get user ID: {str(e)}")

    # Basic security check: ensure the object_name belongs to the user
    if not object_name.startswith(f"{user_id}/files/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found or access denied.")

    try:
        # Delegate the deletion logic to the service
        success = await file_service.delete_file(user_id=user_id, object_name=object_name)
        
        if success:
            return {"message": f"Object '{object_name}' deleted successfully."}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found in storage.")
            
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="File deletion service is not yet fully implemented."
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete file: {str(e)}")