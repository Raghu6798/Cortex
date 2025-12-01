# app/services/file_service.py

import uuid
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
from fastapi import UploadFile, HTTPException, status
from typing import Dict, Any, List

# Import logger and settings for configuration and logging
from app.config.settings import settings
from app.integrations.mini_client import minio_client
from app.utils.logger import logger


class FileService:
    """
    Service class for handling all file storage operations via MinIO.
    """

    def __init__(self, minio_client: Minio):
        """
        Initializes the FileService with a MinIO client instance.
        """
        self.minio_client = minio_client
        self.bucket_name = settings.MINIO_BUCKET

    async def upload_file(self, user_id: str, file: UploadFile) -> Dict[str, Any]:
        """
        Uploads a file stream to MinIO, securing it under a unique path.
        Path format: {user_id}/files/{file_id}/{original_filename}
        """
        if file.size is None or file.size <= 0:
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST, 
                 detail="File is empty or size could not be determined."
             )
             
        file_id = str(uuid.uuid4())
        object_name = f"{user_id}/files/{file_id}/{file.filename}"
        
        try:
            logger.info(f"Attempting to upload file '{file.filename}' to MinIO path: {object_name}")
            
            # Use put_object to stream the file content directly
            result = self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file.file,
                length=file.size,
                content_type=file.content_type if file.content_type else 'application/octet-stream',
            )
            
            # The internal reference URL
            file_url = f"/storage/{result.bucket_name}/{result.object_name}"
            
            logger.success(f"File uploaded successfully. ETag: {result.etag}")

            return {
                "message": "File uploaded successfully",
                "file_id": file_id,
                "original_filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": file.size,
                "object_name": result.object_name,
                "file_url": file_url 
            }
        except S3Error as e:
            logger.error(f"MinIO S3 Error during upload: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"S3 Storage Error: {e.code}. Details: {e.message}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"File upload failed: {str(e)}"
            )

    async def list_user_files(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Lists all files belonging to a specific user.
        It filters objects starting with the prefix "{user_id}/files/".
        """
        prefix = f"{user_id}/files/"
        files_list = []

        try:
            # list_objects returns an iterator
            objects = self.minio_client.list_objects(
                self.bucket_name, 
                prefix=prefix, 
                recursive=True
            )

            for obj in objects:
                # Extract the original filename from the path: user_id/files/uuid/filename.ext
                # Splitting by '/' and taking the last part
                try:
                    display_name = obj.object_name.split('/')[-1]
                except Exception:
                    display_name = obj.object_name

                files_list.append({
                    "object_name": obj.object_name,
                    "display_name": display_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                })
            
            return files_list

        except S3Error as e:
            logger.error(f"MinIO Error listing files: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Storage Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

    async def delete_file(self, user_id: str, object_name: str) -> bool:
        """
        Deletes a specific file from MinIO.
        """
        # Double check ownership (though router usually handles this too)
        if not object_name.startswith(f"{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied to this object.")

        try:
            self.minio_client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted object: {object_name}")
            return True
        except S3Error as e:
            logger.error(f"MinIO Error deleting file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Storage Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error deleting file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    async def create_presigned_download_url(self, user_id: str, object_name: str) -> str:
        """
        Generates a temporary (1 hour) pre-signed URL for downloading/viewing the file.
        """
        # Double check ownership
        if not object_name.startswith(f"{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied to this object.")

        try:
            url = self.minio_client.presigned_get_object(
                self.bucket_name, 
                object_name, 
                expires=timedelta(hours=1)
            )
            return url
        except S3Error as e:
            logger.error(f"MinIO Error generating URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Storage Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error generating URL: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to generate URL: {str(e)}")


# Initialize singleton
file_service = FileService(minio_client=minio_client)