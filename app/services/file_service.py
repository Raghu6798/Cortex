# app/services/file_service.py
import uuid
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
from fastapi import UploadFile, HTTPException, status
from typing import Dict, Any, List

from app.config.settings import settings
from app.integrations.mini_client import minio_client
from app.integrations.aws.aws_client import aws_s3_client
from app.utils.logger import logger

class FileService:
    """
    Service class for handling file storage via MinIO (Self-Hosted) OR AWS S3.
    """

    def __init__(self):
        self.minio = minio_client
        self.s3 = aws_s3_client
        
        self.minio_bucket = settings.MINIO_BUCKET
        self.s3_bucket = settings.AWS_S3_BUCKET

    async def upload_file(self, user_id: str, file: UploadFile, use_s3: bool = False) -> Dict[str, Any]:
        """
        Uploads file to either MinIO or S3 based on use_s3 flag.
        """
        if file.size is None:
             
             pass 
             
        file_id = str(uuid.uuid4())

        provider_prefix = "s3" if use_s3 else "minio"
        object_name = f"{user_id}/{provider_prefix}/{file_id}/{file.filename}"
        
        try:
            logger.info(f"Uploading '{file.filename}' to {'AWS S3' if use_s3 else 'MinIO'} at {object_name}")
            
            if use_s3:
        
                self.s3.upload_fileobj(
                    file.file,
                    self.s3_bucket,
                    object_name,
                    ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
                )
                
              
                region = getattr(settings, "AWS_REGION", "us-east-1")
              
                file_url = f"https://{self.s3_bucket}.s3.{region}.amazonaws.com/{object_name}"
            else:
               
                result = self.minio.put_object(
                    bucket_name=self.minio_bucket,
                    object_name=object_name,
                    data=file.file,
                    length=file.size if file.size else -1, 
                    part_size=10*1024*1024,
                    content_type=file.content_type or 'application/octet-stream',
                )
             
                file_url = f"/storage/{result.bucket_name}/{result.object_name}"

            return {
                "message": "File uploaded successfully",
                "provider": "aws_s3" if use_s3 else "minio",
                "file_id": file_id,
                "original_filename": file.filename,
                "object_name": object_name,
                "file_url": file_url 
            }

        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    async def list_user_files(self, user_id: str, use_s3: bool = False) -> List[Dict[str, Any]]:
        """Lists files. User must specify which storage to look in."""
        provider_prefix = "s3" if use_s3 else "minio"
        prefix = f"{user_id}/{provider_prefix}/"
        files_list = []

        try:
            if use_s3:
                response = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)
                if 'Contents' in response:
                    for obj in response['Contents']:
                        files_list.append({
                            "object_name": obj['Key'],
                            "display_name": obj['Key'].split('/')[-1],
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'],
                            "provider": "aws_s3"
                        })
            else:
                objects = self.minio.list_objects(self.minio_bucket, prefix=prefix, recursive=True)
                for obj in objects:
                    files_list.append({
                        "object_name": obj.object_name,
                        "display_name": obj.object_name.split('/')[-1],
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "provider": "minio"
                    })
            return files_list

        except Exception as e:
            logger.error(f"List files failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_file(self, user_id: str, object_name: str) -> bool:
        """
        Deletes file. Auto-detects provider based on path prefix.
        """
        if not object_name.startswith(f"{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied")


        is_s3 = f"/{user_id}/s3/" in f"/{object_name}" 

        try:
            if is_s3:
                self.s3.delete_object(Bucket=self.s3_bucket, Key=object_name)
            else:
                self.minio.remove_object(self.minio_bucket, object_name)
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def create_presigned_download_url(self, user_id: str, object_name: str) -> str:
        """
        Generates a temporary (1 hour) pre-signed URL for downloading/viewing the file.
        Works for both MinIO and AWS S3.
        """

        if not object_name.startswith(f"{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied to this object.")

        is_s3 = f"/{user_id}/s3/" in f"/{object_name}"

        try:
            if is_s3:
                url = self.s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.s3_bucket, 'Key': object_name},
                    ExpiresIn=3600
                )
                return url
            else:
                url = self.minio.presigned_get_object(
                    self.minio_bucket, 
                    object_name, 
                    expires=timedelta(hours=1)
                )
                return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {object_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Storage Error: {str(e)}")

file_service = FileService()