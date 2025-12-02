# app/integrations/mini_client.py
from minio import Minio
from app.config.settings import settings
from app.utils.logger import logger

class MinIOClient:
    _instance: Minio | None = None
    
    @classmethod
    def get_client(cls) -> Minio:
        """
        Initializes the client object only. 
        This creates the Python object but DOES NOT make a network connection yet.
        This is safe to run even if the server is down.
        """
        if cls._instance is None:
            cls._instance = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
        return cls._instance

    @classmethod
    def verify_connection(cls):
        """
        Attempts to check if the bucket exists.
        If it fails, it LOGS the error but DOES NOT CRASH the app.
        """
        client = cls.get_client()
        try:
            if not client.bucket_exists(settings.MINIO_BUCKET):
                logger.warning(f"Bucket '{settings.MINIO_BUCKET}' not found. Attempting to create...")
                client.make_bucket(settings.MINIO_BUCKET)
                logger.info(f"Created bucket '{settings.MINIO_BUCKET}'")
            else:
                logger.info(f"Connected to Storage. Bucket '{settings.MINIO_BUCKET}' exists.")
                
        except Exception as e:
            logger.error(f"⚠️ STORAGE CONNECTION FAILED: {e}")
            logger.warning("The app has started, but file upload/download features may not work.")

minio_client = MinIOClient.get_client()