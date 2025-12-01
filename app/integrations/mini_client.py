# app/integrations/minio_client.py (NEW FILE)
from minio import Minio
from app.config.settings import settings
from app.utils.logger import logger

class MinIOClient:
    _instance: Minio | None = None
    
    @classmethod
    def get_client(cls) -> Minio:
        if cls._instance is None:
            # Note: For Docker networking, use the service name 'minio'
            endpoint = settings.MINIO_ENDPOINT
            
            try:
                cls._instance = Minio(
                    endpoint,
                    access_key=settings.MINIO_ROOT_USER,
                    secret_key=settings.MINIO_ROOT_PASSWORD,
                    secure=settings.MINIO_SECURE
                )
                
                # Check/Create the required bucket on startup
                if not cls._instance.bucket_exists(settings.MINIO_BUCKET):
                    cls._instance.make_bucket(settings.MINIO_BUCKET)
                    logger.info(f"MinIO: Created bucket '{settings.MINIO_BUCKET}'")
                else:
                    logger.info(f"MinIO: Bucket '{settings.MINIO_BUCKET}' already exists")
                    
            except Exception as e:
                logger.error(f"Failed to initialize MinIO Client: {e}")
                # You might want to raise here to prevent startup if storage is essential
                raise

        return cls._instance


minio_client = MinIOClient.get_client()
if __name__ == "__main__":
    minio_client = MinIOClient.get_client()
    print(minio_client)