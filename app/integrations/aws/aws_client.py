# app/integrations/aws_client.py
import boto3
from botocore.exceptions import ClientError
from app.config.settings import settings
from app.utils.logger import logger

class AWSS3Client:
    _instance = None

    @classmethod
    def get_client(cls):
        if cls._instance is None:
            try:
                cls._instance = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY,
                    aws_secret_access_key=settings.AWS_SECRET_KEY,
                    region_name=settings.AWS_REGION
                )
                logger.info("AWS S3 Client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize AWS S3 Client: {e}")
                raise e
        return cls._instance

    @classmethod
    def ensure_bucket_exists(cls):
        """Checks if the S3 bucket exists."""
        s3 = cls.get_client()
        try:
            s3.head_bucket(Bucket="cortex-production-storage")
            logger.info(f"AWS S3: Bucket 'cortex-production-storage' validated.")
        except ClientError as e:
            logger.error(f"AWS S3: Bucket 'cortex-production-storage' access failed or does not exist: {e}")
          
aws_s3_client = AWSS3Client.get_client()

if __name__ == "__main__":
    print(AWSS3Client.ensure_bucket_exists())