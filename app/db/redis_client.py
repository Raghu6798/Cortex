"""
Redis client configuration and utilities.

This module provides Redis connection management following the same pattern
as the PostgreSQL database connection in database.py.
"""

import redis
from typing import Optional
from app.config.settings import settings


class RedisClient:
    """Redis client singleton class."""
    
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """
        Get or create a Redis client instance.
        
        Returns:
            redis.Redis: Redis client instance
        """
        if cls._instance is None:
            cls._instance = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_DB', 0),
                password=getattr(settings, 'REDIS_PASSWORD', None),
                decode_responses=True,  # Automatically decode bytes to strings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
        return cls._instance
    
    @classmethod
    def close(cls):
        """Close the Redis connection."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_redis() -> redis.Redis:
    """
    FastAPI dependency to get a Redis client.
    
    Returns:
        redis.Redis: Redis client instance
    """
    return RedisClient.get_client()

redis_client = RedisClient.get_client()

