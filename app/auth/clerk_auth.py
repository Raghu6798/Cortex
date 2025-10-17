# REVISED and CORRECTED FILE: app/auth/clerk_auth.py

import os
from fastapi import Request, Depends, HTTPException
from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

from app.config.settings import settings
from app.utils.logger import logger

if not settings.CLERK_SECRET_KEY:
    raise ValueError("CLERK_SECRET_KEY environment variable not set.")

# Initialize the main Clerk SDK instance
clerk_sdk = Clerk(bearer_auth=settings.CLERK_SECRET_KEY)

async def get_current_user(request: Request) -> dict:
    """
    A FastAPI dependency that authenticates a user using the official
    `authenticate_request` method and returns the token payload.
    """
    try:
        auth_header = request.headers.get('Authorization')
        logger.info(f"BACKEND DEBUG: Authorization header received: {auth_header}")
        request_state = clerk_sdk.authenticate_request(
            request=request,
            options=AuthenticateRequestOptions()
        )
        
        if not request_state.is_signed_in or not request_state.payload:
            raise HTTPException(status_code=401, detail=f"Not authenticated: {request_state.reason or 'No payload'}")

        return request_state.payload
    except Exception as e:
        print("--- CRITICAL ERROR IN CLERK AUTH ---")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


