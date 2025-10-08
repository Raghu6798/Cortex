# REVISED and CORRECTED FILE: test_auth_dependency.py

import uvicorn
import os
from fastapi import FastAPI, Depends, HTTPException, Request # <-- Import Request
from clerk_backend_api import Clerk
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions
from dotenv import load_dotenv

# --- 1. SETUP: Load Environment Variables ---
load_dotenv()
print("Attempting to load environment variables from .env file...")

# --- 2. The CORRECT Authentication Logic ---

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
if not CLERK_SECRET_KEY:
    raise ValueError("CRITICAL: CLERK_SECRET_KEY not set.")

print("CLERK_SECRET_KEY loaded successfully.")

# Initialize the Clerk SDK
clerk_sdk = Clerk(bearer_auth=CLERK_SECRET_KEY)

# The Main Authentication Dependency that we are testing
async def get_current_user(request: Request) -> dict:
    """
    A FastAPI dependency that authenticates a user using the official
    `authenticate_request` method and returns the token payload.
    """
    try:
        # The official method to authenticate a request from a web framework.
        # It automatically finds the token in headers or cookies.
        request_state = clerk_sdk.authenticate_request(
            request=request,
            options=AuthenticateRequestOptions()
        )
        
        # Check if the user is signed in
        if not request_state.is_signed_in:
            raise HTTPException(status_code=401, detail=f"Not authenticated: {request_state.reason}")

        # The token data is stored in the .payload attribute. THIS IS THE KEY.
        if not request_state.payload:
            raise HTTPException(status_code=403, detail="Invalid token: no payload.")
        
        # Return the dictionary of claims
        return request_state.payload

    except ClerkAPIException as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication credentials: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

app = FastAPI()

@app.get("/test-auth")
async def protected_route_test(session_claims: dict = Depends(get_current_user)):
    """
    A test endpoint protected by our dependency.
    """
    print("--- Inside protected_route_test ---")
    user_id = session_claims.get("sub")
    session_id = session_claims.get("sid")
    print(f"Successfully retrieved user ID from token claims: {user_id}")
    
    return {
        "message": "Authentication successful!",
        "authenticated_user_id": user_id,
        "session_id": session_id
    }

# --- 4. A Main Block to Run This Test Server ---

if __name__ == "__main__":
    print("\n--- Starting Standalone Authentication Test Server ---")
    print("Use Postman or a script to send a GET request to http://127.0.0.1:8001/test-auth")
    
    uvicorn.run(app, host="127.0.0.1", port=8001)