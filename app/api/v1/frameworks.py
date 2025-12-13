"""
API routes for framework management.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi_clerk_auth import HTTPAuthorizationCredentials

from app.models.framework import FrameworkRegistry, FrameworkConfig
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer

clerk_config = ClerkConfig(
    jwks_url="https://supreme-caribou-95.clerk.accounts.dev/.well-known/jwks.json",
    auto_error=True
)
clerk_auth_guard = ClerkHTTPBearer(config=clerk_config, add_state=True)

router = APIRouter(prefix="/frameworks", tags=["frameworks"])

@router.get("")
async def get_available_frameworks(
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
) -> Dict[str, Any]:
    """Get all available agent frameworks."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    frameworks = FrameworkRegistry.get_framework_list()
    
    return {
        "frameworks": frameworks,
        "total": len(frameworks)
    }

@router.get("/enabled")
async def get_enabled_frameworks(
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
) -> Dict[str, Any]:
    """Get only enabled frameworks."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    enabled_frameworks = FrameworkRegistry.get_enabled_frameworks()
    frameworks = [
        {
            "id": f_id,
            "name": config.name,
            "description": config.description,
            "logo_url": config.logo_url,
            "status": config.status.value,
            "supported_providers": config.supported_providers,
            "default_model": config.default_model,
            "features": config.features,
            "endpoint": {
                "path": config.endpoint.path,
                "method": config.endpoint.method,
                "description": config.endpoint.description,
                "required_fields": config.endpoint.required_fields,
                "optional_fields": config.endpoint.optional_fields
            }
        }
        for f_id, config in enabled_frameworks.items()
    ]
    
    return {
        "frameworks": frameworks,
        "total": len(frameworks)
    }

@router.get("/{framework_id}")
async def get_framework_details(
    framework_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(clerk_auth_guard)
) -> Dict[str, Any]:
    """Get detailed information about a specific framework."""
    user_id = credentials.decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    
    framework = FrameworkRegistry.get_framework(framework_id)
    if not framework:
        raise HTTPException(status_code=404, detail="Framework not found")
    
    return {
        "id": framework_id,
        "name": framework.name,
        "description": framework.description,
        "logo_url": framework.logo_url,
        "status": framework.status.value,
        "supported_providers": framework.supported_providers,
        "default_model": framework.default_model,
        "features": framework.features,
        "documentation_url": framework.documentation_url,
        "endpoint": {
            "path": framework.endpoint.path,
            "method": framework.endpoint.method,
            "description": framework.endpoint.description,
            "required_fields": framework.endpoint.required_fields,
            "optional_fields": framework.endpoint.optional_fields
        }
    }
