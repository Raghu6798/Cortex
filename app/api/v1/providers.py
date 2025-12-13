# app/api/v1/providers.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.database import get_db
from app.db.models import LLMProviderDB, LLMModelDB
from app.integrations.llm_router import llm_router
from app.auth.clerk_auth import get_current_user
from app.schemas.provider_schemas import ProviderResponse, ModelResponse
from app.utils.logger import logger

router = APIRouter(prefix="/api/v1/providers", tags=["providers"])

@router.get("", response_model=List[ProviderResponse])
async def get_all_providers(db: Session = Depends(get_db)):
    """Get all available LLM providers with their models"""
    logger.info("Accessing get_all_providers endpoint")
    try:
        providers = db.query(LLMProviderDB).all()
        result = []
        for provider in providers:
            models = db.query(LLMModelDB).filter(
                LLMModelDB.provider_id == provider.id
            ).all()
            
            provider_data = {
                "id": provider.id,
                "name": provider.name,
                "display_name": provider.display_name,
                "base_url": provider.base_url,
                "logo_url": provider.logo_url,
                "description": provider.description,
                "requires_api_key": provider.requires_api_key,
                "supports_streaming": provider.supports_streaming,
                "supports_tools": provider.supports_tools,
                "supports_embeddings": provider.supports_embeddings,
                "max_tokens": provider.max_tokens if provider.max_tokens else 4096,
                "models": [
                    {
                        "id": model.id,
                        "model_id": model.model_id,
                        "display_name": model.display_name,
                        "description": model.description,
                        "context_length": model.context_length
                    }
                    for model in models
                ]
            }
            result.append(provider_data)
        return result
    except Exception as e:
        print(f"DEBUG: Error in get_all_providers: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")   

@router.get("/{provider_id}/models", response_model=List[ModelResponse])
async def get_provider_models(provider_id: str,db: Session = Depends(get_db)):
    """Get models for a specific provider"""
    provider = db.query(LLMProviderDB).filter(LLMProviderDB.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    models = db.query(LLMModelDB).filter(
        LLMModelDB.provider_id == provider_id
    ).all()
    
    return [
        ModelResponse(
            id=model.id,
            model_id=model.model_id,
            display_name=model.display_name,
            description=model.description,
            context_length=model.context_length
        )
        for model in models
    ]

@router.post("/sync")
async def sync_providers(db: Session = Depends(get_db)):
    """Sync providers and models from external APIs to database"""
    try:
        await llm_router.sync_providers_to_db(db)
        return {"message": "Providers and models synced successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing providers: {str(e)}")

@router.post("/{provider_id}/chat")
async def chat_with_provider(
provider_id: str,request: Dict[str, Any],current_user: dict = Depends(get_current_user)):
    """Test chat completion with a specific provider"""
    try:
        provider = await llm_router.get_provider(provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        
        messages = request.get("messages", [])
        model = request.get("model")
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1000)
        
        if not messages or not model:
            raise HTTPException(status_code=400, detail="Messages and model are required")
        
        response = await provider.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with chat completion: {str(e)}")
