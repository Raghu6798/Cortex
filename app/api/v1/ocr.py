# app/api/v1/ocr.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.auth.clerk_auth import get_current_user
from app.services.multimodal_parser import multimodal_parser_service

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR & Parsing"])

class ParseRequest(BaseModel):
    object_name: str

class ParseResponse(BaseModel):
    content: str

@router.post("/parse", response_model=ParseResponse)
async def parse_document(
    request: ParseRequest,
    token_payload: dict = Depends(get_current_user)
):
    """
    Triggers parsing of an already uploaded file.
    Expects the 'object_name' returned by the upload endpoint.
    """
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not request.object_name.startswith(f"{user_id}/"):
         raise HTTPException(status_code=403, detail="Access denied to this object")

    content = await multimodal_parser_service.parse_object(user_id, request.object_name)
    
    return ParseResponse(content=content)