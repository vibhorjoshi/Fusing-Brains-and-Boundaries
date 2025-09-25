"""
Error handling utilities for the building footprint API
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    status_code: int
    errors: Optional[List[Dict[str, Any]]] = None
    request_id: Optional[str] = None

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions
    
    Args:
        request: Request object
        exc: Exception
        
    Returns:
        JSON response
    """
    # Get request ID from headers or generate new one
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    # Log exception
    logger.error(
        f"HTTP error: {exc.status_code} - {exc.detail}",
        extra={"request_id": request_id}
    )
    
    # Create response
    response = ErrorResponse(
        detail=str(exc.detail),
        status_code=exc.status_code,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump()
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle validation exceptions
    
    Args:
        request: Request object
        exc: Exception
        
    Returns:
        JSON response
    """
    # Get request ID from headers or generate new one
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    # Get errors
    errors = exc.errors()
    
    # Format error details
    error_details = []
    for error in errors:
        error_details.append({
            "location": error.get("loc", ["unknown"]),
            "message": error.get("msg", "Unknown error"),
            "type": error.get("type", "unknown_error")
        })
    
    # Log exception
    logger.error(
        f"Validation error: {error_details}",
        extra={"request_id": request_id}
    )
    
    # Create response
    response = ErrorResponse(
        detail="Validation Error",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        errors=error_details,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response.model_dump()
    )

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unhandled exceptions
    
    Args:
        request: Request object
        exc: Exception
        
    Returns:
        JSON response
    """
    # Get request ID from headers or generate new one
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    # Log exception
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={"request_id": request_id}
    )
    
    # Create response
    response = ErrorResponse(
        detail="Internal Server Error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump()
    )

def register_exception_handlers(app):
    """
    Register exception handlers
    
    Args:
        app: FastAPI app
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)