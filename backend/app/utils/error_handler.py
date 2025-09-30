"""
Error handling utilities for GeoAI Research Backend
"""

import traceback
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..models.response_models import ErrorResponse
from .logger import get_logger

logger = get_logger(__name__)


class ApplicationError(Exception):
    """Base application error"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(ApplicationError):
    """Authentication related errors"""
    pass


class AuthorizationError(ApplicationError):
    """Authorization related errors"""
    pass


class ProcessingError(ApplicationError):
    """Processing related errors"""
    pass


class ValidationError(ApplicationError):
    """Validation related errors"""
    pass


class ExternalServiceError(ApplicationError):
    """External service related errors"""
    pass


def setup_error_handlers(app: FastAPI) -> None:
    """Setup global error handlers for the application"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}"
            ).dict()
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions"""
        logger.error(f"Starlette HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=str(exc.detail),
                error_code=f"HTTP_{exc.status_code}"
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.error(f"Validation Error: {exc.errors()} - Path: {request.url.path}")
        
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation failed",
                error_code="VALIDATION_ERROR",
                details={"validation_errors": error_details}
            ).dict()
        )
    
    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError):
        """Handle authentication errors"""
        logger.error(f"Authentication Error: {exc.message} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code or "AUTHENTICATION_ERROR",
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(request: Request, exc: AuthorizationError):
        """Handle authorization errors"""
        logger.error(f"Authorization Error: {exc.message} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code or "AUTHORIZATION_ERROR",
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request: Request, exc: ProcessingError):
        """Handle processing errors"""
        logger.error(f"Processing Error: {exc.message} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code or "PROCESSING_ERROR",
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle validation errors"""
        logger.error(f"Validation Error: {exc.message} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code or "VALIDATION_ERROR",
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(ExternalServiceError)
    async def external_service_error_handler(request: Request, exc: ExternalServiceError):
        """Handle external service errors"""
        logger.error(f"External Service Error: {exc.message} - Path: {request.url.path}")
        
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code or "EXTERNAL_SERVICE_ERROR",
                details=exc.details
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        logger.error(f"Unhandled Exception: {str(exc)} - Path: {request.url.path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                error_code="INTERNAL_ERROR",
                details={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)
                }
            ).dict()
        )


class ErrorReporter:
    """Error reporting utility"""
    
    def __init__(self):
        self.logger = get_logger("error_reporter")
    
    def report_error(self, error: Exception, context: Dict[str, Any] = None):
        """Report error with context"""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(f"Error reported: {error_info}")
    
    def report_validation_error(self, field: str, value: Any, expected: str):
        """Report validation error"""
        self.logger.error(f"Validation error - Field: {field}, Value: {value}, Expected: {expected}")
    
    def report_processing_error(self, job_id: str, error: Exception, step: str = None):
        """Report processing error"""
        context = {"job_id": job_id}
        if step:
            context["processing_step"] = step
        
        self.report_error(error, context)
    
    def report_authentication_error(self, username: str, reason: str, ip: str = None):
        """Report authentication error"""
        context = {
            "username": username,
            "reason": reason,
            "ip_address": ip
        }
        
        self.logger.error(f"Authentication error reported: {context}")


def handle_async_errors(func):
    """Decorator to handle async function errors"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ApplicationError:
            raise  # Re-raise application errors as they have specific handlers
        except Exception as e:
            logger.error(f"Async function error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ProcessingError(f"Error in {func.__name__}: {str(e)}")
    
    return wrapper


def handle_sync_errors(func):
    """Decorator to handle sync function errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApplicationError:
            raise  # Re-raise application errors as they have specific handlers
        except Exception as e:
            logger.error(f"Sync function error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ProcessingError(f"Error in {func.__name__}: {str(e)}")
    
    return wrapper