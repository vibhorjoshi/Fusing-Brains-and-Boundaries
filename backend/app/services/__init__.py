"""
Services package for GeoAI Research Backend
Handles data access and external service integrations
"""

from .auth_service import AuthService
from .processing_service import ProcessingService
from .training_service import TrainingService
from .storage_service import StorageService

__all__ = [
    'AuthService',
    'ProcessingService',
    'TrainingService', 
    'StorageService'
]