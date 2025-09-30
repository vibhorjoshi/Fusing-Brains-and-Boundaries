"""
Training Controller for GeoAI Research Backend
"""

from typing import Dict, Any
from ..services.training_service import TrainingService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrainingController:
    """Training controller for managing training sessions"""
    
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return await self.training_service.get_training_status()
    
    async def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start new training session"""
        return await self.training_service.start_training_session(config)