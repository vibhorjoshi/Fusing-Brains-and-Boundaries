"""
Health Controller for system monitoring
"""

from ..utils.monitoring import SystemMonitor
from ..models.response_models import HealthCheckResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HealthController:
    """Health check controller"""
    
    def __init__(self, system_monitor: SystemMonitor):
        self.system_monitor = system_monitor
    
    async def get_health_status(self) -> HealthCheckResponse:
        """Get system health status"""
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime=self.system_monitor.get_uptime(),
            system_info=await self.system_monitor.get_system_stats()
        )