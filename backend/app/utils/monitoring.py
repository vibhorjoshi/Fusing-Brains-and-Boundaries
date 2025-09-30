"""
System monitoring utilities for GeoAI Research Backend
"""

import psutil
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.monitoring = False
        self.stats_history = []
        
    async def start(self):
        """Start system monitoring"""
        self.monitoring = True
        logger.info("System monitoring started")
        
        # Start background monitoring task
        asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                stats = await self.get_system_stats()
                self.stats_history.append(stats)
                
                # Keep only last 100 entries
                if len(self.stats_history) > 100:
                    self.stats_history.pop(0)
                
                # Log warnings for high resource usage
                if stats["cpu_percent"] > 80:
                    logger.warning(f"High CPU usage: {stats['cpu_percent']:.1f}%")
                
                if stats["memory_percent"] > 80:
                    logger.warning(f"High memory usage: {stats['memory_percent']:.1f}%")
                
                if stats["disk_percent"] > 90:
                    logger.warning(f"High disk usage: {stats['disk_percent']:.1f}%")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory stats
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk stats
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network stats (if available)
            try:
                network = psutil.net_io_counters()
                network_sent = network.bytes_sent
                network_recv = network.bytes_recv
            except:
                network_sent = 0
                network_recv = 0
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_1min = load_avg[0]
                load_5min = load_avg[1]
                load_15min = load_avg[2]
            except (AttributeError, OSError):
                load_1min = load_5min = load_15min = 0
            
            # GPU stats (if available)
            gpu_stats = await self._get_gpu_stats()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime": self.get_uptime(),
                "cpu_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
                "memory_percent": round(memory_percent, 2),
                "memory_available_gb": round(memory_available / (1024**3), 2),
                "memory_total_gb": round(memory_total / (1024**3), 2),
                "disk_percent": round(disk_percent, 2),
                "disk_free_gb": round(disk_free / (1024**3), 2),
                "disk_total_gb": round(disk_total / (1024**3), 2),
                "network_sent_gb": round(network_sent / (1024**3), 2),
                "network_recv_gb": round(network_recv / (1024**3), 2),
                "process_count": process_count,
                "load_1min": round(load_1min, 2),
                "load_5min": round(load_5min, 2),
                "load_15min": round(load_15min, 2),
                **gpu_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics (if available)"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    "gpu_available": True,
                    "gpu_name": gpu.name,
                    "gpu_memory_used": round(gpu.memoryUsed, 2),
                    "gpu_memory_total": round(gpu.memoryTotal, 2),
                    "gpu_memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                    "gpu_temperature": gpu.temperature,
                    "gpu_load": round(gpu.load * 100, 2)
                }
            else:
                return {"gpu_available": False}
                
        except ImportError:
            return {"gpu_available": False, "note": "GPUtil not installed"}
        except Exception as e:
            return {"gpu_available": False, "error": str(e)}
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.start_time
    
    def get_uptime_formatted(self) -> str:
        """Get formatted uptime string"""
        uptime_seconds = int(self.get_uptime())
        
        days = uptime_seconds // 86400
        hours = (uptime_seconds % 86400) // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts)
    
    def get_recent_stats(self, count: int = 10) -> list:
        """Get recent system statistics"""
        return self.stats_history[-count:] if self.stats_history else []
    
    def get_average_stats(self, minutes: int = 5) -> Optional[Dict[str, Any]]:
        """Get average statistics for the last N minutes"""
        if not self.stats_history:
            return None
        
        recent_stats = [
            stat for stat in self.stats_history
            if (datetime.now() - datetime.fromisoformat(stat["timestamp"])).total_seconds() <= minutes * 60
        ]
        
        if not recent_stats:
            return None
        
        # Calculate averages
        avg_stats = {
            "period_minutes": minutes,
            "sample_count": len(recent_stats),
            "avg_cpu_percent": sum(s.get("cpu_percent", 0) for s in recent_stats) / len(recent_stats),
            "avg_memory_percent": sum(s.get("memory_percent", 0) for s in recent_stats) / len(recent_stats),
            "avg_disk_percent": sum(s.get("disk_percent", 0) for s in recent_stats) / len(recent_stats),
        }
        
        return avg_stats
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform system health check"""
        stats = await self.get_system_stats()
        
        health_status = "healthy"
        issues = []
        
        # Check CPU usage
        if stats.get("cpu_percent", 0) > 90:
            health_status = "warning"
            issues.append("High CPU usage")
        
        # Check memory usage
        if stats.get("memory_percent", 0) > 90:
            health_status = "critical"
            issues.append("Critical memory usage")
        elif stats.get("memory_percent", 0) > 80:
            health_status = "warning"
            issues.append("High memory usage")
        
        # Check disk usage
        if stats.get("disk_percent", 0) > 95:
            health_status = "critical"
            issues.append("Critical disk usage")
        elif stats.get("disk_percent", 0) > 85:
            health_status = "warning"
            issues.append("High disk usage")
        
        return {
            "status": health_status,
            "issues": issues,
            "stats": stats,
            "uptime": self.get_uptime_formatted()
        }