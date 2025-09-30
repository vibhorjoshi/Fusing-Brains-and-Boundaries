"""
Logging utility for GeoAI Research Backend
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup application logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure log format
    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    general_log_file = os.path.join(log_dir, "app.log")
    file_handler = logging.handlers.RotatingFileHandler(
        general_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file handler
    error_log_file = os.path.join(log_dir, "errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Access log file handler
    access_log_file = os.path.join(log_dir, "access.log")
    access_handler = logging.handlers.RotatingFileHandler(
        access_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(formatter)
    
    # Create access logger
    access_logger = logging.getLogger("access")
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class RequestLogger:
    """Request logging utility"""
    
    def __init__(self):
        self.logger = get_logger("access")
    
    def log_request(self, method: str, path: str, ip: str, user_agent: Optional[str] = None):
        """Log HTTP request"""
        message = f"{method} {path} - {ip}"
        if user_agent:
            message += f" - {user_agent}"
        self.logger.info(message)
    
    def log_response(self, method: str, path: str, status_code: int, response_time: float):
        """Log HTTP response"""
        message = f"{method} {path} - {status_code} - {response_time:.3f}s"
        
        if status_code >= 500:
            self.logger.error(message)
        elif status_code >= 400:
            self.logger.warning(message)
        else:
            self.logger.info(message)


class SecurityLogger:
    """Security event logging"""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_failed_login(self, username: str, ip: str):
        """Log failed login attempt"""
        self.logger.warning(f"Failed login attempt - Username: {username} - IP: {ip}")
    
    def log_invalid_api_key(self, api_key_prefix: str, ip: str):
        """Log invalid API key usage"""
        self.logger.warning(f"Invalid API key attempt - Key: {api_key_prefix}... - IP: {ip}")
    
    def log_unauthorized_access(self, path: str, ip: str):
        """Log unauthorized access attempt"""
        self.logger.warning(f"Unauthorized access attempt - Path: {path} - IP: {ip}")
    
    def log_successful_login(self, username: str, ip: str):
        """Log successful login"""
        self.logger.info(f"Successful login - Username: {username} - IP: {ip}")


class ProcessingLogger:
    """Processing operation logging"""
    
    def __init__(self):
        self.logger = get_logger("processing")
    
    def log_job_started(self, job_id: str, job_type: str, user_id: int):
        """Log processing job started"""
        self.logger.info(f"Job started - ID: {job_id} - Type: {job_type} - User: {user_id}")
    
    def log_job_completed(self, job_id: str, processing_time: float):
        """Log processing job completed"""
        self.logger.info(f"Job completed - ID: {job_id} - Time: {processing_time:.2f}s")
    
    def log_job_failed(self, job_id: str, error: str):
        """Log processing job failed"""
        self.logger.error(f"Job failed - ID: {job_id} - Error: {error}")
    
    def log_training_epoch(self, session_id: str, epoch: int, loss: float, accuracy: float):
        """Log training epoch completion"""
        self.logger.info(f"Training epoch - Session: {session_id} - Epoch: {epoch} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")


class PerformanceLogger:
    """Performance monitoring logging"""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_slow_request(self, method: str, path: str, response_time: float):
        """Log slow requests"""
        self.logger.warning(f"Slow request - {method} {path} - {response_time:.3f}s")
    
    def log_high_memory_usage(self, memory_percent: float):
        """Log high memory usage"""
        self.logger.warning(f"High memory usage - {memory_percent:.1f}%")
    
    def log_high_cpu_usage(self, cpu_percent: float):
        """Log high CPU usage"""
        self.logger.warning(f"High CPU usage - {cpu_percent:.1f}%")
    
    def log_system_stats(self, stats: dict):
        """Log system statistics"""
        self.logger.info(f"System stats - {stats}")