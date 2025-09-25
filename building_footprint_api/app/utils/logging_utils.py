"""
Logging utilities for the building footprint API
"""

import os
import logging
import logging.config
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.config import get_settings

settings = get_settings()

def setup_logging(log_level: str = None, log_to_file: bool = True) -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Optional log level override
        log_to_file: Whether to log to file
    """
    # Get log level from settings or parameter
    level = log_level or settings.LOG_LEVEL
    
    # Create log directory if it doesn't exist
    if log_to_file:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"api_{timestamp}.log"
    
    # Configure logging
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "json": {
                "()": "app.utils.logging_utils.JsonFormatter",
                "fmt_keys": {
                    "timestamp": "%(asctime)s",
                    "level": "%(levelname)s",
                    "name": "%(name)s",
                    "message": "%(message)s"
                }
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": level,
                "propagate": True
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": level,
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": level,
                "propagate": False
            }
        }
    }
    
    # Add file handler if logging to file
    if log_to_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "json",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10
        }
        
        # Add file handler to loggers
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["uvicorn"]["handlers"].append("file")
        config["loggers"]["uvicorn.access"]["handlers"].append("file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {level}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")

class JsonFormatter(logging.Formatter):
    """JSON formatter for logging"""
    
    def __init__(self, fmt_keys: Optional[Dict[str, str]] = None):
        """
        Initialize JSON formatter
        
        Args:
            fmt_keys: Format keys mapping
        """
        super().__init__()
        self.fmt_keys = fmt_keys or {
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "name": "%(name)s",
            "message": "%(message)s"
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log entry
        """
        log_data = {}
        
        # Apply format keys
        for key, fmt in self.fmt_keys.items():
            log_data[key] = self.formatMessage(logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=fmt % record.__dict__,
                args=(),
                exc_info=None
            ))
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", 
                          "filename", "funcName", "id", "levelname", "levelno", 
                          "lineno", "module", "msecs", "message", "msg", "name", 
                          "pathname", "process", "processName", "relativeCreated", 
                          "stack_info", "thread", "threadName"]:
                log_data[key] = value
        
        return json.dumps(log_data)

class RequestIdFilter(logging.Filter):
    """Filter to add request ID to log records"""
    
    def __init__(self, request_id: str):
        """
        Initialize request ID filter
        
        Args:
            request_id: Request ID
        """
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record and add request ID
        
        Args:
            record: Log record
            
        Returns:
            Whether to include the record in the log
        """
        record.request_id = self.request_id
        return True

def get_request_logger(request_id: str) -> logging.Logger:
    """
    Get logger with request ID
    
    Args:
        request_id: Request ID
        
    Returns:
        Logger with request ID filter
    """
    logger = logging.getLogger(f"request.{request_id}")
    logger.addFilter(RequestIdFilter(request_id))
    return logger