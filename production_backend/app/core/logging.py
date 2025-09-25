"""
Production logging configuration
CloudWatch integration for AWS deployment
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from .config import settings

def setup_logging():
    """Setup production-ready logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JsonFormatter,
            }
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": "logs/error.log",
                "maxBytes": 10485760,
                "backupCount": 5
            }
        },
        "loggers": {
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file", "error_file"]
        }
    }
    
    # Add CloudWatch handler for production
    if settings.ENVIRONMENT == "production" and settings.CLOUDWATCH_LOG_GROUP:
        try:
            logging_config["handlers"]["cloudwatch"] = {
                "level": "INFO",
                "class": "watchtower.CloudWatchLogsHandler",
                "formatter": "json",
                "log_group": settings.CLOUDWATCH_LOG_GROUP,
                "stream_name": settings.CLOUDWATCH_LOG_STREAM,
                "use_queues": True,
                "send_interval": 10,
                "max_batch_size": 100,
                "max_batch_count": 10000
            }
            
            # Add CloudWatch to loggers
            for logger_name in ["app", "root"]:
                logging_config["loggers"][logger_name]["handlers"].append("cloudwatch")
                
        except ImportError:
            # watchtower not installed, skip CloudWatch logging
            pass
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Set up application logger
    logger = logging.getLogger("app")
    logger.info(f"🔧 Logging configured for environment: {settings.ENVIRONMENT}")
    logger.info(f"📊 Log level: {settings.LOG_LEVEL}")
    
    return logger

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        # Basic log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)

class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests"""
    
    def __init__(self, app):
        self.app = app
        self.logger = logging.getLogger("app.requests")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = datetime.utcnow()
            
            # Log request
            self.logger.info(
                "HTTP Request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "query_string": scope["query_string"].decode(),
                    "client": scope.get("client"),
                    "user_agent": dict(scope.get("headers", {})).get(b"user-agent", b"").decode()
                }
            )
            
            # Process request
            await self.app(scope, receive, send)
            
            # Log response time
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "HTTP Response",
                extra={
                    "method": scope["method"], 
                    "path": scope["path"],
                    "duration_seconds": duration
                }
            )
        else:
            await self.app(scope, receive, send)

# Performance monitoring logger
perf_logger = logging.getLogger("app.performance")

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    perf_logger.info(
        f"Performance: {operation}",
        extra={
            "operation": operation,
            "duration_seconds": duration,
            **kwargs
        }
    )

# Error tracking logger  
error_logger = logging.getLogger("app.errors")

def log_error(error: Exception, context: str = None, **kwargs):
    """Log errors with context"""
    error_logger.error(
        f"Error in {context or 'application'}",
        extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            **kwargs
        },
        exc_info=True
    )