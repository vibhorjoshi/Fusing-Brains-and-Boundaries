"""
Database configuration and connection management
Production-ready PostgreSQL with SQLAlchemy
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging
from typing import Generator

from .config import settings

logger = logging.getLogger(__name__)

# Database engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_POOL_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

# Metadata for database operations
metadata = MetaData()

async def init_db():
    """Initialize database - create tables if they don't exist"""
    try:
        # Import all models to ensure they're registered
        from app.models import user, building_footprint, processing_job, file_storage
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    Automatically handles connection lifecycle
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

class DatabaseManager:
    """Database manager for production operations"""
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.session_factory()
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        return {
            "url": settings.DATABASE_URL,
            "pool_size": settings.DATABASE_POOL_SIZE,
            "pool_overflow": settings.DATABASE_POOL_OVERFLOW,
            "active_connections": self.engine.pool.checked_in(),
            "total_connections": self.engine.pool.checked_out()
        }

# Global database manager instance
db_manager = DatabaseManager()