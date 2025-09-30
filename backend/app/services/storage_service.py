"""
Storage Service for file management
"""

import os
import shutil
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StorageService:
    """Service for managing file storage"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save_file(self, file_data: bytes, filename: str, subdir: str = "") -> str:
        """Save file to storage"""
        try:
            file_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
            os.makedirs(file_dir, exist_ok=True)
            
            file_path = os.path.join(file_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """Get file from storage"""
        try:
            full_path = os.path.join(self.base_dir, file_path)
            
            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting file: {str(e)}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from storage"""
        try:
            full_path = os.path.join(self.base_dir, file_path)
            
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"File deleted: {full_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False