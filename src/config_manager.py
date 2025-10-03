import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default environment if not specified
DEFAULT_ENV = "development"

# Path to configuration files
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
ENV_CONFIG_DIR = os.path.join(CONFIG_DIR, "environments")

class ConfigManager:
    """
    Configuration manager for environment-specific settings
    """
    _instance = None
    _config_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._current_env = os.environ.get("ENVIRONMENT", DEFAULT_ENV)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load the configuration for the current environment"""
        env = self._current_env
        if env in self._config_cache:
            return
        
        config_path = os.path.join(ENV_CONFIG_DIR, f"{env}.json")
        
        try:
            with open(config_path, "r") as f:
                self._config_cache[env] = json.load(f)
            logger.info(f"Loaded configuration for environment: {env}")
        except FileNotFoundError:
            logger.warning(f"Config file not found for environment '{env}', falling back to development")
            # If the specified environment doesn't exist, fall back to development
            if env != DEFAULT_ENV:
                self._current_env = DEFAULT_ENV
                self._load_config()
            else:
                raise ValueError(f"Configuration file not found for default environment: {DEFAULT_ENV}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")
    
    def get_config(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get the configuration for the current environment
        
        Args:
            refresh: If True, reload the configuration from disk
            
        Returns:
            The configuration dictionary
        """
        if refresh or self._current_env not in self._config_cache:
            self._load_config()
        
        return self._config_cache[self._current_env]
    
    def set_environment(self, env: str):
        """
        Change the current environment
        
        Args:
            env: The environment to switch to
        """
        self._current_env = env
        self._load_config()
        
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its dot-notation path
        
        Args:
            key_path: Dot-notation path to the configuration value (e.g., "services.api.port")
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        config = self.get_config()
        keys = key_path.split(".")
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def overlay_config(self, override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new config by overlaying the provided config on top of the current environment config
        
        Args:
            override_config: Configuration values to override
            
        Returns:
            The merged configuration
        """
        import copy
        config = copy.deepcopy(self.get_config())
        
        def _recursive_update(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    _recursive_update(target[key], value)
                else:
                    target[key] = value
        
        _recursive_update(config, override_config)
        return config


# Singleton instance
config_manager = ConfigManager()

# Convenience functions
def get_config(refresh: bool = False) -> Dict[str, Any]:
    """Get the current environment configuration"""
    return config_manager.get_config(refresh)

def get_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by its dot-notation path"""
    return config_manager.get_value(key_path, default)

def set_environment(env: str):
    """Change the current environment"""
    config_manager.set_environment(env)

def get_environment() -> str:
    """Get the current environment name"""
    return config_manager._current_env

def is_development() -> bool:
    """Check if the current environment is development"""
    return get_environment() == "development"

def is_production() -> bool:
    """Check if the current environment is production"""
    return get_environment() == "production"

def is_staging() -> bool:
    """Check if the current environment is staging"""
    return get_environment() == "staging"