"""
Test imports to verify our setup is working
"""

try:
    from pydantic_settings import BaseSettings
    print("✅ pydantic_settings imported successfully")
except ImportError:
    print("❌ Error importing pydantic_settings")

try:
    from app.core.websocket_manager import WebSocketManager
    print("✅ WebSocketManager imported successfully")
except ImportError as e:
    print(f"❌ Error importing WebSocketManager: {e}")

try:
    from app.services.workflow import WorkflowManager
    print("✅ WorkflowManager imported successfully")
except ImportError as e:
    print(f"❌ Error importing WorkflowManager: {e}")

print("Import test complete.")