"""
WebSocket Connection Manager
"""

import logging
from typing import Dict, List, Set, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket connection manager
    
    Handles WebSocket connections, disconnections, and message broadcasting
    """
    
    def __init__(self):
        # Maps client ID to list of websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Set of all active connections
        self.all_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Connect a client
        
        Args:
            websocket: WebSocket connection
            client_id: Client ID
        """
        await websocket.accept()
        
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        
        self.active_connections[client_id].append(websocket)
        self.all_connections.add(websocket)
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.all_connections)}")
    
    def disconnect(self, client_id: str):
        """
        Disconnect a client
        
        Args:
            client_id: Client ID
        """
        if client_id in self.active_connections:
            # Remove all websockets for this client from all_connections
            for websocket in self.active_connections[client_id]:
                self.all_connections.discard(websocket)
            
            # Remove client from active_connections
            del self.active_connections[client_id]
            
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.all_connections)}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """
        Send a message to a specific client
        
        Args:
            message: Message to send
            client_id: Client ID
        """
        if client_id in self.active_connections:
            for websocket in self.active_connections[client_id]:
                await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """
        Broadcast a message to all clients
        
        Args:
            message: Message to broadcast
        """
        disconnected_websockets = []
        
        for websocket in self.all_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {str(e)}")
                disconnected_websockets.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected_websockets:
            self.all_connections.discard(websocket)
            
            # Also remove from client-specific connections
            for client_id in list(self.active_connections.keys()):
                if websocket in self.active_connections[client_id]:
                    self.active_connections[client_id].remove(websocket)
                
                # Clean up if client has no more connections
                if len(self.active_connections[client_id]) == 0:
                    del self.active_connections[client_id]