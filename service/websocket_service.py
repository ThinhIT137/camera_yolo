import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)

import asyncio
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self): 
        self.active_connections: dict[str, list[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, cam_id: str):
        await websocket.accept()
        if cam_id not in self.active_connections: 
            self.active_connections[cam_id] = []
        self.active_connections[cam_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, cam_id: str):
        if cam_id in self.active_connections and websocket in self.active_connections[cam_id]:
            self.active_connections[cam_id].remove(websocket)
            
    async def broadcast(self, message: dict, cam_id: str):
        if cam_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[cam_id]:
                try: await connection.send_json(message)
                except Exception: disconnected.append(connection)
            for conn in disconnected: 
                self.disconnect(conn, cam_id)

# Tạo một instance duy nhất để các file khác gọi vào
manager = ConnectionManager()

def queue_to_websocket(loop, q):
    """Luồng ngầm đọc tọa độ từ Queue và bơm thẳng ra WebSocket"""
    logger.info("🚀 Đã khởi động trạm trung chuyển Queue -> WebSocket")
    while True:
        try:
            data = q.get() 
            if data:
                cam_id = data.get("cam_id")
                asyncio.run_coroutine_threadsafe(manager.broadcast(data, cam_id), loop)
        except Exception as e: 
            logger.error(f"⚠️ Lỗi trung chuyển WebSocket: {e}")