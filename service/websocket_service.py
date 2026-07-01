import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)

logger = logging.getLogger(__name__)

import zmq
import queue
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import asyncio
from fastapi import WebSocket

load_dotenv()
MASTER_URL = os.environ.get("MASTER_URL", "http://127.0.0.1:5000")
SERVER_ID = os.environ.get("SERVER_ID", "UNKNOWN_SERVER")

# Bóc tách IP từ MASTER_URL (VD: "http://10.40.91.11:5000" -> "10.40.91.11")
parsed_url = urlparse(MASTER_URL)
MASTER_IP = parsed_url.hostname or "127.0.0.1"
MASTER_ZMQ_PORT = 5558

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

def queue_to_master(q):
    """Luồng ngầm đọc tọa độ từ Queue của các tiến trình YOLO cục bộ, 
       rồi ném (PUSH) thẳng qua Master Node qua mạng LAN/Internet"""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    # Kết nối TCP tới máy Master
    zmq_address = f"tcp://{MASTER_IP}:{MASTER_ZMQ_PORT}"
    socket.connect(zmq_address)
    logger.info(f"🚀 [WORKER {SERVER_ID}] Trạm ZMQ đã kết nối TỚI Master tại {zmq_address}")
    while True:
        try:
            # Lấy data từ giỏ (queue) do các luồng YOLO ném vào
            data = q.get() 
            if data:
                # 🔥 Đính kèm luôn Thẻ tên (SERVER_ID) để Master biết thằng nào đang cày
                data["server_id"] = SERVER_ID
                print(data)
                # Ném phi tiêu qua Master! 
                socket.send_json(data, zmq.NOBLOCK)
        except zmq.error.Again:
            pass # Bộ đệm mạng đầy / Master sập -> vứt frame này đi để bảo vệ RAM Worker
        except Exception as e: 
            logger.error(f"⚠️ Lỗi gửi ZMQ tới Master: {e}")