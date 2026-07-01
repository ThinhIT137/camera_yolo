import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)

import asyncio
import psutil
import GPUtil
import os
import zmq
import queue
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from service.websocket_service import manager
from dotenv import load_dotenv
from urllib.parse import urlparse

router = APIRouter()
load_dotenv()

MASTER_URL = os.environ.get("MASTER_URL", "http://127.0.0.1:5000")
SERVER_ID = os.environ.get("SERVER_ID", "UNKNOWN_SERVER")

# Bóc tách IP từ MASTER_URL (VD: "http://10.40.91.11:5000" -> "10.40.91.11")
parsed_url = urlparse(MASTER_URL)
MASTER_IP = parsed_url.hostname or "127.0.0.1"
MASTER_ZMQ_PORT = 5558

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
                # Ném phi tiêu qua Master! 
                socket.send_json(data, zmq.NOBLOCK)
        except zmq.error.Again:
            pass # Bộ đệm mạng đầy / Master sập -> vứt frame này đi để bảo vệ RAM Worker
        except Exception as e: 
            logger.error(f"⚠️ Lỗi gửi ZMQ tới Master: {e}")

@router.websocket("/ws/tracking/{cam_id}")
async def websocket_endpoint(websocket: WebSocket, cam_id: str):
    await manager.connect(websocket, cam_id)
    try:
        while True: 
            await websocket.receive_text()
    except WebSocketDisconnect: 
        manager.disconnect(websocket, cam_id)

@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            gpus = GPUtil.getGPUs()
            gpu_load = gpus[0].load * 100 if gpus else 0.0
            stats_data = {
                "cpu": round(psutil.cpu_percent(), 1),
                "gpu": round(gpu_load, 1),
                "ram": round(psutil.virtual_memory().percent, 1)
            }
            logger.debug(f"{stats_data}")
            await websocket.send_json(stats_data)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass

