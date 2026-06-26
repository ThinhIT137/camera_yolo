import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)

import asyncio
import psutil
import GPUtil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from service.websocket_service import manager

router = APIRouter()

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