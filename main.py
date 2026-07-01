from until.logger_utils import setup_global_system_logger # Sửa lại chữ until/config cho đúng folder của bro nha
logger = setup_global_system_logger(log_file_prefix="worker-node")
logger.info("🎬 [WORKER] Hệ thống Logger toàn cục theo phiên đã kích hoạt!")

import os
import threading
import asyncio
import uvicorn
import multiprocessing
import requests

# 🌟 BÙA CHÚ CHỐNG LỖI WINDOWS VÀ TỐI ƯU MẠNG
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout;3000000|rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import các Module tự viết
from service.state import tracking_queue
from service.websocket_service import queue_to_master
from service.heartbeat_service import start_heartbeat
from service.heartbeat_service import config
from service.yolo_service import calculate_optimal_chunk_size
from api.router import cameras_router, websocket_router, system_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # 🟢 LÚC SERVER KHỞI ĐỘNG (Trần gian)
    # ==========================================
    # loop = asyncio.get_running_loop()
    # threading.Thread(target=queue_to_master, args=(tracking_queue), daemon=True).start()
    start_heartbeat() 
    yield 

    # ==========================================
    # 🔴 LÚC SERVER BỊ TẮT (Xuống suối vàng)
    # ==========================================
    logger.info("\n🛑 Server YOLO đang tắt máy! Đang báo cáo về Master để xả Camera...")
    try:
        master_url = config.get("MASTER_URL")
        server_id = config.get("SERVER_ID")
        requests.post(f"{master_url}/api/servers/offline", json={"server_id": server_id}, timeout=3)
        logger.debug("✅ Đã bàn giao lại toàn bộ Camera cho Master thành công!\n")
    except Exception as e:
        logger.error(f"❌ Không thể báo cáo cho Master. Lỗi: {e}")   

# Khởi tạo App
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Gắn (Include) các API Router vào App chính
app.include_router(cameras_router.router)
app.include_router(websocket_router.router)
app.include_router(system_router.router)

if __name__ == "__main__":
    # Ép dùng 'spawn' cho đa tiến trình (Bắt buộc với CUDA/GPU)
    multiprocessing.set_start_method('spawn', force=True)
    BASE = calculate_optimal_chunk_size()
    logger.info("\n=======================================================")
    logger.info("🤖 WORKER YOLO AI ĐÃ SẴN SÀNG - ĐANG CHỜ MASTER GIAO VIỆC")
    logger.info("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)