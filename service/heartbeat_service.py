import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)

# File: services/heartbeat_service.py (hoặc để cùng thư mục đổi tên thành heartbeat_service.py)
import psutil
import requests
import time
import threading
import os
import cv2
import subprocess
from dotenv import load_dotenv

load_dotenv()

config = {
    "MASTER_URL": os.getenv("MASTER_URL", "http://127.0.0.1:5000"),
    "SERVER_ID": os.getenv("SERVER_ID", "SV_01")
}

def _get_gpu_vram():
    """Hàm lấy VRAM bằng nvidia-smi để tránh lỗi CUDA Context của PyTorch"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        # Sẽ trả về string kiểu "3254\n" (đơn vị MiB)
        free_mib = float(out.stdout.strip().split('\n')[0])
        return round(free_mib / 1024.0, 2)
    except Exception:
        return 0.0

def _heartbeat_loop():
    """Vòng lặp chạy ngầm đo nhịp tim"""
    while True:
        current_master_url = config["MASTER_URL"]
        current_server_id = config["SERVER_ID"]
        try:
            free_vram = _get_gpu_vram() # Gọi hàm mới
            
            payload = {
                "server_id": current_server_id,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "has_gpu": free_vram > 0, # Có VRAM > 0 nghĩa là có GPU Nvidia
                "gpu_usage": 0, 
                "vram_free_gb": free_vram 
            }
            
            logger.debug(f"💓 Nhịp tim gửi Master: {payload}")
            requests.post(f"{current_master_url}/api/heartbeat", json=payload, timeout=2)
        except Exception:
            pass 
        time.sleep(5)

def start_heartbeat():
    """Hàm khởi tạo luồng chạy ngầm để gọi từ bên ngoài"""
    thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    thread.start()
    logger.info("💓 [Heartbeat Service] Đã kích hoạt máy đo nhịp tim chạy ngầm...")

def check_rtsp_stream(cam):
    """Hàm test nhanh: Trả về True nếu đọc ổn định được 5 khung hình liên tiếp"""
    try:
        # Ép dùng backend FFMPEG cho chuẩn với RTSP
        cap = cv2.VideoCapture(cam.url, cv2.CAP_FFMPEG) 
        if not cap.isOpened():
            return cam, False
        success_count = 0
        # Bắt nó nhai thử 5 khung hình xem có bị nghẹn không =))
        for _ in range(5):
            ret, _ = cap.read()
            if ret:
                success_count += 1
            else:
                # Đang đọc mà rớt mạng giữa chừng -> Nghỉ khỏe!
                break      
        cap.release()
        # Sống sót qua 5 khung hình mới cấp visa "Accepted"
        is_alive = (success_count == 5)
        return cam, is_alive
    except Exception as e:
        logger.error(f"⚠️ Lỗi check camera {cam.id}: {e}")
        return cam, False
