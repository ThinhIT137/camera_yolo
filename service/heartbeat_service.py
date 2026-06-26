# File: services/heartbeat_service.py (hoặc để cùng thư mục đổi tên thành heartbeat_service.py)
import torch
import psutil
import requests
import time
import threading
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

config = {
    "MASTER_URL": os.getenv("MASTER_URL", "http://127.0.0.1:5000"),
    "SERVER_ID": os.getenv("SERVER_ID", "SV_01")
}

def _heartbeat_loop():
    """Vòng lặp chạy ngầm đo nhịp tim"""

    while True:
        current_master_url = config["MASTER_URL"]
        current_server_id = config["SERVER_ID"]
        try:
            payload = {
                "server_id": current_server_id,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "has_gpu": False,
                "gpu_usage": 0,
                "vram_free_gb": 0.0 
            }
            
            # Nếu có GPU Nvidia
            if torch.cuda.is_available():
                payload["has_gpu"] = True
                t_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                a_vram = torch.cuda.memory_allocated(0) / (1024**3)
                payload["vram_free_gb"] = round(t_vram - a_vram, 2)
            print(f"{payload}")
            requests.post(f"{current_master_url}/api/heartbeat", json=payload, timeout=2)
        except Exception:
            pass # Lỗi mạng thì im lặng chờ vòng lặp sau
        
        time.sleep(5) # 5 giây báo cáo 1 lần

def start_heartbeat():
    """Hàm khởi tạo luồng chạy ngầm để gọi từ bên ngoài"""
    thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    thread.start()
    print("💓 [Heartbeat Service] Đã kích hoạt máy đo nhịp tim chạy ngầm...")

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
        print(f"⚠️ Lỗi check camera {cam.id}: {e}")
        return cam, False
