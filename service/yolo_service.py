import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)

import os
import time
import zmq
import threading
import torch
import math
import subprocess
import numpy as np
from service.rtsp_service import check_rtsp_alive, ReconnectWatcher
import multiprocessing
from ultralytics import YOLO
from service.gallery import ReIDGallery
from service.tracker import CameraTracker

from service.state import (
    active_processes, 
    active_cameras_data, 
    ai_worker_process, 
    calculate_optimal_chunk_size
)
current_config_hash = ""

def zmq_listener(tracker_app):
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.connect("tcp://10.40.90.214:5557") 
    while True:
        try:
            msg = pull.recv_json()
            if msg["action"] == "track" and msg["name"] not in tracker_app.pending_targets:
                tracker_app.pending_targets.append(msg["name"])
        except Exception: pass

def ai_worker_process(chunk_id, cameras_chunk, out_queue):
    """
    Tiến trình công nhân AI: Gánh một lô camera.
    """
    logger.info(f"⚙️ [YOLO SỐ {chunk_id}] TIẾP NHẬN LÔ: {list(cameras_chunk.keys())} - ÉP SỬ DỤNG GPU TRƠN TRU")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.debug(f"⚙️ [CPU hay GPU] Server đang chạy yolo bằng: {device}")
        # 🌟 1. KHỞI TẠO DUY NHẤT 1 BỘ REID GALLERY DÙNG CHUNG
        # Để các camera chia sẻ trí nhớ khuôn mặt cho nhau
        gallery = ReIDGallery(reid_weights="osnet_x1_0_msmt17.pth")
        if hasattr(gallery, 'reid_model') and "cuda" in device:
            gallery.reid_model = gallery.reid_model.cuda()
        logger.info(f"✅ [YOLO SỐ {chunk_id}] Đã nạp thành công ReIDGallery lên GPU.")
        threads = []
        for cam_id, stream_url in cameras_chunk.items():
            roi = (0, 0, 1920, 1080)
            door_poly = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.int32)
            # 🔥 2. MỖI CAMERA TỰ KHỞI TẠO 1 BẢN SAO YOLO RIÊNG (Nhưng nằm chung Process nên xài chung CUDA Context, siêu tiết kiệm VRAM)
            yolo_model = YOLO("yolo11s.pt", task="detect")
            yolo_model.to(device)
            logger.info(f"🤖 [CHECK GPU] YOLO của {cam_id} đang chạy trên: {yolo_model.device}")
            
            # 3. Gắn YOLO riêng và Gallery chung vào Tracker
            tracker_instance = CameraTracker(
                cam_id=cam_id,
                source_url=stream_url,
                gallery=gallery,
                yolo_model=yolo_model, # <--- Giờ mỗi cam ôm 1 con YOLO độc lập, không sợ đá ID của nhau
                roi=roi,
                door_poly=door_poly,
                tracker_config="custom_tracker.yaml"
            )
            
            tracker_instance.out_queue = out_queue
            
            target_method = None
            for method_name in ['start', 'run', 'start_tracking', 'track_loop']:
                if hasattr(tracker_instance, method_name):
                    target_method = getattr(tracker_instance, method_name)
                    break
            
            if target_method:
                t = threading.Thread(target=target_method, daemon=True)
                t.start()
                threads.append(t)
                logger.info(f"🚀 [SUCCESS] Đã kích hoạt luồng AI chạy ngầm cho {cam_id}")

        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"🚨 Lỗi nghiêm trọng tại YOLO Process lô số {chunk_id}: {e}", exc_info=True)
    
def start_yolo_background(cameras_dict, tracking_queue):
    """
    Hàm này được gọi từ Service để chạy ngầm khởi động YOLO
    """
    def _run():
        cameras_items = list(cameras_dict.items())
        total_cams = len(cameras_items)

        if total_cams > 0:
            CHUNK_SIZE = calculate_optimal_chunk_size(total_cams) 
            logger.debug(f"🚀 [SERVICE] BẮT ĐẦU CHIA LÔ KHỞI ĐỘNG {total_cams} CAMERA...")

            for i in range(0, total_cams, CHUNK_SIZE):
                chunk = dict(cameras_items[i : i + CHUNK_SIZE])
                chunk_id = (i // CHUNK_SIZE) + 1
                
                # Tạo process mới
                p = multiprocessing.Process(
                    target=ai_worker_process, 
                    args=(chunk_id, chunk, tracking_queue)
                )
                active_processes.append(p)
                p.start()
                
                logger.debug(f"✅ Đã bật Lô {chunk_id}. Nghỉ 6s nhường tài nguyên GPU...")
                time.sleep(6) 
            logger.debug("🎉 [SERVICE] TOÀN BỘ YOLO ĐÃ LÊN HÌNH!")
    # Chạy trong một luồng riêng để không block API chính
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def calculate_optimal_chunk_size(total_cams):
    """Thuật toán tự động tính tỷ lệ vàng (Dùng subprocess để không lỗi CUDA)"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        free_vram_gb = float(out.stdout.strip().split('\n')[0]) / 1024.0
        
        YOLO_BASE_VRAM = 1.2   
        CAM_VRAM_COST = 0.4    
        LATENCY_LIMIT = 6      
        
        if free_vram_gb < (YOLO_BASE_VRAM + CAM_VRAM_COST):
            return 1 
        max_possible_yolos = math.floor(free_vram_gb / (YOLO_BASE_VRAM + CAM_VRAM_COST))
        if max_possible_yolos >= total_cams:
            logger.debug(f"💎 [HỆ ĐẠI GIA] VRAM dư sức! Mở {total_cams} YOLO cho {total_cams} Camera. Tỷ lệ 1:1")
            return 1
        optimal_chunk = math.ceil(total_cams / max_possible_yolos)
        final_chunk = min(optimal_chunk, LATENCY_LIMIT)
        logger.debug(f"🧠 [AUTO-SCALE] VRAM {free_vram_gb:.1f}GB | Phân bổ tối ưu: {final_chunk} Cam / 1 YOLO")
        return max(1, final_chunk)
        
    except Exception as e:
        logger.error(f"⚠️ Không dùng được GPU (Hoặc lỗi nvidia-smi): {e}. Ép chạy lô 2 cho an toàn!")
        return 2
    
def test_hardcore_gpu_vram():
    print(f"--- BẮT ĐẦU TEST GPU ---")
    # 1. Kiểm tra driver
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    # 2. Setup thiết bị
    device = "cuda:0"
    # 3. Load model với log VRAM
    print(f"VRAM trước khi load: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    model = YOLO("yolo11s.pt").to(device)
    print(f"VRAM sau khi load model: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    # 4. QUAN TRỌNG: Phải "ép" nó chạy 1 frame thì VRAM mới nhả ra đúng
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    print("Đang chạy 1 frame giả lập...")
    _ = model.predict(dummy_input, device=0)
    print(f"VRAM sau khi chạy 1 frame: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print("--- TEST XONG ---")