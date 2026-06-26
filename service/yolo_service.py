import os
import time
import zmq
import threading
import torch
import math
from service.rtsp_service import check_rtsp_alive, ReconnectWatcher
from service.tracker import PersonTracker
import multiprocessing
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
    print(f"\n⚙️ [YOLO SỐ {chunk_id}] TIẾP NHẬN LÔ: {list(cameras_chunk.keys())} - SỬ DỤNG GPU ĐỂ TRACKING")
    
    tracker_app = PersonTracker(model_path="yolo11s.pt", reid_weights="osnet_x1_0_msmt17.pth")
    threading.Thread(target=zmq_listener, args=(tracker_app,), daemon=True).start()
    
    while True:
        alive_chunk = {}
        dead_chunk = {}
        
        for cam_id, url in cameras_chunk.items():
            if check_rtsp_alive(url): alive_chunk[cam_id] = url
            else: 
                dead_chunk[cam_id] = url
                print(f"⚠️ [YOLO SỐ {chunk_id}] {cam_id} MẤT KẾT NỐI. Đang chờ phục hồi...")
                
        if not alive_chunk:
            print(f"💀 [YOLO SỐ {chunk_id}] Toàn bộ Lô mất tín hiệu. Đợi 2s check lại...")
            time.sleep(2)
            continue 
            
        stream_file = f"streams_chunk_{chunk_id}.streams"
        url_to_cam_id = {} 
        with open(stream_file, "w") as f:
            for cam_id, url in alive_chunk.items():
                f.write(url + "\n")
                url_to_cam_id[url] = cam_id
                normalized_url = url.replace(":", "_")
                url_to_cam_id[normalized_url] = cam_id

        frame_counters = {cam_id: 0 for cam_id in alive_chunk.keys()}
        watcher = ReconnectWatcher(dead_chunk)

        print(f"🚀 [YOLO SỐ {chunk_id}] Đang quét tọa độ cho: {list(alive_chunk.keys())}")
        try:
            for stream_path, frame, detections in tracker_app.track(stream_file):
                if watcher.found_alive:
                    print(f"🎉 [YOLO SỐ {chunk_id}] CAMERA SỐNG LẠI! Nạp nóng luồng mới...")
                    break 

                cam_id = url_to_cam_id.get(stream_path)
                if cam_id is None:
                    for raw_url, mapped_id in alive_chunk.items():
                        cam_suffix = raw_url.rsplit("/", 1)[-1]  
                        if stream_path.endswith(cam_suffix):
                            cam_id = mapped_id
                            break

                if cam_id is None: continue

                frame_counters[cam_id] += 1
                if frame_counters[cam_id] % 30 == 0:
                    print(f"👁️ [YOLO SỐ {chunk_id} -> {cam_id}] Bắt được {len(detections)} người")

                orig_h, orig_w = frame.shape[:2]
                tracking_data = []

                for p in detections:
                    x1, y1, x2, y2 = p["bbox"]
                    tracking_data.append({
                        "id": p.get("global_id"), 
                        "name": p["name"],
                        "x": int(x1), "y": int(y1), 
                        "w": int(x2 - x1), "h": int(y2 - y1),
                        "orig_w": int(orig_w), "orig_h": int(orig_h)
                    })
                    
                # Bắn ra Queue dùng chung
                out_queue.put({"cam_id": cam_id, "timestamp": time.time(), "boxes": tracking_data})
                
        except Exception as e:
            print(f"💥 [YOLO SỐ {chunk_id}] Lỗi luồng ({e}). Đang tự phục hồi...")
            time.sleep(2)
        finally:
            watcher.stop() 
            if os.path.exists(stream_file): os.remove(stream_file)

def start_yolo_background(cameras_dict, tracking_queue):
    """
    Hàm này được gọi từ Service để chạy ngầm khởi động YOLO
    """
    def _run():
        cameras_items = list(cameras_dict.items())
        total_cams = len(cameras_items)

        if total_cams > 0:
            CHUNK_SIZE = calculate_optimal_chunk_size(total_cams) 
            print(f"🚀 [SERVICE] BẮT ĐẦU CHIA LÔ KHỞI ĐỘNG {total_cams} CAMERA...")

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
                
                print(f"✅ Đã bật Lô {chunk_id}. Nghỉ 6s nhường tài nguyên GPU...")
                time.sleep(6) 
            print("🎉 [SERVICE] TOÀN BỘ YOLO ĐÃ LÊN HÌNH!")
    # Chạy trong một luồng riêng để không block API chính
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def calculate_optimal_chunk_size(total_cams):
    """Thuật toán tự động tính tỷ lệ vàng (Min YOLO - Max Camera) - HỆ ĐẠI GIA"""
    # Nếu chạy bằng CPU, ép chạy lô 2 cho an toàn
    if not torch.cuda.is_available():
        return 2 
    # Lấy tổng VRAM đang trống (GB)
    t_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    a_vram = torch.cuda.memory_allocated(0) / (1024**3)
    free_vram_gb = t_vram - a_vram
    YOLO_BASE_VRAM = 1.2   # 1.2GB khởi tạo Model
    CAM_VRAM_COST = 0.4    # 0.4GB cho mỗi luồng stream
    LATENCY_LIMIT = 6      # Không vượt quá 6 Cam/YOLO để chống lag
    # Chống móm
    if free_vram_gb < (YOLO_BASE_VRAM + CAM_VRAM_COST):
        return 1 
    # 1. TÍNH XEM SERVER NÀY MỞ ĐƯỢC TỐI ĐA BAO NHIÊU YOLO (Tỷ lệ 1:1)
    max_possible_yolos = math.floor(free_vram_gb / (YOLO_BASE_VRAM + CAM_VRAM_COST))
    # 2. HỆ ĐẠI GIA: Nếu sức chứa >= Số Camera -> Mở mỗi cam 1 YOLO
    if max_possible_yolos >= total_cams:
        print(f"💎 [HỆ ĐẠI GIA] VRAM dư sức! Mở {total_cams} YOLO cho {total_cams} Camera. Tỷ lệ 1:1")
        return 1
    # 3. HỆ TIẾT KIỆM: Nếu VRAM không đủ mở 1:1, bắt đầu gộp lô (Batching)
    optimal_chunk = math.ceil(total_cams / max_possible_yolos)
    final_chunk = min(optimal_chunk, LATENCY_LIMIT)
    print(f"🧠 [AUTO-SCALE] VRAM {free_vram_gb:.1f}GB | Phân bổ tối ưu: {final_chunk} Cam / 1 YOLO")
    return max(1, final_chunk)