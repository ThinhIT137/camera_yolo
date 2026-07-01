import logging
# Tạo logger cục bộ cho file này (nó sẽ tự thừa kế cấu hình Root ở app.py / main.py)
logger = logging.getLogger(__name__)
import psutil
import time
import zmq
import threading
import torch
import subprocess
import numpy as np
import json
import os
# from service.rtsp_service import check_rtsp_alive, ReconnectWatcher
import queue
import multiprocessing
from urllib.parse import urlparse
from dotenv import load_dotenv
from multiprocessing import Process, Queue, Manager
from ultralytics import YOLO
from service.gallery import ReIDGallery
from service.tracker import CameraTracker

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "Thong_So_Yolo.json")
with open(STATE_PATH, encoding="utf-8") as f:
    data = json.load(f)

load_dotenv()
MASTER_URL = os.environ.get("MASTER_URL", "http://127.0.0.1:5000")
SERVER_ID = os.environ.get("SERVER_ID", "UNKNOWN_SERVER")

parsed_url = urlparse(MASTER_URL)
MASTER_IP = parsed_url.hostname or "127.0.0.1"
MASTER_ZMQ_PORT = 5558

_global_manager = None
shared_stats = {}

if multiprocessing.current_process().name == 'MainProcess':
    _global_manager = Manager()
    shared_stats = _global_manager.dict()

from service.state import (
    active_processes, 
    active_cameras_data, 
    yolo_vram,
    camera_vram,
    sai_so_yolo,
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

def ai_worker_process(chunk_id, cameras_chunk, out_queue, cmd_queue, shared_stats):
    logger.info(f"⚙️ [YOLO SỐ {chunk_id}] TIẾP NHẬN LÔ...")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gallery = ReIDGallery(reid_weights="osnet_x1_0_msmt17.pth")

        if hasattr(gallery, 'reid_model') and "cuda" in device:
            gallery.reid_model = gallery.reid_model.cuda()
            
        active_trackers = {} # Từ điển chứa các Object Camera đang chạy
        active_threads = {}

        # Viết hàm bật 1 con camera gọn lại để xài lại nhiều lần
        # Viết hàm bật 1 con camera gọn lại để xài lại nhiều lần
        def start_single_camera(cam_id, stream_url):
            logger.info(f"🚀 [YOLO {chunk_id}] Đang mở luồng {cam_id}...")
            
            # 🔥 TẠO MỘT HÀM WRAPPER ĐỂ ÉP VÒNG LẶP YIELD CHẠY
            def run_tracking_loop():
                context = zmq.Context()
                zmq_socket = context.socket(zmq.PUSH)
                zmq_address = f"tcp://{MASTER_IP}:{MASTER_ZMQ_PORT}"
                zmq_socket.connect(zmq_address)
                logger.info(f"🚀 [ZMQ] Camera {cam_id} đã cầm súng, nhắm thẳng Master tại {zmq_address}")

                try:
                    roi = (0, 0, 1920, 1080)
                    door_poly = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.int32)
                    
                    # Chuyển khởi tạo YOLO vào trong luồng để GPU không bị tranh chấp (CUDA Context)
                    yolo_model = YOLO("yolo11n.pt", task="detect")
                    time.sleep(1) 
                    yolo_model.to(device)
                    
                    tracker_instance = CameraTracker(
                        cam_id=cam_id, source_url=stream_url, gallery=gallery,
                        yolo_model=yolo_model, roi=roi, door_poly=door_poly,
                        tracker_config="custom_tracker.yaml"
                    )
                    tracker_instance.out_queue = out_queue
                    tracker_instance.shared_stats = shared_stats
                    tracker_instance.chunk_id = chunk_id
                    
                    active_trackers[cam_id] = tracker_instance
                    shared_stats[cam_id] = {"yolo_id": chunk_id, "latency": 15, "url": stream_url}

                    logger.info(f"🎥 [START] YOLO bắt đầu ăn VRAM và quét frame cho {cam_id}...")
                    
                    cnt = 0

                    # ✅ CÚ CHỐT: Bắt buộc phải dùng vòng lặp FOR để hàm có yield thực sự chạy
                    for frame, detections in tracker_instance.track_loop():
                        # Lặp qua đây thì code trong track_loop mới được thực thi!
                        orig_h, orig_w = frame.shape[:2]
                        cnt +=1
                        boxes_to_send = []
                        for det in detections:
                            x1, y1, x2, y2 = det["bbox"]
                            boxes_to_send.append({
                                "id": det["global_id"], 
                                "name": det["name"],
                                "x": int(x1),
                                "y": int(y1),
                                "w": int(x2 - x1),
                                "h": int(y2 - y1),
                                "orig_w": orig_w,
                                "orig_h": orig_h
                            })
                        
                        try:
                            data_payload = {
                                "server_id": SERVER_ID,
                                "cam_id": cam_id,
                                "boxes": boxes_to_send
                            }
                            # print(data_payload) # Mở comment cái này ra là thấy nó in data chạy nhòe màn hình luôn
                            
                            zmq_socket.send_json(data_payload, zmq.NOBLOCK)
                        except zmq.error.Again:
                            pass # Chống lag mạng

                        # # Bắn data vào Queue để trạm trung chuyển bơm ra WebSocket
                        # if tracker_instance.out_queue is not None:
                        #     try:
                        #         # Dùng put_nowait để nếu hàng đợi đầy thì bỏ qua frame này, tránh kẹt AI
                        #         tracker_instance.out_queue.put_nowait({
                        #             "cam_id": cam_id,
                        #             "boxes": boxes_to_send
                        #         })
                        #     except queue.Full:
                        #         pass
                        
                except Exception as e:
                    logger.error(f"❌ Lỗi sập luồng camera {cam_id}: {e}", exc_info=True)
                finally:
                    zmq_socket.close()

            # Gắn luồng vào cái hàm wrapper vừa tạo
            t = threading.Thread(target=run_tracking_loop, daemon=True)
            t.start()
            active_threads[cam_id] = t

        # 1. Bật toàn bộ camera lô ban đầu
        for cam_id, stream_url in cameras_chunk.items():
            start_single_camera(cam_id, stream_url)

        # 2. Vòng lặp trực tổng đài nghe lệnh từ Master
        while True:
            try:
                cmd = cmd_queue.get(timeout=1)
                
                if cmd['action'] == 'start':
                    # Lệnh bế cam từ nơi khác về
                    start_single_camera(cmd['cam_id'], cmd['url'])
                    
                elif cmd['action'] == 'stop':
                    # Lệnh quăng cam đi chỗ khác
                    cam_id = cmd['cam_id']
                    if cam_id in active_trackers:
                        logger.info(f"🛑 [YOLO {chunk_id}] Đóng băng {cam_id} để giảm tải...")
                        # Phải có hàm stop bên trong CameraTracker của sếp nhé
                        if hasattr(active_trackers[cam_id], 'stop'):
                            active_trackers[cam_id].stop() 
                        
                        # Cực kỳ quan trọng: Dọn dẹp để không tràn RAM
                        del active_trackers[cam_id]
                        del active_threads[cam_id]
                        if "cuda" in device:
                            torch.cuda.empty_cache() # Nhả VRAM trả lại cho GPU ngay lập tức

            except queue.Empty:
                pass # Không có tin nhắn thì lặp tiếp

    except Exception as e:
        logger.error(f"🚨 Lỗi nghiêm trọng tại YOLO Process lô số {chunk_id}: {e}", exc_info=True)
    
def auto_rebalancer(process_queues, shared_stats):
    """Giám sát độ trễ (ms) và đảo camera từ YOLO nghẽn sang YOLO rảnh"""
    logger.info("⚖️ [BALANCER] Bộ phận cân bằng tải động đã khởi động!")
    while True:
        time.sleep(10) # 10s kiểm tra 1 lần
        if not shared_stats: continue

        # Tính tổng độ trễ của từng YOLO
        load_per_yolo = {}
        cam_details = {}

        # shared_stats có dạng: {"cam_3": {"yolo_id": 1, "latency": 33, "url": "rtsp..."}}
        for cam_id, data in shared_stats.items():
            y_id = data['yolo_id']
            lat = data['latency']
            load_per_yolo[y_id] = load_per_yolo.get(y_id, 0) + lat
            cam_details[cam_id] = data

        if len(load_per_yolo) < 2:
            continue # Chỉ có 1 YOLO thì không có chỗ để chuyển

        # Tìm thằng khổ sai nhất và thằng rảnh nhất
        heaviest_yolo = max(load_per_yolo, key=load_per_yolo.get)
        lightest_yolo = min(load_per_yolo, key=load_per_yolo.get)

        max_load = load_per_yolo[heaviest_yolo]
        min_load = load_per_yolo[lightest_yolo]

        # NẾU LỆCH NHAU QUÁ 20ms VÀ THẰNG NẶNG ĐANG QUÁ 40ms -> CÓ DẤU HIỆU NGHẼN
        if (max_load - min_load) > 20 and max_load > 40:
            # Tìm camera nặng nhất của thằng khổ sai
            cams_in_heavy = {k: v['latency'] for k, v in cam_details.items() if v['yolo_id'] == heaviest_yolo}
            if not cams_in_heavy: continue
            
            # Chọn thằng cam đang delay nặng nhất để bốc đi
            heaviest_cam_id = max(cams_in_heavy, key=cams_in_heavy.get)
            heaviest_cam_url = cam_details[heaviest_cam_id]['url']

            logger.warning(f"🔄 [CỨU TRỢ] YOLO {heaviest_yolo} đang quá tải ({max_load}ms). Thuyên chuyển {heaviest_cam_id} sang YOLO {lightest_yolo} ({min_load}ms)")

            # 1. Bấm bộ đàm lệnh cho YOLO cũ TẮT cam này
            process_queues[heaviest_yolo].put({"action": "stop", "cam_id": heaviest_cam_id})

            # 2. Bấm bộ đàm lệnh cho YOLO mới BẬT cam này
            process_queues[lightest_yolo].put({"action": "start", "cam_id": heaviest_cam_id, "url": heaviest_cam_url})

            # Cập nhật tạm thời để vòng lặp sau không bắt nhầm
            shared_stats[heaviest_cam_id] = {
                "yolo_id": lightest_yolo,
                "latency": cams_in_heavy[heaviest_cam_id],
                "url": heaviest_cam_url
            }

def start_yolo_background(cameras_dict, tracking_queue):
    global shared_stats # Lôi biến shared_stats xịn ở trên xuống xài
    
    process_queues = {}
    
    def _run():
        cameras_items = list(cameras_dict.items())
        total_cams = len(cameras_items)

        if total_cams > 0:
            optimal_params = calculate_optimal_chunk_size() 
            max_yolo_processes = optimal_params.get("yolo", 1)
            num_yolo_to_open = min(max_yolo_processes, total_cams)

            logger.debug(f"🚀 [SERVICE] MỞ {num_yolo_to_open} YOLO PROCESS CHO {total_cams} CAMERA...")
            
            yolo_chunks = [{} for _ in range(num_yolo_to_open)]

            # Lượt 1: Chia bài
            for idx, (cam_id, stream_url) in enumerate(cameras_items):
                yolo_idx = idx % num_yolo_to_open
                yolo_chunks[yolo_idx][cam_id] = stream_url

            # Kích hoạt các YOLO
            for i, chunk in enumerate(yolo_chunks):
                if not chunk: continue
                chunk_id = i + 1
                cmd_queue = Queue()
                process_queues[chunk_id] = cmd_queue

                # Truyền cái shared_stats (đã bọc thép) vào Process con
                p = Process(
                    target=ai_worker_process, 
                    args=(chunk_id, chunk, tracking_queue, cmd_queue, shared_stats)
                )
                active_processes.append(p)
                p.start()
                time.sleep(6) 
            
            logger.debug("🎉 [SERVICE] TOÀN BỘ YOLO ĐÃ LÊN HÌNH VÀ VÀO VỊ TRÍ CHIẾN ĐẤU!")
            
            try:
                auto_rebalancer(process_queues, shared_stats)
            except Exception as e:
                logger.error(f"⚠️ [CẢNH BÁO] Lỗi trong auto_rebalancer: {e}")
            
            # Khiên đỡ luồng không cho sập
            while True:
                time.sleep(10)
                
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def calculate_optimal_chunk_size():
    """
    Thuật toán tự động tính tỷ lệ vàng (Dùng subprocess để không lỗi CUDA)
    Tính ra được camera và yolo tối đa của máy
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        free_vram_gb = float(out.stdout.strip().split('\n')[0]) / 1024.0 - 0.4
        YOLO_BASE_VRAM = yolo_vram 
        CAM_VRAM_COST = camera_vram
        TARGET_LOAD = sai_so_yolo
        BASE = {
            "yolo" : 1,
            "cam" : 0,
            "avg" : 0,
            "score": float("inf"),
        }
        max_yolo = int(free_vram_gb / YOLO_BASE_VRAM)
        for i in range(2, max_yolo + 1):
            yolo = i
            remaining_vram = free_vram_gb - YOLO_BASE_VRAM * yolo
            if remaining_vram <= 0:
                continue
            cam = int(remaining_vram / CAM_VRAM_COST)
            if cam < yolo or cam <= 0: 
                continue
            avg = cam / yolo
            score = abs(TARGET_LOAD-avg)
            if score < BASE["score"]:
                BASE = {
                    "yolo": yolo,
                    "cam": cam,
                    "avg": avg,
                    "score": score
                }
        data["BASE"] = BASE
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return BASE
    except Exception as e:
        logger.error(f"⚠️ Không dùng được GPU (Hoặc lỗi nvidia-smi): {e}. Tính toán số cam cho 1 yolo")
        try :
            cpu_usage = psutil.cpu_percent(interval=0.5)
            if cpu_usage < 40:
                cam = 2
            else:
                cam = 1  
            BASE =  {
                "yolo" : 1,
                "cam" : cam,
                "avg" : 2,
                "score": 0,
            }
            data["BASE"] = BASE
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return BASE
        except Exception as ex:
            logger.error(f"⚠️ Máy có vấn đề nặng không chạy được cả cpu lẫn gpu =))")