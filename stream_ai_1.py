import cv2
import subprocess
import time
import zmq
import json
import multiprocessing # Dùng Multiprocessing thay vì Threading để chạy 5 cam không bị nghẽn!
from kafka import KafkaProducer # Gọi Kafka về làm nhiệm vụ báo cáo
import asyncio
import socket

# Import hàm track "thần thánh"
from Backend.camera_jolo.service.tracker import PersonTracker

def zmq_listener(tracker_app):
    """Lắng nghe tên người từ hệ thống ZMQ (Giữ nguyên)"""
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.connect("tcp://10.40.90.214:5557") 

    while True:
        try:
            msg = pull.recv_json()
            if msg["action"] == "track":
                name = msg["name"]
                if name not in tracker_app.pending_targets:
                    tracker_app.pending_targets.append(name)
        except Exception as e:
            pass

def stream_ai_to_mediamtx(cam_id, rtsp_in, rtsp_out, kafka_broker_ip):

    print(f"🚀 [Khởi động] {cam_id} -> Đang nạp Model AI vào GPU...")
    
    # 1. Khởi tạo cục AI
    tracker_app = PersonTracker(model_path="yolo11s.pt", reid_weights="osnet_x1_0_msmt17.pth")

    # 2. Khởi tạo luồng ZMQ
    import threading
    threading.Thread(target=zmq_listener, args=(tracker_app,), daemon=True).start()

    wait_for_kafka()

    # 3. KHỞI TẠO KAFKA PRODUCER (Để bắn log điểm danh về Server Quản lý)
    producer = KafkaProducer(
        bootstrap_servers=[kafka_broker_ip],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        linger_ms=10, # Bắn data cực nhanh
        api_version=(2, 5, 0)
    )

    width, height = 640, 360
    fps = 15 

    # 4. BÍ THUẬT FFMPEG (NVENC)
    # 4. BÍ THUẬT FFMPEG (NVENC) - ĐÃ ĐƯỢC NÂNG CẤP CHO WEBRTC
    # 4. BÍ THUẬT FFMPEG (NVENC) - ĐÃ ĐƯỢC NÂNG CẤP ÉP XUNG MAX PING
    ffmpeg_cmd = [
        'ffmpeg', 
        '-y', 
        
        # --- ÉP ĐẦU VÀO KHÔNG QUA BỘ ĐỆM ---
        '-fflags', 'nobuffer',      
        '-analyzeduration', '0',    
        '-probesize', '32',         
        
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', 
        '-s', f'{width}x{height}', 
        '-r', str(fps), 
        '-i', '-',                      
        
        # --- ÉP ĐẦU RA (CARD NVIDIA) XỬ LÝ TỨC THÌ ---
        '-c:v', 'h264_nvenc', 
        '-preset', 'p1',         
        '-tune', 'ull',
        '-profile:v', 'baseline',       
        '-delay', '0',           # Ép NVENC không được chờ frame tiếp theo
        '-rc', 'cbr', 
        '-b:v', '1M', 
        '-bf', '0',              
        '-g', '5',          
        '-pix_fmt', 'yuv420p',   
        '-f', 'rtsp', 
        '-rtsp_transport', 'tcp', 
        rtsp_out                        
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 5. Vòng lặp xử lý chính
    for frame, detections in tracker_app.track(rtsp_in):
        # --- A. BƠM VIDEO LÊN MEDIAMTX ---
        frame_resized = cv2.resize(frame, (width, height))

        try:
            proc.stdin.write(frame_resized.tobytes())
            proc.stdin.flush()
        except Exception:
            break
            
        # --- B. BƠM DỮ LIỆU ĐIỂM DANH LÊN KAFKA ---
        # Lọc ra những người đã nhận diện được tên (không phải Unknown)
        known_persons = [p for p in detections if p.get("name") and p["name"] != "Unknown"]
        
        if known_persons:
            # Bắn 1 tin nhắn nhẹ hều về Kafka để Database lưu lại
            log_data = {
                "cam_id": cam_id,
                "timestamp": time.time(),
                "detected": [{"id": p.get("global_id"), "name": p["name"]} for p in known_persons]
            }
            producer.send('attendance_logs', value=log_data)

    proc.stdin.close()
    proc.wait()

def wait_for_kafka():
    """Hàm đợi Kafka sống dậy rồi mới cho chạy AI"""
    print("⏳ Bắt đầu kiểm tra kết nối Kafka...")
    while True:
        try:
            with socket.create_connection(("127.0.0.1", 9092), timeout=1):
                print("✅ Kafka đã sẵn sàng!")
                break
        except (socket.timeout, ConnectionRefusedError):
            print("⏳ Đang đợi Kafka...")
            asyncio.sleep(2)

if __name__ == "__main__":
    # KHAI BÁO IP KAFKA & MEDIAMTX (Chỉnh lại nếu chạy khác máy)
    KAFKA_IP = "127.0.0.1:9092" 
    MEDIAMTX_HOST = "127.0.0.1:8554"

    # # =========================================================
    # # 🌟 FIX LỖI XUNG ĐỘT FILE: Tải trước model ở luồng chính
    # # =========================================================
    # print("📥 Đang kiểm tra và tải model YOLO (Chỉ tải 1 lần)...")
    # try:
    #     from ultralytics import YOLO
    #     _ = YOLO("yolo11s.pt") # Gọi nháp 1 lần để nó tải file an toàn
    #     print("✅ Đã chuẩn bị xong file YOLO!")
    # except Exception as e:
    #     print(f"⚠️ Lỗi khởi tạo YOLO: {e}. Vui lòng xóa file yolo11s.pt bị hỏng và chạy lại.")
    #     exit()
    # # =========================================================

    # 1. ĐỌC FILE JSON LẤY DANH SÁCH CAMERA
    import os
    import json
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    json_path = "cameras.json"
    if not os.path.exists(json_path):
        print(f"❌ Lỗi: Không tìm thấy file {json_path}")
        exit()

    with open(json_path, "r", encoding="utf-8") as f:
        cameras_dict = json.load(f)

    print(f"🚀 TÌM THẤY {len(cameras_dict)} CAMERA TRONG CẤU HÌNH. BẮT ĐẦU KHỞI ĐỘNG MULTIPROCESSING 🚀")
    
    processes = []
    
    # 2. TỰ ĐỘNG SINH TIẾN TRÌNH CHO TỪNG CAMERA
    for cam_id, rtsp_in in cameras_dict.items():
        # Tự động tạo link đầu ra cho MediaMTX (VD: rtsp://127.0.0.1:8554/cam_01_ai)
        rtsp_out = f"rtsp://{MEDIAMTX_HOST}/{cam_id}_ai"
        
        print(f"⏳ Đang chuẩn bị luồng cho {cam_id}...")
        
        p = multiprocessing.Process(
            target=stream_ai_to_mediamtx, 
            args=(cam_id, rtsp_in, rtsp_out, KAFKA_IP)
        )
        processes.append(p)
        p.start()
        
        # Ngủ 2-3 giây giữa mỗi cam để Card RTX 3050 có thời gian nạp model vào VRAM
        # Tránh việc nạp 5 model cùng 1 tích tắc gây văng lỗi sập bộ nhớ
        time.sleep(3) 

    # Đợi các tiến trình chạy vĩnh viễn
    for p in processes:
        p.join()