import multiprocessing
import os
import json

current_dir = os.path.dirname(__file__)
STATE_PATH =  os.path.join(os.path.dirname(current_dir), "Thong_So_Yolo.json")
print(f"Đường dẫn json: {STATE_PATH}")
with open(STATE_PATH, encoding="utf-8") as f:
    yolo = json.load(f)

# Các biến Global dùng chung cho toàn hệ thống
active_processes = []
active_cameras_data = []

# Hàng đợi siêu tốc trên RAM
tracking_queue = multiprocessing.Queue()

ai_worker_process = ""
yolo_vram = yolo["yolo11s"]["yolo"]
camera_vram = yolo["yolo11s"]["camera"]
sai_so_yolo = yolo["yolo11s"]["sai_so"]
