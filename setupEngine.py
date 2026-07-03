import os
from ultralytics import YOLO

def get_optimized_model(model_name="yolo11n.pt"):
    # Đường dẫn file engine tương ứng
    engine_name = model_name.replace(".pt", ".engine")
    
    # Kiểm tra xem file engine đã tồn tại chưa
    if os.path.exists(engine_name):
        print(f"🚀 [YOLO] Phát hiện file Engine! Đang nạp {engine_name} (Tốc độ tối đa)")
        # Khi nạp engine, không cần task="detect" vì nó đã được đóng gói sẵn kiến trúc
        return YOLO(engine_name)
    else:
        print(f"⚠️ [YOLO] Chưa có file .engine. Đang nạp {model_name} (Chế độ thường).")
        return YOLO(model_name, task="detect")

# Sửa lại dòng khởi tạo yolo_model của sếp thành:
yolo_model = get_optimized_model("yolo11n.pt")

cmd = "yolo export model=yolo11n.pt format=engine imgsz=480 half=True device=0 simplify=True"