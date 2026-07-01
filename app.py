from ultralytics import YOLO
import time
import torch

# 1. Load file .pt của bạn
model = YOLO('yolo11s.pt')

# Đưa mô hình lên GPU (nếu có)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 2. Chạy "khởi động" (Warmup) vài lần để GPU nóng máy, kết quả mới chuẩn
dummy_img = torch.zeros((1, 3, 640, 640), device=device) # Kích thước ảnh đầu vào chuẩn
for _ in range(10):
    model(dummy_img, verbose=False)

# 3. Đo tốc độ thực tế (Test khoảng 100 vòng)
start_time = time.time()
iterations = 100
for _ in range(iterations):
    model(dummy_img, verbose=False)
end_time = time.time()

# 4. Tính toán thời gian trung bình (ms)
total_time = end_time - start_time
avg_inference_time_ms = (total_time / iterations) * 1000

print(f"Thời gian xử lý trung bình: {avg_inference_time_ms:.2f} ms/ảnh")