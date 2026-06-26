# 1. Dùng base image Python có hỗ trợ CUDA (Khuyên dùng cho YOLO/ReID)
# Nếu bạn không cài bản có CUDA thì GPU của bạn sẽ không được sử dụng trong Docker!
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Cài đặt Python và các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Chuyển python3 thành lệnh python mặc định
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# 2. Tạo thư mục làm việc trong container
WORKDIR /app

# 3. Copy file requirements và cài đặt thư viện
# Hãy đảm bảo bạn ĐÃ TẠO file requirements.txt trong cùng thư mục này nhé!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 4. SỬA LỖI Ở ĐÂY: Copy toàn bộ mã nguồn hiện tại vào thư mục /app
# Dấu chấm đầu tiên (.) nghĩa là thư mục camera_jolo trên máy bạn.
# Dấu chấm thứ hai (.) nghĩa là thư mục /app trong container.
COPY . .

# 5. Lệnh chạy app
CMD ["python", "stream_ai.py"]