import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_global_system_logger(log_folder="logs", log_file_prefix="master-node"):
    """
    Cấu hình Root Logger tự động tạo file log mới theo ngày-tháng-năm-giờ.phút.giây mỗi khi khởi động.
    Sử dụng os.environ để chống lệch giây giữa tiến trình Mẹ và tiến trình YOLO con trên Windows.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        
    # 🔥 BẢO BỐI CHỐNG LỆCH GIÂY ĐA TIẾN TRÌNH:
    # Nếu là tiến trình con được spawn ra, nó sẽ bú lại chính xác tên file của tiến trình mẹ
    env_key = f"CURRENT_LOG_FILE_{log_file_prefix.upper().replace('-', '_')}"
    
    if env_key in os.environ:
        final_log_file = os.environ[env_key]
    else:
        # Nếu là tiến trình mẹ chạy lần đầu -> Khắc tên ngày giờ chuẩn chỉnh
        # Định dạng: prefix-26-6-2026-10.05.30.log (khớp ý bro)
        current_time = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
        final_log_file = f"{log_file_prefix}-{current_time}.log"
        # Khóa chết tên file này vào bộ nhớ môi trường để các tiến trình con kế thừa
        os.environ[env_key] = final_log_file

    log_path = os.path.join(log_folder, final_log_file)
    
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) 
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 1. Handler in ra màn hình Terminal (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO) 
    
    # 2. Handler ghi vào File ngầm 
    # (Vẫn giữ cờ 5MB xoay vòng để phòng trường hợp server chạy cả tháng file log phình to mấy chục GB làm nghẽn ổ cứng)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) 
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Bẫy lỗi sập nguồn đột ngột
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("🚨 [HỆ THỐNG SẬP NGUỒN ĐỘT NGỘT] Lỗi chưa được xử lý:", 
                             exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_unhandled_exception
    
    return root_logger