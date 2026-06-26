import multiprocessing

# Các biến Global dùng chung cho toàn hệ thống
active_processes = []
active_cameras_data = []

# Hàng đợi siêu tốc trên RAM
tracking_queue = multiprocessing.Queue()

ai_worker_process = ""
calculate_optimal_chunk_size = ""