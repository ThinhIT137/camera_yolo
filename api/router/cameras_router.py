import time
import concurrent.futures
import multiprocessing
from fastapi import APIRouter
from pydantic import BaseModel
from service.yolo_service import ai_worker_process, calculate_optimal_chunk_size
from service.state import active_processes, active_cameras_data, tracking_queue
from service.heartbeat_service import check_rtsp_stream

router = APIRouter()

class CameraItem(BaseModel):
    id: str
    url: str

class SyncCameraPayload(BaseModel):
    cameras: list[CameraItem]

@router.post("/api/sync_cameras")
def sync_cameras(payload: SyncCameraPayload):
    print(f"\n🛑 Nhận lệnh Sync từ Master! Đang dọn dẹp {len(active_processes)} luồng YOLO cũ...")
    for p in active_processes:
        p.terminate()
        p.join()
    active_processes.clear()
    active_cameras_data.clear()

    if not payload.cameras:
        return {"message": "Đã dừng toàn bộ Camera. Server YOLO đang nghỉ ngơi."}

    accepted_cams = []
    rejected_cams = []
    cameras_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payload.cameras)) as executor:
        # Ném toàn bộ camera vào bể bơi đa luồng
        future_to_cam = {executor.submit(check_rtsp_stream, cam): cam for cam in payload.cameras}
        
        for future in concurrent.futures.as_completed(future_to_cam):
            cam = future_to_cam[future]
            try:
                # Ép timeout 5 giây! Quá 5 giây OpenCV không trả lời = Cam ngỏm / Mạng lag
                _, is_alive = future.result(timeout=5)
                
                if is_alive:
                    accepted_cams.append(cam.id)
                    cameras_dict[cam.id] = cam.url
                    
                    active_cameras_data.append({
                        "id": cam.id, 
                        "name": f"Camera {cam.id}",
                        "url": f"http://localhost:8889/{cam.id}", 
                        "ws_port": "8000" 
                    })
                    print(f"✅ [OK] Camera {cam.id} - Luồng hình ảnh ổn định.")
                else:
                    rejected_cams.append(cam.id)
                    print(f"❌ [DEAD] Camera {cam.id} - Mất tín hiệu RTSP.")
            except concurrent.futures.TimeoutError:
                rejected_cams.append(cam.id)
                print(f"⚠️ [TIMEOUT] Camera {cam.id} - Treo quá 5 giây (Bỏ qua)!")
            except Exception as e:
                rejected_cams.append(cam.id)
                print(f"❌ [ERROR] Camera {cam.id} - Lỗi không xác định: {e}")

    cameras_items = list(cameras_dict.items())
    total_cams = len(cameras_items)

    CHUNK_SIZE = calculate_optimal_chunk_size(total_cams) 
    cameras_items = list(cameras_dict.items())
    print(f"🚀 BẮT ĐẦU CHIA LÔ CHO {len(cameras_items)} CAMERA (Tối ưu: {CHUNK_SIZE} Cam/1 YOLO) 🚀")

    print(f"🚀 BẮT ĐẦU CHIA LÔ CHO {len(cameras_items)} CAMERA MỚI 🚀")
    for i in range(0, len(cameras_items), CHUNK_SIZE):
        chunk = dict    (cameras_items[i : i + CHUNK_SIZE])
        chunk_id = (i // CHUNK_SIZE) + 1
        p = multiprocessing.Process(target=ai_worker_process, args=(chunk_id, chunk, tracking_queue))
        active_processes.append(p)
        p.start()
        time.sleep(6)

    return {"message": f"Thành công! Đã khởi động {len(payload.cameras)} camera trên {len(active_processes)} luồng AI.",
            "accepted": accepted_cams,
            "rejected": rejected_cams}

@router.get("/api/cameras")
def get_cameras(): 
    return active_cameras_data

@router.get("/api/ping")
def ping():
    return {"status": "pong"}