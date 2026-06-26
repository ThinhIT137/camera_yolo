import os
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import set_key

# Import config để sửa trực tiếp trên RAM
from service.heartbeat_service import config

# DÙNG ABSPATH ĐỂ KHÓA CHẾT ĐƯỜNG DẪN RA NGOÀI CÙNG (.env)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")

router = APIRouter()

# Đã thêm server_id vào bộ khung JSON
class UpdateMasterPayload(BaseModel):
    new_master_url: str
    server_id: str

@router.post("/api/update_master_url")
def api_update_master_url(payload: UpdateMasterPayload):
    new_url = payload.new_master_url
    new_id = payload.server_id
    
    # 1. Sửa trực tiếp biến trên RAM cho nhịp tim quay xe lập tức
    config["MASTER_URL"] = new_url
    config["SERVER_ID"] = new_id
    
    # 2. Ghi đè vĩnh viễn vào file .env
    try:
        set_key(ENV_PATH, "MASTER_URL", new_url)
        set_key(ENV_PATH, "SERVER_ID", new_id) # Ghi thêm dòng Server ID vào .env
        
        print(f"\n🔄 [CẬP NHẬT] Nhận lệnh từ Master! Đổi đích: {new_url} | ID: {new_id}\n")
        return {
            "status": "success", 
            "message": f"Dạ em đã cập nhật. URL: {new_url}, ID: {new_id}"
        }
    except Exception as e:
        print(f"❌ [LỖI WORKER] Không thể ghi file .env: {e}")
        return {
            "status": "error", 
            "message": f"Không thể ghi đè file .env: {e}"
        }