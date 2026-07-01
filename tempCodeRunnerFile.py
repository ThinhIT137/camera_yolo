import vlc
import time

USER = "admin"
PASSWORD = "HungVuong@2023!"
IP = "10.40.20.60"
PORT = "554"  # Cổng RTSP mặc định

# Đường dẫn luồng RTSP của Hikvision
rtsp_url = f"rtsp://{USER}:{PASSWORD}@{IP}:{PORT}/Streaming/Channels/101"

# Khởi tạo trình phát VLC
instance = vlc.Instance()
player = instance.media_player_new()
media = instance.media_new(rtsp_url)
player.set_media(media)

print("Đang mở luồng camera... Bạn sẽ nghe thấy âm thanh từ mic và thấy hình ảnh.")
player.play()

try:
    # Giữ cho chương trình chạy để nghe tiếng liên tục
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Đang dừng luồng nghe...")
    player.stop()