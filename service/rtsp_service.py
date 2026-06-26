import cv2
import time
import threading

def check_rtsp_alive(url, retries=2):
    for attempt in range(retries):
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret: return True
            if cap is not None: cap.release()
        except Exception:
            pass
        time.sleep(1) 
    return False

class ReconnectWatcher:
    def __init__(self, dead_chunk):
        self.dead_chunk = dead_chunk
        self.found_alive = False
        self.running = True
        if self.dead_chunk:
            self.thread = threading.Thread(target=self.watch, daemon=True)
            self.thread.start()
            
    def watch(self):
        while self.running and self.dead_chunk:
            for cam, url in self.dead_chunk.items():
                if not self.running: break
                if check_rtsp_alive(url):
                    self.found_alive = True
                    return 
            time.sleep(2) 
            
    def stop(self):
        self.running = False
        if self.dead_chunk and self.thread.is_alive():
            self.thread.join(timeout=1)