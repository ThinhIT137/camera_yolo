import logging
import os
import threading
import time
import subprocess
import cv2
import psutil
from flask import Flask, jsonify, render_template, Response

from Backend.camera_jolo.service.tracker import PersonTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:L26DDDDF@10.40.91.14:554/cam/realmonitor?channel=1&subtype=1",
)

latest_frame: bytes | None = None
latest_dets: list[dict] = []
frame_idx: int = 0
linked_targets: dict[int, str] = {}
pending_targets: list[str] = []
lock = threading.Lock()

tracker_app: PersonTracker | None = None

_proc = psutil.Process()
_proc.cpu_percent()


def _get_gpu_percent() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        vals = [float(v.strip()) for v in out.stdout.strip().split('\n') if v.strip()]
        return vals[0] if vals else 0.0
    except Exception:
        return 0.0


def _get_cpu_percent() -> float:
    return _proc.cpu_percent(interval=0) / psutil.cpu_count()


def _zmq_listener():
    import zmq

    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.connect("tcp://10.40.90.214:5557")

    while True:
        try:
            msg = pull.recv_json()
            if msg["action"] == "track":
                name = msg["name"]
                with lock:
                    if tracker_app is not None and name not in tracker_app.pending_targets:
                        tracker_app.pending_targets.append(name)
                        logger.info(
                            "[FLASK] Target added via ZMQ: %s — queue size: %d",
                            name,
                            len(tracker_app.pending_targets),
                        )
                    elif tracker_app is None and name not in pending_targets:
                        pending_targets.append(name)
                        logger.info(
                            "[FLASK] Target buffered (tracker loading): %s — queue size: %d",
                            name,
                            len(pending_targets),
                        )
                    else:
                        logger.info("[FLASK] Target skipped (already pending): %s", name)
        except Exception as e:
            logger.error("[FLASK] ZMQ error: %s", e)


def _track_worker():
    global latest_frame, latest_dets, frame_idx, linked_targets, pending_targets, tracker_app

    tracker_app = PersonTracker(
        model_path="yolo11s.pt",
        reid_weights="osnet_x1_0_msmt17.pth",
    )

    with lock:
        for name in pending_targets:
            if name not in tracker_app.pending_targets:
                tracker_app.pending_targets.append(name)
        pending_targets.clear()

    linked_targets = tracker_app._linked_targets
    pending_targets = tracker_app.pending_targets

    for frame, detections in tracker_app.track(RTSP_URL):
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        with lock:
            latest_frame = jpeg.tobytes()
            latest_dets = detections
            frame_idx = tracker_app.frame_idx
            linked_targets = tracker_app._linked_targets


def _generate_frames():
    while True:
        with lock:
            frame_bytes = latest_frame

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with lock:
        total_tracked = len(tracker_app.trackid_to_global) if tracker_app is not None else 0
        frame_gids = [d["global_id"] for d in latest_dets]
        linked_in_frame = sum(1 for gid in frame_gids if gid in linked_targets)
        linked_in_frame_names = [d["name"] for d in latest_dets if d["global_id"] in linked_targets]
        return jsonify(
            linked_targets={str(gid): name for gid, name in linked_targets.items()},
            pending_targets=pending_targets,
            detection_count=len(latest_dets),
            total_tracked=total_tracked,
            linked_in_frame=linked_in_frame,
            linked_in_frame_names=linked_in_frame_names,
            cpu_percent=_get_cpu_percent(),
            gpu_percent=_get_gpu_percent(),
        )


if __name__ == "__main__":
    threading.Thread(target=_zmq_listener, daemon=True).start()

    threading.Thread(target=_track_worker, daemon=True).start()

    app.run(host="10.40.90.221", port=5150, threaded=True, debug=False)
###