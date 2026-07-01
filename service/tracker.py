import logging
import cv2
import numpy as np

from gallery import ReIDGallery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraTracker:
    def __init__(
        self,
        cam_id: str,
        source_url: str,
        gallery: ReIDGallery,
        yolo_model,
        roi: tuple[int, int, int, int],
        door_poly: np.ndarray,
        tracker_config: str = "custom_tracker.yaml",
        occlusion_iou_threshold: float = 0.6,
        stability_frames: int = 3,
        track_ttl: int = 150,
        roi_entry_stability: int = 3,
        imgsz: int = 640,
        conf: float = 0.15,
        face_detector=None,
        face_recognizer=None,
        face_recognition_interval: int = 15,
    ):
        self.cam_id = cam_id
        self.source_url = source_url
        self.gallery = gallery
        self.model = yolo_model
        self.tracker_config = tracker_config
        self.ROI = roi
        self.DOOR_POLYGON = door_poly

        self.OCCLUSION_IOU_THRESHOLD = occlusion_iou_threshold
        self.STABILITY_FRAMES = stability_frames
        self.TRACK_TTL = track_ttl
        self.ROI_ENTRY_STABILITY = roi_entry_stability
        self.IMGSZ = imgsz
        self.CONF = conf

        # Face detection + recognition (optional, per-camera)
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.face_recognition_interval = face_recognition_interval
        self._face_cooldown: dict[int, int] = {}  # global_id → next allowed frame

        self.trackid_to_global: dict[int, int] = {}
        self.track_history: dict[int, int] = {}
        self.track_last_seen: dict[int, int] = {}
        self.track_last_position: dict[int, tuple[int, int]] = {}
        self.roi_entry_frames: dict[int, int] = {}
        self.frame_idx = 0

        self.COLOR_PALETTE = np.array([
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 255), (150, 255, 50),
        ], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_roi(self, point: tuple[int, int]) -> bool:
        x, y = point
        rx1, ry1, rx2, ry2 = self.ROI
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    def _in_door_area(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.DOOR_POLYGON, point, False) >= 0

    def get_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    # ------------------------------------------------------------------
    # Stale track eviction
    # ------------------------------------------------------------------

    def _evict_stale_tracks(self):
        stale = [
            t_id
            for t_id, last in self.track_last_seen.items()
            if self.frame_idx - last > self.TRACK_TTL
        ]
        for t_id in stale:
            pos = self.track_last_position.pop(t_id, None)
            gid = self.trackid_to_global.pop(t_id, None)
            if gid is not None:
              self.gallery.unlink(gid)
              self._face_cooldown.pop(gid, None)
            self.track_history.pop(t_id, None)
            self.track_last_seen.pop(t_id, None)

        evicted = self.gallery.evict_unnamed()
        for gid in evicted:
            dead = [t for t, g in self.trackid_to_global.items() if g == gid]
            for t in dead:
                self.trackid_to_global.pop(t, None)
                self.track_history.pop(t, None)
                self.track_last_seen.pop(t, None)
                self.track_last_position.pop(t, None)

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track_loop(self):
        results = self.model.track(
            source=self.source_url,
            conf=self.CONF,
            imgsz=self.IMGSZ,
            stream=True,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],
        )

        for r in results:
            self.frame_idx += 1
            frame = r.orig_img
            current_detections = []

            self._evict_stale_tracks()

            assigned_in_frame: dict[int, float] = {}

            if r.boxes is not None and r.boxes.id is not None:
                data = r.boxes.data.cpu().numpy()
                all_boxes = data[:, :4]
                order = np.argsort(-data[:, 5])
                data = data[order]

                # --- PHASE 1: Collect ReID requests ---
                reid_requests = []
                for row in data:
                    x1, y1, x2, y2 = row[:4]
                    track_id = int(row[4])
                    box = [int(x1), int(y1), int(x2), int(y2)]

                    self.track_last_seen[track_id] = self.frame_idx
                    self.track_last_position[track_id] = (int((x1 + x2) / 2), int(y2))

                    need_reid = False
                    if track_id in self.trackid_to_global:
                        need_reid = True
                    else:
                        is_occluded = any(
                            not np.array_equal(box, other.astype(int))
                            and self.get_iou(box, other) > self.OCCLUSION_IOU_THRESHOLD
                            for other in all_boxes
                        )
                        if not is_occluded:
                            self.track_history[track_id] = self.track_history.get(track_id, 0) + 1
                            if self.track_history[track_id] >= self.STABILITY_FRAMES:
                                need_reid = True
                    
                    if need_reid:
                        reid_requests.append({'track_id': track_id, 'box': box})

                # --- PHASE 2: Batch ReID inference ---
                embeddings_dict = {}
                if reid_requests:
                    boxes_to_infer = [req['box'] for req in reid_requests]
                    embs = self.gallery.get_embeddings(frame, boxes_to_infer)
                    for req, emb in zip(reid_requests, embs):
                        embeddings_dict[req['track_id']] = emb

                # --- PHASE 3: Process detections ---
                for row in data:
                    x1, y1, x2, y2 = row[:4]
                    track_id = int(row[4])
                    score = float(row[5])
                    box = [int(x1), int(y1), int(x2), int(y2)]

                    global_id = None
                    emb = embeddings_dict.get(track_id)

                    if track_id in self.trackid_to_global:
                        global_id = self.trackid_to_global[track_id]
                        if emb is not None:
                            emb /= np.linalg.norm(emb) + 1e-6
                            self.gallery.update_embedding(global_id, emb)
                            self.gallery.mark_seen(global_id)
                    else:
                        is_occluded = any(
                            not np.array_equal(box, other.astype(int))
                            and self.get_iou(box, other) > self.OCCLUSION_IOU_THRESHOLD
                            for other in all_boxes
                        )
                        if not is_occluded and self.track_history.get(track_id, 0) >= self.STABILITY_FRAMES:
                            if emb is not None:
                                emb /= np.linalg.norm(emb) + 1e-6
                                best_gid, was_matched = self.gallery.match_or_register(emb)
                                if was_matched:
                                    self.gallery.update_embedding(best_gid, emb)
                                    self.gallery.mark_seen(best_gid)
                                self.trackid_to_global[track_id] = best_gid
                                self.gallery.mark_seen(best_gid)
                                global_id = best_gid

                    if global_id is not None:
                        if global_id in assigned_in_frame:
                            if score <= assigned_in_frame[global_id]:
                                continue
                        assigned_in_frame[global_id] = score

                        base_point = (int((x1 + x2) / 2), int(y2))
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                        # ─── FACE RECOGNITION (auto-link) ───
                        if (self.face_detector is not None
                                and self.face_recognizer is not None
                                and not self.gallery.is_linked(global_id)
                                and self.frame_idx >= self._face_cooldown.get(global_id, 0)):

                            bx1, by1, bx2, by2 = box
                            body_crop = frame[by1:by2, bx1:bx2]

                            if body_crop.size > 0:
                                detected_faces = self.face_detector.detect(body_crop)

                                if detected_faces:
                                    best_face = detected_faces[0]
                                    passed, _ = self.face_detector.quality_check(
                                        body_crop, best_face
                                    )

                                    # Translate face coords to frame space for drawing
                                    fx = best_face["x"] + bx1
                                    fy = best_face["y"] + by1
                                    fw, fh = best_face["w"], best_face["h"]

                                    if passed:
                                        face_crop = self.face_detector.crop_face(
                                            body_crop, best_face
                                        )
                                        if face_crop is not None and face_crop.size > 0:
                                            rec_name, rec_conf = (
                                                self.face_recognizer.recognize(face_crop)
                                            )

                                            if rec_name != "Unknown":
                                                self.gallery.link_by_recognition(
                                                    global_id, rec_name
                                                )
                                                # Capture ReID slot for future re-matching
                                                if emb is not None:
                                                    self.gallery.capture_slot(
                                                        rec_name, emb
                                                    )
                                                # Green face box — recognized
                                                cv2.rectangle(
                                                    frame,
                                                    (fx, fy),
                                                    (fx + fw, fy + fh),
                                                    (0, 255, 0), 2,
                                                )
                                                face_label = (
                                                    f"{rec_name} ({rec_conf:.2f})"
                                                )
                                                (tw, th), _ = cv2.getTextSize(
                                                    face_label,
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5, 1,
                                                )
                                                cv2.rectangle(
                                                    frame,
                                                    (fx, fy - th - 6),
                                                    (fx + tw, fy),
                                                    (0, 255, 0), -1,
                                                )
                                                cv2.putText(
                                                    frame,
                                                    face_label,
                                                    (fx, fy - 4),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5, (0, 0, 0), 1,
                                                )
                                            else:
                                                # Yellow face box — detected but unknown
                                                cv2.rectangle(
                                                    frame,
                                                    (fx, fy),
                                                    (fx + fw, fy + fh),
                                                    (0, 255, 255), 2,
                                                )
                                    else:
                                        # Red face box — quality check failed
                                        cv2.rectangle(
                                            frame,
                                            (fx, fy),
                                            (fx + fw, fy + fh),
                                            (0, 0, 255), 1,
                                        )

                            self._face_cooldown[global_id] = (
                                self.frame_idx + self.face_recognition_interval
                            )

                        # ─── ROI LINKING (fallback) ───
                        if not self.gallery.is_linked(global_id):
                            if self._in_roi(center):
                                self.roi_entry_frames[global_id] = (
                                    self.roi_entry_frames.get(global_id, 0) + 1
                                )
                            else:
                                self.roi_entry_frames[global_id] = 0

                            if self.roi_entry_frames.get(global_id, 0) >= self.ROI_ENTRY_STABILITY:
                                pending_name = self.gallery.peek_pending_target()
                                if pending_name is not None:
                                    if self.gallery.link_in_roi(global_id, pending_name):
                                        if emb is not None:
                                            self.gallery.capture_slot(pending_name, emb)
                                        self.roi_entry_frames[global_id] = 0

                        # ─── Pending capture for already-linked targets ───
                        if emb is not None:
                            self.gallery.consume_pending_capture(global_id, emb)

                        display_name = self.gallery.get_linked_name(global_id)
                        if display_name is None:
                            display_name = f"Unknown (ID {global_id})"

                        current_detections.append({
                            "id": track_id,
                            "global_id": global_id,
                            "bbox": box,
                            "base_point": base_point,
                            "name": display_name,
                        })

                        color_idx = global_id % len(self.COLOR_PALETTE)
                        box_color = tuple(int(c) for c in self.COLOR_PALETTE[color_idx])

                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 3)
                        label = f"TRACKING: {display_name}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (box[0], box[1] - th - 10), (box[0] + tw, box[1]), box_color, -1)
                        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
                    else:
                        prov_color = (160, 160, 160)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), prov_color, 2)
                        label = f"Detecting (track {track_id})"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (box[0], box[1] - th - 10), (box[0] + tw, box[1]), prov_color, -1)
                        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)                       
            rx1, ry1, rx2, ry2 = self.ROI
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(frame, "TRACK ZONE", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.polylines(frame, [self.DOOR_POLYGON], True, (0, 0, 255), 2)
            cx = int(np.mean(self.DOOR_POLYGON[:, 0]))
            cv2.putText(frame, "DOOR ZONE", (cx, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            yield frame, current_detections
