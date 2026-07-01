import logging
import threading
import time
import cv2
import torch
import torchreid
import faiss
import numpy as np
import os
try:
    import onnxruntime as ort
except ImportError:
    ort = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReIDGallery:
    def __init__(
        self,
        reid_weights="osnet_x1_0_msmt17.pth",
        onnx_model="osnet_x1_0.onnx",
        sim_threshold=0.65,
        reid_top_k=3,
        ema_alpha=0.1,
        unnamed_emb_ttl=900,
        slot_capture_interval=5.0,
        max_slots=100,
    ):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ort_session = None
        self.reid_model = None

        if ort is not None and os.path.exists(onnx_model):
            logger.info(f"Loading ONNX model from {onnx_model}...")
            # Try to use TensorRT, fallback to CUDA, then CPU
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(onnx_model, providers=providers)
            self.onnx_input_name = self.ort_session.get_inputs()[0].name
        else:
            logger.info("ONNXRuntime not available or ONNX model not found. Falling back to PyTorch OSNet.")
            self.reid_model = torchreid.models.build_model(
                name="osnet_x1_0", num_classes=1000, pretrained=False
            )
            torchreid.utils.load_pretrained_weights(self.reid_model, reid_weights)
            self.reid_model.to(self.device)
            self.reid_model.eval()
            self.reid_model = self.reid_model.float()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.dim = 512
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self.embeddings: dict[int, np.ndarray] = {}
        self.next_global_id = 0

        self.SIM_THRESHOLD = sim_threshold
        self.REID_TOP_K = reid_top_k
        self.EMA_ALPHA = ema_alpha
        self.UNNAMED_EMBEDDING_TTL = unnamed_emb_ttl

        self._linked_targets: dict[int, str] = {}
        self._name_to_global_id: dict[str, int] = {}
        self.pending_targets: list[str] = []
        self.pending_capture: dict[str, int] = {}

        self.last_slot_capture_time: dict[str, float] = {}
        self.SLOT_CAPTURE_INTERVAL = slot_capture_interval
        self.next_slot_id = -1
        self.slot_faiss_id_to_name: dict[int, str] = {}
        self.MAX_SLOTS = max_slots

        # Wall-clock last-seen for unnamed embedding eviction
        self._last_seen_time: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Embedding extraction  (read-only OSNet, no lock needed)
    # ------------------------------------------------------------------

    def get_embedding(self, frame, box) -> np.ndarray | None:
        embs = self.get_embeddings(frame, [box])
        if embs and embs[0] is not None:
            return embs[0]
        return None

    def get_embeddings(self, frame, boxes) -> list[np.ndarray | None]:
        if not boxes:
            return []

        valid_boxes = []
        valid_indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("Empty crop for box %s — skipping embedding.", box)
                continue

            img = cv2.resize(crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            valid_boxes.append(img)
            valid_indices.append(i)

        results = [None] * len(boxes)
        if not valid_boxes:
            return results

        batch_imgs = np.stack(valid_boxes, axis=0).astype(np.float32) # shape: (N, 3, 256, 128)

        if self.ort_session is not None:
            ort_inputs = {self.onnx_input_name: batch_imgs}
            feat = self.ort_session.run(None, ort_inputs)[0]
            # Normalize
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            feat = feat.astype(np.float32)
        else:
            img_tensor = torch.tensor(batch_imgs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                feat = self.reid_model(img_tensor)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feat = feat.view(feat.size(0), -1)
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            feat = feat.cpu().numpy().astype(np.float32)

        for i, idx in enumerate(valid_indices):
            results[idx] = feat[i]

        return results

    # ------------------------------------------------------------------
    # FAISS match / register / update
    # ------------------------------------------------------------------

    def match_or_register(self, emb: np.ndarray) -> tuple[int, bool]:
        with self.lock:
            if self.index.ntotal == 0:
                gid = self.next_global_id
                self.next_global_id += 1
                self.embeddings[gid] = emb.copy()
                self.index.add_with_ids(emb.reshape(1, -1), np.array([gid]))
                return gid, False

            k = min(self.REID_TOP_K, self.index.ntotal)
            D, I = self.index.search(emb.reshape(1, -1), k)

            candidates: dict[int, list[float]] = {}
            for sim, gid in zip(D[0], I[0]):
                if gid != -1 and sim > self.SIM_THRESHOLD:
                    candidates.setdefault(int(gid), []).append(sim)

            if candidates:
                best_gid = max(
                    candidates,
                    key=lambda g: (len(candidates[g]), sum(candidates[g]) / len(candidates[g])),
                )
                if best_gid < 0:
                    name = self.slot_faiss_id_to_name.get(best_gid)
                    if name is not None:
                        existing_gid = self._name_to_global_id.get(name)
                        if existing_gid is not None:
                            return existing_gid, True
                        new_gid = self.next_global_id
                        self.next_global_id += 1
                        self.embeddings[new_gid] = emb.copy()
                        self.index.add_with_ids(emb.reshape(1, -1), np.array([new_gid]))
                        return new_gid, True
                return best_gid, True

            gid = self.next_global_id
            self.next_global_id += 1
            self.embeddings[gid] = emb.copy()
            self.index.add_with_ids(emb.reshape(1, -1), np.array([gid]))
            return gid, False

    def update_embedding(self, global_id: int, new_emb: np.ndarray):
        with self.lock:
            old_emb = self.embeddings[global_id]
            updated = (1 - self.EMA_ALPHA) * old_emb + self.EMA_ALPHA * new_emb
            updated /= np.linalg.norm(updated) + 1e-6
            self.embeddings[global_id] = updated

            self.index.remove_ids(np.array([global_id]))
            self.index.add_with_ids(updated.reshape(1, -1), np.array([global_id]))

    # ------------------------------------------------------------------
    # Seen tracking & unnamed eviction  (wall-clock based)
    # ------------------------------------------------------------------

    def mark_seen(self, gid: int):
        with self.lock:
            self._last_seen_time[gid] = time.time()

    def evict_unnamed(self) -> list[int]:
        with self.lock:
            now = time.time()
            stale = [
                gid
                for gid, last_time in self._last_seen_time.items()
                if gid not in self._linked_targets
                and now - last_time > self.UNNAMED_EMBEDDING_TTL
            ]
            for gid in stale:
                self.embeddings.pop(gid, None)
                self.index.remove_ids(np.array([gid]))
                self._last_seen_time.pop(gid, None)
                logger.info("UNNAMED EMBEDDING EVICTED: global_id=%d", gid)
            return stale

    # ------------------------------------------------------------------
    # Identity linking
    # ------------------------------------------------------------------

    def is_linked(self, gid: int) -> bool:
        with self.lock:
            return gid in self._linked_targets

    def get_linked_name(self, gid: int) -> str | None:
        with self.lock:
            return self._linked_targets.get(gid)

    def link_in_roi(self, gid: int, name: str) -> bool:
        with self.lock:
            if not self.pending_targets:
                return False
            if self.pending_targets[0] != name:
                return False
            old_gid = self._name_to_global_id.pop(name, None)
            if old_gid is not None:
                del self._linked_targets[old_gid]
            self.pending_targets.pop(0)
            self._linked_targets[gid] = name
            self._name_to_global_id[name] = gid
            logger.info("TRACK LINKED: '%s' → global_id=%d at ROI zone", name, gid)
            return True

    def unlink(self, gid: int) -> str | None:
        with self.lock:
            name = self._linked_targets.pop(gid, None)
            if name is not None:
                self._name_to_global_id.pop(name, None)
                logger.info("TRACK UNLINKED: '%s' (global_id=%d)", name, gid)
            return name

    def link_by_recognition(self, gid: int, name: str) -> bool:
        """Link a name to a global_id via face recognition (no pending queue)."""
        with self.lock:
            # Already linked with this exact name — nothing to do
            if self._linked_targets.get(gid) == name:
                return False

            # If this name was linked to a different gid, unlink the old one
            old_gid = self._name_to_global_id.pop(name, None)
            if old_gid is not None and old_gid != gid:
                self._linked_targets.pop(old_gid, None)

            # If this gid was linked to a different name, clean up
            old_name = self._linked_targets.get(gid)
            if old_name is not None and old_name != name:
                self._name_to_global_id.pop(old_name, None)

            # Link
            self._linked_targets[gid] = name
            self._name_to_global_id[name] = gid

            # Remove from pending queue if present (already auto-identified)
            if name in self.pending_targets:
                self.pending_targets.remove(name)

            logger.info("FACE RECOGNIZED: '%s' → global_id=%d", name, gid)
            return True

    # ------------------------------------------------------------------
    # Slot capture
    # ------------------------------------------------------------------

    def capture_slot(self, name: str, emb: np.ndarray) -> bool:
        with self.lock:
            now = time.time()
            last = self.last_slot_capture_time.get(name, 0.0)
            if now - last < self.SLOT_CAPTURE_INTERVAL:
                logger.debug("SLOT CAPTURE DELAYED: '%s' — %.1fs < %.1fs interval",
                             name, now - last, self.SLOT_CAPTURE_INTERVAL)
                return False

            slot_id = self.next_slot_id
            self.next_slot_id -= 1
            self.slot_faiss_id_to_name[slot_id] = name
            self.index.add_with_ids(emb.reshape(1, -1), np.array([slot_id]))
            self.last_slot_capture_time[name] = now
            logger.info("SLOT CAPTURED: '%s' → slot_id=%d", name, slot_id)

            while len(self.slot_faiss_id_to_name) > self.MAX_SLOTS:
                oldest_id = max(self.slot_faiss_id_to_name.keys())
                old_name = self.slot_faiss_id_to_name.pop(oldest_id)
                self.index.remove_ids(np.array([oldest_id]))
                logger.info("SLOT EVICTED: slot_id=%d for '%s'", oldest_id, old_name)

            return True

    # ------------------------------------------------------------------
    # Pending capture queue
    # ------------------------------------------------------------------

    def queue_pending_capture(self, name: str):
        with self.lock:
            self.pending_capture[name] = self.pending_capture.get(name, 0) + 1

    def consume_pending_capture(self, gid: int, emb: np.ndarray) -> bool:
        with self.lock:
            name = self._linked_targets.get(gid)
            if name is None:
                return False
            count = self.pending_capture.get(name, 0)
            if count == 0:
                return False
            if self.capture_slot(name, emb):
                self.pending_capture[name] = count - 1
                return True
            return False

    # ------------------------------------------------------------------
    # Pending target queue
    # ------------------------------------------------------------------

    def add_pending_target(self, name: str):
        with self.lock:
            if name in self._name_to_global_id:
                old_gid = self._name_to_global_id[name]
                if old_gid in self._last_seen_time and \
                time.time() - self._last_seen_time[old_gid] < 5.0:
                    self.pending_capture[name] = self.pending_capture.get(name, 0) + 1
                    return
                self._linked_targets.pop(old_gid, None)
                self._name_to_global_id.pop(name, None)
            if name not in self.pending_targets:
                    self.pending_targets.append(name)

    def has_pending_targets(self) -> bool:
        with self.lock:
            return len(self.pending_targets) > 0

    def peek_pending_target(self) -> str | None:
        with self.lock:
            return self.pending_targets[0] if self.pending_targets else None

    # ------------------------------------------------------------------
    # Query helpers (for Flask status endpoints)
    # ------------------------------------------------------------------

    def get_linked_targets_snapshot(self) -> dict[int, str]:
        with self.lock:
            return dict(self._linked_targets)

    def get_pending_targets_snapshot(self) -> list[str]:
        with self.lock:
            return list(self.pending_targets)

    def get_embedding_count(self) -> int:
        with self.lock:
            return len(self.embeddings)

    def get_index_ntotal(self) -> int:
        with self.lock:
            return self.index.ntotal
