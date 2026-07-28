#!/usr/bin/env python3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InternalPerceptionState:
    enabled: bool = False
    backend: str = "disabled"
    frame_count: int = 0
    detection_count: int = 0
    last_latency_ms: float = 0.0
    last_error: str = ""
    detections: List[Dict[str, Any]] = field(default_factory=list)


class InternalYoloDetector:
    """Optional in-process detector that never publishes debug images."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        imgsz: int = 416,
        device: str = "0",
        min_period_sec: float = 0.2,
    ):
        self.model_path = str(model_path).strip()
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.min_period_sec = max(0.0, float(min_period_sec))
        self.last_process_time = 0.0
        self.state = InternalPerceptionState(enabled=bool(self.model_path))
        self.model = None
        self.cv2 = None
        self.np = None
        if self.model_path:
            self._load_backend()

    def _load_backend(self):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            from ultralytics import YOLO  # type: ignore

            self.cv2 = cv2
            self.np = np
            self.model = YOLO(self.model_path)
            self.state.backend = "ultralytics"
        except Exception as exc:  # pragma: no cover - depends on robot image
            self.state.backend = "unavailable"
            self.state.last_error = str(exc)

    def process_image(self, msg) -> InternalPerceptionState:
        if not self.state.enabled:
            return self.state
        now = time.monotonic()
        if now - self.last_process_time < self.min_period_sec:
            return self.state
        self.last_process_time = now
        self.state.frame_count += 1
        if self.model is None or self.cv2 is None or self.np is None:
            return self.state

        started = time.perf_counter()
        try:
            image = self._message_to_array(msg)
            results = self.model.predict(
                image,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            detections = []
            names = getattr(self.model, "names", {})
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": str(names.get(cls_id, cls_id)),
                            "confidence": float(box.conf[0]),
                            "xyxy": [float(value) for value in box.xyxy[0]],
                        }
                    )
            self.state.detections = detections
            self.state.detection_count = len(detections)
            self.state.last_latency_ms = (time.perf_counter() - started) * 1000.0
            self.state.last_error = ""
        except Exception as exc:  # pragma: no cover - depends on camera encoding
            self.state.last_error = str(exc)
        return self.state

    def _message_to_array(self, msg):
        np = self.np
        if np is None:
            raise RuntimeError("numpy backend is unavailable")
        channels = 3
        encoding = str(getattr(msg, "encoding", "")).lower()
        if encoding in {"mono8", "8uc1"}:
            channels = 1
        array = np.frombuffer(msg.data, dtype=np.uint8)
        if channels == 1:
            return array.reshape((msg.height, msg.width))
        image = array.reshape((msg.height, msg.width, channels))
        if encoding == "rgb8":
            return image
        if encoding == "bgr8":
            return image[:, :, ::-1]
        return image

    def snapshot(self) -> dict:
        return {
            "enabled": self.state.enabled,
            "backend": self.state.backend,
            "frame_count": self.state.frame_count,
            "detection_count": self.state.detection_count,
            "last_latency_ms": self.state.last_latency_ms,
            "last_error": self.state.last_error,
            "detections": list(self.state.detections),
        }
