from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - import error surfaced at runtime
	YOLO = None  # type: ignore


@dataclass
class Detection:
	xyxy: Tuple[int, int, int, int]
	confidence: float
	label: str


@dataclass
class DetectionResult:
	detections: List[Detection]


class Detector:
	"""
	YOLOv8-based object detector focused on 'cell phone' and 'person'.
	"""

	def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None, conf_threshold: float = 0.35) -> None:
		"""
		Args:
			model_name: Model weights (local path or Ultralytics hub name).
			device: Device to run on ('cpu', 'cuda', or specific like 'cuda:0'). If None, YOLO decides.
			conf_threshold: Confidence threshold for filtering detections.
		"""
		if YOLO is None:
			raise RuntimeError(
				"Ultralytics is not available. Please install it via 'pip install ultralytics'."
			)
		self.model = YOLO(model_name)
		self.device = device
		self.conf_threshold = conf_threshold
		# Cache class name mapping
		self.class_names = self.model.names if hasattr(self.model, "names") else {}
		self.target_labels = {"cell phone", "person"}

	def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
		"""
		Run detection on a single BGR frame.

		Args:
			frame_bgr: Input frame in BGR format (OpenCV default).

		Returns:
			DetectionResult with filtered detections.
		"""
		img = frame_bgr
		results = self.model.predict(
			source=img,
			verbose=False,
			device=self.device,
			conf=self.conf_threshold,
			iou=0.45,
			imgsz=640,
		)

		detections: List[Detection] = []
		if not results:
			return DetectionResult(detections=[])

		result = results[0]
		if not hasattr(result, "boxes") or result.boxes is None:
			return DetectionResult(detections=[])

		boxes = result.boxes
		xyxy = boxes.xyxy.cpu().numpy().astype(int)
		confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
		clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

		for i in range(xyxy.shape[0]):
			class_id = clss[i]
			label = self.class_names.get(int(class_id), str(class_id))
			if label not in self.target_labels:
				continue
			x1, y1, x2, y2 = xyxy[i].tolist()
			conf = float(confs[i]) if i < len(confs) else 0.0
			detections.append(Detection(xyxy=(x1, y1, x2, y2), confidence=conf, label=label))

		return DetectionResult(detections=detections)


