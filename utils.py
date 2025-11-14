import time
from collections import deque
from typing import Deque

import cv2
import numpy as np


class FPSCounter:
	"""
	Tracks frames per second over a smoothing window.
	"""

	def __init__(self, smoothing_window: int = 30) -> None:
		self.timestamps: Deque[float] = deque(maxlen=max(5, smoothing_window))

	def update(self) -> float:
		now = time.time()
		self.timestamps.append(now)
		if len(self.timestamps) <= 1:
			return 0.0
		elapsed = self.timestamps[-1] - self.timestamps[0]
		if elapsed <= 0:
			return 0.0
		return (len(self.timestamps) - 1) / elapsed


class AlertCounter:
	"""
	Counts how many times an alert was triggered (rising-edge).
	"""

	def __init__(self) -> None:
		self.count = 0
		self.last_time = 0.0

	def increment(self) -> int:
		self.count += 1
		self.last_time = time.time()
		return self.count


def draw_fps(frame_bgr: np.ndarray, fps: float) -> None:
	"""
	Draw FPS counter on the top-left corner.
	"""
	cv2.putText(
		frame_bgr,
		f"FPS: {fps:5.1f}",
		(10, 24),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 255, 0),
		2,
		cv2.LINE_AA,
	)


