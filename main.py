import sys
import time
import argparse
from typing import Optional, Tuple, List

import cv2
import numpy as np

from detector import Detector, DetectionResult
from alert import AlertManager
from utils import FPSCounter, AlertCounter, draw_fps


def initialize_camera(preferred_size: Tuple[int, int] = (640, 480), preferred_index: Optional[int] = None, preferred_backend: Optional[str] = None) -> cv2.VideoCapture:
	"""
	Initialize the webcam by trying multiple indices and backends.

	Args:
		preferred_size: Desired (width, height) for the capture device.

	Returns:
		An initialized cv2.VideoCapture instance.

	Raises:
		RuntimeError: If the camera cannot be opened.
	"""
	print("[FocusVision] Initializing camera...")
	
	# Simple, direct approach that works based on our testing
	cam = cv2.VideoCapture(0)
	time.sleep(1)
	
	if not cam.isOpened():
		raise RuntimeError("Could not open webcam. Ensure a camera is connected and not in use.")
	
	# Set a conservative resolution
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cam.set(cv2.CAP_PROP_FPS, 30)
	time.sleep(0.5)
	
	# Test read to ensure it works
	print("[FocusVision] Testing camera...")
	for attempt in range(5):
		ret, test_frame = cam.read()
		if ret and test_frame is not None and test_frame.size > 0:
			print(f"[FocusVision] Camera initialized successfully!")
			return cam
		time.sleep(0.3)
	
	cam.release()
	raise RuntimeError("Camera opened but cannot read frames. It may be in use by another application.")


def main() -> int:
	"""
	FocusVision entry point. Captures webcam frames, detects cell phones and people,
	and issues alerts when a phone is detected.
	"""
	parser = argparse.ArgumentParser(description="FocusVision - detect phone usage via webcam")
	parser.add_argument("--camera-index", type=int, default=None, help="Preferred camera index (e.g., 0, 1, 2)")
	parser.add_argument("--backend", type=str, choices=["dshow", "msmf", "any"], default=None, help="Preferred OpenCV backend")
	args, _ = parser.parse_known_args()

	try:
		camera = initialize_camera(preferred_index=args.camera_index, preferred_backend=args.backend)
	except Exception as exc:
		print(f"[FocusVision] ERROR: {exc}")
		return 1

	# Core components
	detector = Detector(model_name="yolov8n.pt", device=None, conf_threshold=0.35)
	alert_manager = AlertManager(enable_sound=True, use_pygame=False, flash_period_seconds=0.5)
	fps_counter = FPSCounter(smoothing_window=30)
	alert_counter = AlertCounter()

	window_name = "FocusVision - Press 'q' to quit"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	
	# Load warning image
	warning_image_path = "warning.jpg"
	warning_image = cv2.imread(warning_image_path)
	if warning_image is None:
		print(f"[FocusVision] WARNING: Could not load warning image from {warning_image_path}")
	else:
		print(f"[FocusVision] Warning image loaded successfully")
	
	warning_window_name = "⚠️ WARNING - PUT YOUR PHONE DOWN! ⚠️"
	warning_window_open = False

	# Processing optimizations
	process_every_n_frames = 2  # skip-rate to improve throughput
	frame_index = 0
	last_detection: Optional[DetectionResult] = None
	last_phone_detected = False

	consecutive_failures = 0
	max_consecutive_failures = 30
	
	try:
		while True:
			ok, frame = camera.read()
			if not ok or frame is None or frame.size == 0:
				consecutive_failures += 1
				if consecutive_failures >= max_consecutive_failures:
					print("[FocusVision] ERROR: Too many consecutive frame read failures. Camera may have disconnected.")
					break
				# Don't print warning on every failure, just skip
				if consecutive_failures % 10 == 1:
					print(f"[FocusVision] WARNING: Failed to read frame from webcam (failure {consecutive_failures}).")
				time.sleep(0.05)
				continue
			
			# Reset failure counter on successful read
			consecutive_failures = 0

			# Optional: Resize for faster inference (already set to 640x480, so skip this for now)
			# target_width = 640
			# if frame.shape[1] > target_width:
			# 	scale = target_width / frame.shape[1]
			# 	frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

			# Run detection on a subset of frames to save compute
			if frame_index % process_every_n_frames == 0:
				last_detection = detector.detect(frame)
			frame_index += 1

			# Draw detections if available
			phone_detected = False
			if last_detection is not None:
				for det in last_detection.detections:
					x1, y1, x2, y2 = det.xyxy
					label = f"{det.label} {det.confidence:.2f}"
					color = (0, 255, 0) if det.label == "person" else (0, 140, 255)  # orange for phone
					cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
					cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
					if det.label == "cell phone":
						phone_detected = True

			# Manage warning window based on phone detection
			if phone_detected:
				# Show and keep showing warning image while phone is detected
				if warning_image is not None:
					if not warning_window_open:
						# First time detection - increment counter and create window
						if not last_phone_detected:
							total = alert_counter.increment()
							print(f"[FocusVision] ALERT: Phone detected! Total alerts: {total}")
						
						cv2.namedWindow(warning_window_name, cv2.WINDOW_NORMAL)
						cv2.setWindowProperty(warning_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
						warning_window_open = True
					
					# Keep showing the image while phone is detected
					cv2.imshow(warning_window_name, warning_image)
			else:
				# Close warning window immediately when phone is no longer detected
				if warning_window_open:
					cv2.destroyWindow(warning_window_name)
					warning_window_open = False
			
			last_phone_detected = phone_detected

			# Draw alert overlay / sound
			frame = alert_manager.apply(frame, active=phone_detected)

			# Show FPS
			fps = fps_counter.update()
			draw_fps(frame, fps)

			# Display
			cv2.imshow(window_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
	except KeyboardInterrupt:
		print("\n[FocusVision] Stopping (Ctrl+C)...")
	finally:
		camera.release()
		cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	sys.exit(main())


