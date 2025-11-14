import platform
import time
from typing import Optional

import cv2
import numpy as np


class AlertManager:
	"""
	Manages visual and optional audio alerts for FocusVision.
	"""

	def __init__(self, enable_sound: bool = True, use_pygame: bool = False, flash_period_seconds: float = 0.6) -> None:
		"""
		Args:
			enable_sound: Whether to play a beep when alert is active.
			use_pygame: If True, use pygame for cross-platform sound. If False, try winsound on Windows.
			flash_period_seconds: Period of flashing overlay. Half period is ON, half is OFF.
		"""
		self.enable_sound = enable_sound
		self.use_pygame = use_pygame
		self.flash_period_seconds = max(0.2, float(flash_period_seconds))
		self._last_beep_time: float = 0.0
		self._beep_interval: float = 1.0  # rate-limit beeps
		self._init_sound_backend()

	def _init_sound_backend(self) -> None:
		self._winsound = None
		self._pygame_mixer = None
		if not self.enable_sound:
			return
		if self.use_pygame:
			try:
				import pygame

				pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=256)
				self._pygame_mixer = pygame.mixer
			except Exception:
				self._pygame_mixer = None
		else:
			# Prefer winsound on Windows if available
			if platform.system().lower().startswith("win"):
				try:
					import winsound

					self._winsound = winsound
				except Exception:
					self._winsound = None

	def _beep(self) -> None:
		now = time.time()
		if now - self._last_beep_time < self._beep_interval:
			return
		self._last_beep_time = now

		if not self.enable_sound:
			return

		if self._winsound is not None:
			# 800 Hz for 150 ms
			try:
				self._winsound.Beep(800, 150)
			except Exception:
				pass
		elif self._pygame_mixer is not None:
			try:
				# Generate a simple square wave beep
				import numpy as np

				sample_rate = 22050
				duration = 0.15
				frequency = 800
				t = np.linspace(0, duration, int(sample_rate * duration), False)
				wave = 0.5 * np.sign(np.sin(2 * np.pi * frequency * t))
				audio = (wave * 32767).astype(np.int16)
				snd = self._pygame_mixer.Sound(buffer=audio.tobytes())
				snd.play()
			except Exception:
				pass

	def apply(self, frame_bgr: np.ndarray, active: bool) -> np.ndarray:
		"""
		Apply visual alert overlay and optional sound if active.

		Args:
			frame_bgr: Frame to draw on (modified in place).
			active: Whether the alert should show and sound.

		Returns:
			The frame with any overlays applied.
		"""
		if not active:
			return frame_bgr

		# Flashing logic: ON for half period, OFF for half period
		phase = (time.time() % self.flash_period_seconds) / self.flash_period_seconds
		flash_on = phase < 0.5

		if flash_on:
			overlay = frame_bgr.copy()
			cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], frame_bgr.shape[0]), (0, 0, 255), thickness=-1)
			alpha = 0.25
			cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

		# Big warning text with emoji
		message = "📵 Put your phone down and stay focused!"
		scale = max(0.8, frame_bgr.shape[1] / 1280.0)
		cv2.putText(
			frame_bgr,
			message,
			(20, int(60 * scale)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.9 * scale,
			(0, 0, 255),
			2,
			cv2.LINE_AA,
		)

		# Optional sound
		self._beep()
		return frame_bgr


