## FocusVision

A lightweight, real-time computer-vision app that helps you stay focused by detecting smartphone usage from your webcam. When a phone is detected, a flashing on-screen alert is displayed and an optional beep plays.

### Features
- **YOLOv8** detection for `cell phone` and `person`
- **OpenCV** webcam capture and visualization
- **Flashing overlay and warning text** when a phone is detected
- **Optional sound** alert (Windows `winsound` or cross-platform `pygame`)
- **Alert counter** printed to console
- **Optimized for real-time** (frame resize, frame skipping, small model)

### Project Structure
- `main.py`: Entry point; webcam loop, detection, overlays, quit handling.
- `detector.py`: `Detector` class that loads YOLOv8 and runs inference.
- `alert.py`: `AlertManager` class for visual flashing and optional sound.
- `utils.py`: `FPSCounter`, `AlertCounter`, and small drawing helpers.
- `requirements.txt`: Dependencies.
- `setup.ps1`: Convenience script to create a venv and install dependencies.

### Prerequisites
- Python 3.9+ recommended
- A working webcam
- On Windows with NVIDIA GPUs, install CUDA toolkit and matching PyTorch build for best performance (optional).

### Quick Start (Windows, PowerShell)

```powershell
./setup.ps1

# Activate virtual environment for current session
.\.venv\Scripts\Activate.ps1

# Run the app
python main.py
```

If PowerShell script execution is restricted, you may need to run (once per machine, in an elevated PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Manual Setup (Any OS)

```bash
python -m venv .venv
# On Windows PowerShell: .\.venv\Scripts\Activate.ps1
# On Windows cmd: .\.venv\Scripts\activate.bat
# On macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Usage
- A window titled “FocusVision - Press 'q' to quit” will open.
- When a phone is detected, a red flashing overlay and a warning message show.
- A beep may play (configurable).
- Press `q` to exit.

### Configuration Notes
- `Detector` uses `yolov8n.pt` for speed. You can switch to `yolov8s.pt` or larger models at higher accuracy cost:
  ```python
  detector = Detector(model_name="yolov8s.pt", device=None, conf_threshold=0.35)
  ```
- To disable sound or switch to pygame-based sound:
  ```python
  alert_manager = AlertManager(enable_sound=False)
  # or
  alert_manager = AlertManager(enable_sound=True, use_pygame=True)
  ```
- Frame skipping and resizing are enabled in `main.py` for performance. Adjust `process_every_n_frames` and `target_width` as needed.

### Troubleshooting
- Webcam not found: Ensure it’s not used by other apps; try another index in `initialize_camera()`.
- Low FPS: Reduce resolution (e.g., target width ~640), increase `process_every_n_frames`, or use a smaller YOLO model.
- PyTorch/Torch install: For GPUs, install a CUDA-enabled PyTorch build matching your GPU drivers (`https://pytorch.org/get-started/locally/`).

### Future Extensions (Designed for scalability)
- Config file (YAML/JSON) for thresholds, model path, etc.
- GUI dashboard (Tkinter or PyQt) for focus stats.
- Pomodoro timer integration and session tracking.

### License
MIT


