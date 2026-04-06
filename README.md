# TF2 “Friend vs Enemy” Detector (Local Windows App)

This repository contains a local Python app for Team Fortress 2 frame capture + YOLO object detection with two classes:

- `Friend` (BLU players)
- `Enemy` (RED players)

It provides:

- Live capture + inference loop (`mss` + `ultralytics`)
- GUI controls (start/stop, thresholds, capture region, stats, preview)
- Global F3 hotkey to toggle detection
- Sanity-check frame capture
- Debug snapshot and `.jsonl` detection logging

## Requirements

- Windows 11 (supported and validated)
- Python 3.10+ (3.11 recommended)
- TF2 in Windowed (No Border), expected default resolution `2560x1440`
- YOLO weights file (Roboflow/Ultralytics compatible `.pt`)

Note: Windows 11 is currently the only supported and validated version in this repository. Windows 10 compatibility is untested.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python tf2_detector_app.py
```

## Configuration

The app reads/writes `config.json` in the repo root.

Key fields:

- `model_path`
- `monitor_index`
- `left`, `top`, `width`, `height`
- `conf_threshold`
- `iou_threshold`
- `min_box_area`
- `persistence_frames`
- `persistence_bucket_size`
- `hotkey_enabled`
- `max_fps`
- `inference_imgsz`
- `debug_dir`

Default region is:

- `left=0`
- `top=0`
- `width=2560`
- `height=1440`

## Usage

1. Start TF2 (Windowed No Border, target `2560x1440`).
2. Launch the app.
3. Set `model_path` to your trained weights (`best.pt`).
4. Click **Sanity Check (Save 3 Frames)** and verify captured images in `debug/`.
5. Click **Start** (or press **F3**) to toggle detection loop.
6. Adjust:
   - confidence threshold
   - IoU threshold
   - min box area
   - persistence frames
7. Use **Save Debug Snapshot** for image + detections JSON.

## Debug Output

By default in `debug/`:

- `sanity_*.jpg` (alignment checks)
- `snapshot_*.jpg` + `snapshot_*.json`
- `detections_YYYYMMDD.jsonl` (streamed detections)

Each detection record includes:

- `class_name`
- `confidence`
- `x1`, `y1`, `x2`, `y2` (pixel coords in capture frame)

## Notes / Scope

- This project is capture + detection + visualization only.
- No gameplay automation (aiming/shooting/online interaction) is implemented.
