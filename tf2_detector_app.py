import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import mss
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO


DEFAULT_CONFIG_PATH = "config.json"
LOGGER = logging.getLogger(__name__)


@dataclass
class AppConfig:
    model_path: str = "best.pt"
    monitor_index: int = 1
    left: int = 0
    top: int = 0
    width: int = 2560
    height: int = 1440
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45
    min_box_area: int = 0
    persistence_frames: int = 1
    hotkey_enabled: bool = True
    hotkey: str = "f3"
    debug_dir: str = "debug"
    preview_width: int = 960
    preview_height: int = 540
    max_fps: float = 30.0

    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        cfg = cls()
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cfg
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        except Exception as ex:
            LOGGER.warning("Failed to load config file %s: %s", cfg_path, ex)
        return cfg

    def save(self, path: str) -> None:
        cfg_path = Path(path)
        cfg_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class RuntimeStats:
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    total_detections: int = 0
    friend_count: int = 0
    enemy_count: int = 0
    frame_id: int = 0


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model: Optional[YOLO] = None
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_detections: List[Detection] = []
        self.stats = RuntimeStats()
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._persistence: Dict[str, int] = {}
        self._jsonl_path: Optional[Path] = None

    def load_model(self) -> None:
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(str(model_path))

    def start(self) -> None:
        if self.running:
            return
        if self.model is None:
            self.load_model()
        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        with mss.mss() as sct:
            monitor = {
                "left": int(self.config.left),
                "top": int(self.config.top),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "mon": int(self.config.monitor_index),
            }
            frame_times: List[float] = []
            inf_times: List[float] = []
            while not self._stop_event.is_set():
                loop_start = time.time()
                raw = sct.grab(monitor)
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                capture_end = time.time()

                detections = self._infer(frame)
                detections = self._filter_by_area(detections)
                detections = self._apply_persistence(detections)
                rendered = self._draw(frame.copy(), detections)

                now = time.time()
                frame_times.append(now - loop_start)
                inf_times.append(now - capture_end)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                if len(inf_times) > 30:
                    inf_times.pop(0)

                friend_count = sum(1 for d in detections if d.class_name.lower() == "friend")
                enemy_count = sum(1 for d in detections if d.class_name.lower() == "enemy")

                with self.lock:
                    self.last_frame = rendered
                    self.last_detections = detections
                    self.stats.capture_fps = self._safe_fps(frame_times)
                    self.stats.inference_fps = self._safe_fps(inf_times)
                    self.stats.total_detections = len(detections)
                    self.stats.friend_count = friend_count
                    self.stats.enemy_count = enemy_count
                    self.stats.frame_id += 1

                self._append_jsonl(frame_id=self.stats.frame_id, detections=detections)
                self._throttle(loop_start)

    def _safe_fps(self, deltas: List[float]) -> float:
        avg = sum(deltas) / max(1, len(deltas))
        return 1.0 / avg if avg > 0 else 0.0

    def _throttle(self, started: float) -> None:
        if self.config.max_fps <= 0:
            return
        target = 1.0 / self.config.max_fps
        elapsed = time.time() - started
        sleep_for = target - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _infer(self, frame: np.ndarray) -> List[Detection]:
        assert self.model is not None
        results = self.model.predict(
            source=frame,
            conf=float(self.config.conf_threshold),
            iou=float(self.config.iou_threshold),
            verbose=False,
            imgsz=max(self.config.width, self.config.height),
        )
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        boxes = result.boxes
        if boxes is None:
            return detections
        for b in boxes:
            cls_idx = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()
            class_name = str(names.get(cls_idx, cls_idx))
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def _filter_by_area(self, detections: List[Detection]) -> List[Detection]:
        min_area = int(self.config.min_box_area)
        if min_area <= 0:
            return detections
        return [d for d in detections if d.area() >= min_area]

    def _persistence_key(self, d: Detection) -> str:
        cx, cy = d.center()
        return f"{d.class_name.lower()}:{int(cx // 24)}:{int(cy // 24)}"

    def _apply_persistence(self, detections: List[Detection]) -> List[Detection]:
        need = max(1, int(self.config.persistence_frames))
        if need <= 1:
            return detections
        seen_keys = set()
        stable: List[Detection] = []
        for d in detections:
            key = self._persistence_key(d)
            seen_keys.add(key)
            self._persistence[key] = self._persistence.get(key, 0) + 1
            if self._persistence[key] >= need:
                stable.append(d)
        for key in list(self._persistence.keys()):
            if key not in seen_keys:
                self._persistence[key] = max(0, self._persistence[key] - 1)
                if self._persistence[key] == 0:
                    del self._persistence[key]
        return stable

    def _draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for d in detections:
            cls = d.class_name.lower()
            if cls == "friend":
                color = (0, 128, 255)  # Orange for Friend class
            elif cls == "enemy":
                color = (0, 0, 255)  # Red for Enemy class
            else:
                color = (0, 255, 255)

            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (d.x1, max(20, d.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def save_debug_snapshot(self) -> Optional[Path]:
        with self.lock:
            if self.last_frame is None:
                return None
            frame = self.last_frame.copy()
            detections = list(self.last_detections)

        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = debug_dir / f"snapshot_{ts}.jpg"
        json_path = debug_dir / f"snapshot_{ts}.json"
        cv2.imwrite(str(img_path), frame)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "detections": [asdict(d) for d in detections],
            "config": asdict(self.config),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return img_path

    def save_sanity_frames(self, count: int = 3) -> List[Path]:
        out_paths: List[Path] = []
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        with mss.mss() as sct:
            monitor = {
                "left": int(self.config.left),
                "top": int(self.config.top),
                "width": int(self.config.width),
                "height": int(self.config.height),
                "mon": int(self.config.monitor_index),
            }
            for i in range(max(1, count)):
                raw = sct.grab(monitor)
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = debug_dir / f"sanity_{i + 1}_{ts}.jpg"
                cv2.imwrite(str(path), frame)
                out_paths.append(path)
        return out_paths

    def _append_jsonl(self, frame_id: int, detections: List[Detection]) -> None:
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        if self._jsonl_path is None:
            date_tag = datetime.now().strftime("%Y%m%d")
            self._jsonl_path = debug_dir / f"detections_{date_tag}.jsonl"
        record = {
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "detections": [asdict(d) for d in detections],
        }
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class AppGUI:
    def __init__(self, root: tk.Tk, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.root = root
        self.root.title("TF2 Friend vs Enemy Detector")
        self.config_path = config_path
        self.config = AppConfig.from_file(config_path)
        self.engine = DetectionEngine(self.config)
        self._preview_image = None
        self._hotkey_registered = False

        self._build_ui()
        self._bind_vars()
        self._register_hotkey_if_enabled()
        self._tick()

    def _build_ui(self) -> None:
        self.root.geometry("1250x850")
        container = ttk.Frame(self.root, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        control = ttk.Frame(container)
        control.pack(side=tk.LEFT, fill=tk.Y)

        preview = ttk.Frame(container)
        preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.start_button = ttk.Button(control, text="Start", command=self.toggle_start_stop)
        self.start_button.pack(fill=tk.X, pady=4)

        self.snapshot_button = ttk.Button(control, text="Save Debug Snapshot", command=self.on_save_snapshot)
        self.snapshot_button.pack(fill=tk.X, pady=4)

        self.sanity_button = ttk.Button(control, text="Sanity Check (Save 3 Frames)", command=self.on_sanity_check)
        self.sanity_button.pack(fill=tk.X, pady=4)

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.model_path_var = tk.StringVar(value=self.config.model_path)
        ttk.Label(control, text="Model Path").pack(anchor=tk.W)
        model_entry = ttk.Entry(control, textvariable=self.model_path_var, width=42)
        model_entry.pack(fill=tk.X, pady=2)
        ttk.Button(control, text="Browse Model", command=self.on_browse_model).pack(fill=tk.X, pady=2)

        self.monitor_var = tk.IntVar(value=self.config.monitor_index)
        self.left_var = tk.IntVar(value=self.config.left)
        self.top_var = tk.IntVar(value=self.config.top)
        self.width_var = tk.IntVar(value=self.config.width)
        self.height_var = tk.IntVar(value=self.config.height)

        self.conf_var = tk.DoubleVar(value=self.config.conf_threshold)
        self.iou_var = tk.DoubleVar(value=self.config.iou_threshold)
        self.min_area_var = tk.IntVar(value=self.config.min_box_area)
        self.persist_var = tk.IntVar(value=self.config.persistence_frames)
        self.max_fps_var = tk.DoubleVar(value=self.config.max_fps)
        self.hotkey_var = tk.BooleanVar(value=self.config.hotkey_enabled)

        for label, var in [
            ("Monitor Index", self.monitor_var),
            ("Left", self.left_var),
            ("Top", self.top_var),
            ("Width", self.width_var),
            ("Height", self.height_var),
            ("Min Box Area", self.min_area_var),
            ("Persistence Frames", self.persist_var),
        ]:
            ttk.Label(control, text=label).pack(anchor=tk.W)
            ttk.Entry(control, textvariable=var).pack(fill=tk.X, pady=2)

        ttk.Label(control, text="Confidence Threshold").pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(control, from_=0.05, to=0.99, variable=self.conf_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(control, text="IoU Threshold").pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(control, from_=0.05, to=0.99, variable=self.iou_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(control, text="Max FPS").pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(control, textvariable=self.max_fps_var).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(control, text="Enable F3 Global Hotkey", variable=self.hotkey_var, command=self.on_hotkey_toggle).pack(anchor=tk.W, pady=6)
        ttk.Button(control, text="Save Config", command=self.save_config).pack(fill=tk.X, pady=4)

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.stats_var = tk.StringVar(value="Idle")
        ttk.Label(control, textvariable=self.stats_var, justify=tk.LEFT).pack(anchor=tk.W)

        self.preview_label = ttk.Label(preview)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _bind_vars(self) -> None:
        for var in [
            self.model_path_var,
            self.monitor_var,
            self.left_var,
            self.top_var,
            self.width_var,
            self.height_var,
            self.conf_var,
            self.iou_var,
            self.min_area_var,
            self.persist_var,
            self.max_fps_var,
            self.hotkey_var,
        ]:
            var.trace_add("write", lambda *_: self._sync_config_from_ui())
        self._sync_config_from_ui()

    def _sync_config_from_ui(self) -> None:
        self.config.model_path = str(self.model_path_var.get()).strip()
        self.config.monitor_index = int(self.monitor_var.get())
        self.config.left = int(self.left_var.get())
        self.config.top = int(self.top_var.get())
        self.config.width = max(1, int(self.width_var.get()))
        self.config.height = max(1, int(self.height_var.get()))
        self.config.conf_threshold = float(self.conf_var.get())
        self.config.iou_threshold = float(self.iou_var.get())
        self.config.min_box_area = max(0, int(self.min_area_var.get()))
        self.config.persistence_frames = max(1, int(self.persist_var.get()))
        self.config.max_fps = max(1.0, float(self.max_fps_var.get()))
        self.config.hotkey_enabled = bool(self.hotkey_var.get())

    def on_browse_model(self) -> None:
        path = filedialog.askopenfilename(title="Select YOLO Weights", filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_path_var.set(path)

    def save_config(self) -> None:
        self._sync_config_from_ui()
        self.config.save(self.config_path)
        messagebox.showinfo("Config", f"Saved to {self.config_path}")

    def on_sanity_check(self) -> None:
        self._sync_config_from_ui()
        try:
            paths = self.engine.save_sanity_frames(3)
            messagebox.showinfo("Sanity Check", "\n".join(str(p) for p in paths))
        except Exception as ex:
            messagebox.showerror("Sanity Check Error", str(ex))

    def on_save_snapshot(self) -> None:
        path = self.engine.save_debug_snapshot()
        if path is None:
            messagebox.showwarning("Snapshot", "No frame available yet.")
            return
        messagebox.showinfo("Snapshot", f"Saved: {path}")

    def toggle_start_stop(self) -> None:
        self._sync_config_from_ui()
        if self.engine.running:
            self.engine.stop()
            self.start_button.configure(text="Start")
            return
        try:
            self.engine = DetectionEngine(self.config)
            self.engine.start()
            self.start_button.configure(text="Stop")
        except Exception as ex:
            messagebox.showerror("Start Error", str(ex))

    def _register_hotkey_if_enabled(self) -> None:
        if self._hotkey_registered:
            return
        if not self.config.hotkey_enabled:
            return
        keyboard.add_hotkey(self.config.hotkey, self.toggle_start_stop, suppress=False, trigger_on_release=True)
        self._hotkey_registered = True

    def _unregister_hotkey(self) -> None:
        if not self._hotkey_registered:
            return
        keyboard.clear_hotkey(self.config.hotkey)
        self._hotkey_registered = False

    def on_hotkey_toggle(self) -> None:
        self._sync_config_from_ui()
        if self.config.hotkey_enabled:
            self._register_hotkey_if_enabled()
        else:
            self._unregister_hotkey()

    def _tick(self) -> None:
        with self.engine.lock:
            frame = None if self.engine.last_frame is None else self.engine.last_frame.copy()
            stats = RuntimeStats(**asdict(self.engine.stats))
        self._render_preview(frame)
        self.stats_var.set(
            "\n".join(
                [
                    f"Status: {'Running' if self.engine.running else 'Stopped'}",
                    f"Capture FPS: {stats.capture_fps:.2f}",
                    f"Inference FPS: {stats.inference_fps:.2f}",
                    f"Total Detections: {stats.total_detections}",
                    f"Friend Count: {stats.friend_count}",
                    f"Enemy Count: {stats.enemy_count}",
                ]
            )
        )
        self.root.after(50, self._tick)

    def _render_preview(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        pw, ph = self.config.preview_width, self.config.preview_height
        h, w = frame.shape[:2]
        scale = min(pw / w, ph / h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._preview_image = ImageTk.PhotoImage(image=img)
        self.preview_label.configure(image=self._preview_image)

    def on_close(self) -> None:
        try:
            if self.engine.running:
                self.engine.stop()
        finally:
            self._unregister_hotkey()
            self.root.destroy()


def ensure_dirs(config: AppConfig) -> None:
    Path(config.debug_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = AppConfig.from_file(DEFAULT_CONFIG_PATH)
    ensure_dirs(cfg)
    root = tk.Tk()
    app = AppGUI(root, config_path=DEFAULT_CONFIG_PATH)
    root.mainloop()


if __name__ == "__main__":
    main()
