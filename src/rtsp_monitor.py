"""
Streaming safety monitor with reconnect, bounded frame queue, and health metrics.

This is the "production-shaped" version of the demo monitor:
- a capture thread reads webcam/RTSP/video frames;
- a tiny queue keeps only the newest frames to avoid growing latency;
- the inference loop runs YOLO person tracking plus the fight model;
- health metrics are written to CSV and shown on the overlay.
"""

import argparse
import csv
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with `pip install ultralytics` "
        "or run `pip install -r requirements.txt`."
    ) from exc

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FightDetector
from src.safety_monitor import (
    PERSON_CLASS_ID,
    clamp_box,
    draw_label_box,
    draw_status,
    extract_people,
    yolo_device,
)


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_index: int
    captured_at: float


def parse_source(source):
    return int(source) if source.isdigit() else source


def is_file_source(source):
    return not source.isdigit() and Path(source).exists()


def select_torch_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StreamStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "STARTING"
        self.last_error = ""
        self.frames_read = 0
        self.dropped_frames = 0
        self.reconnect_count = 0
        self.last_frame_time = 0.0
        self.capture_times = deque(maxlen=60)

    def set_status(self, status, error=""):
        with self.lock:
            self.status = status
            self.last_error = error

    def mark_frame(self, timestamp):
        with self.lock:
            self.frames_read += 1
            self.last_frame_time = timestamp
            self.capture_times.append(timestamp)
            self.status = "LIVE"
            self.last_error = ""

    def mark_drop(self):
        with self.lock:
            self.dropped_frames += 1

    def mark_reconnect(self):
        with self.lock:
            self.reconnect_count += 1
            self.status = "RECONNECTING"

    def snapshot(self):
        with self.lock:
            times = list(self.capture_times)
            capture_fps = 0.0
            if len(times) > 1 and times[-1] > times[0]:
                capture_fps = (len(times) - 1) / (times[-1] - times[0])
            return {
                "status": self.status,
                "last_error": self.last_error,
                "frames_read": self.frames_read,
                "dropped_frames": self.dropped_frames,
                "reconnect_count": self.reconnect_count,
                "last_frame_time": self.last_frame_time,
                "capture_fps": capture_fps,
            }


class RollingFightDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.model = FightDetector(num_classes=2, pretrained=False).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, frame_buffer):
        frames_np = np.stack(frame_buffer).astype(np.float32)
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(frames_tensor)
            probs = torch.softmax(logits, dim=1)

        return float(probs[0, 1].item())


class CaptureWorker(threading.Thread):
    def __init__(self, args, frame_queue, stop_event, stats):
        super().__init__(daemon=True)
        self.args = args
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.stats = stats
        self.file_source = is_file_source(args.source)

    def put_latest(self, packet):
        while not self.stop_event.is_set():
            try:
                self.frame_queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.stats.mark_drop()
                except queue.Empty:
                    return

    def run(self):
        source = parse_source(self.args.source)
        frame_index = 0
        backoff = self.args.reconnect_initial_delay

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                message = "could not open source"
                self.stats.set_status("RECONNECTING" if self.args.reconnect else "ERROR", message)
                if not self.args.reconnect:
                    self.stop_event.set()
                    break
                self.stats.mark_reconnect()
                time.sleep(backoff)
                backoff = min(backoff * 2, self.args.reconnect_max_delay)
                continue

            self.stats.set_status("LIVE")
            backoff = self.args.reconnect_initial_delay
            source_fps = cap.get(cv2.CAP_PROP_FPS) or self.args.file_fps
            frame_interval = 1.0 / max(source_fps, 1.0)

            try:
                while not self.stop_event.is_set():
                    read_started = time.time()
                    ret, frame = cap.read()

                    if not ret:
                        if self.file_source:
                            self.stats.set_status("EOF")
                            self.stop_event.set()
                            break
                        self.stats.set_status("RECONNECTING", "read failed")
                        break

                    now = time.time()
                    self.stats.mark_frame(now)
                    self.put_latest(FramePacket(frame=frame, frame_index=frame_index, captured_at=now))
                    frame_index += 1

                    if self.file_source and self.args.pace_file:
                        elapsed = time.time() - read_started
                        time.sleep(max(0.0, frame_interval - elapsed))

            finally:
                cap.release()

            if self.stop_event.is_set():
                break

            if not self.args.reconnect:
                self.stop_event.set()
                break

            self.stats.mark_reconnect()
            time.sleep(backoff)
            backoff = min(backoff * 2, self.args.reconnect_max_delay)


def append_prediction_rows(rows, metrics, people, width, height):
    base = {
        "time_sec": f"{metrics['time_sec']:.2f}",
        "capture_frame": metrics["capture_frame"],
        "processed_frame": metrics["processed_frame"],
        "stream_status": metrics["stream_status"],
        "fight_label": metrics["fight_label"],
        "p_fight_raw": f"{metrics['p_fight_raw']:.4f}",
        "p_fight_smooth": f"{metrics['p_fight_smooth']:.4f}",
        "people_count": len(people),
        "latency_ms": f"{metrics['latency_ms']:.1f}",
        "capture_fps": f"{metrics['capture_fps']:.2f}",
        "inference_fps": f"{metrics['inference_fps']:.2f}",
        "queue_size": metrics["queue_size"],
        "dropped_frames": metrics["dropped_frames"],
        "reconnect_count": metrics["reconnect_count"],
    }

    if not people:
        rows.append({
            **base,
            "person_id": "",
            "person_confidence": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
        })
        return

    for person in people:
        x1, y1, x2, y2 = clamp_box(person["box"], width, height)
        rows.append({
            **base,
            "person_id": person["person_id"],
            "person_confidence": f"{person['confidence']:.4f}",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })


def append_health_row(rows, metrics):
    rows.append({
        "time_sec": f"{metrics['time_sec']:.2f}",
        "stream_status": metrics["stream_status"],
        "capture_fps": f"{metrics['capture_fps']:.2f}",
        "inference_fps": f"{metrics['inference_fps']:.2f}",
        "latency_ms": f"{metrics['latency_ms']:.1f}",
        "last_frame_age_ms": f"{metrics['last_frame_age_ms']:.1f}",
        "queue_size": metrics["queue_size"],
        "dropped_frames": metrics["dropped_frames"],
        "reconnect_count": metrics["reconnect_count"],
        "frames_read": metrics["frames_read"],
        "frames_processed": metrics["processed_frame"],
    })


def draw_health(frame, metrics):
    h, w = frame.shape[:2]
    status = metrics["stream_status"]
    color = (40, 220, 80)
    if status in {"STALE", "RECONNECTING"}:
        color = (0, 220, 255)
    elif status in {"ERROR", "EOF"}:
        color = (0, 0, 255)

    font_scale = max(0.42, min(0.75, w / 1050.0))
    thickness = max(1, int(round(font_scale * 3)))
    line = (
        f"{status} cap={metrics['capture_fps']:.1f} infer={metrics['inference_fps']:.1f} "
        f"lat={metrics['latency_ms']:.0f}ms drop={metrics['dropped_frames']} "
        f"rec={metrics['reconnect_count']}"
    )
    y = max(24, h - 16)
    cv2.putText(
        frame,
        line,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def write_csv(path, rows, fieldnames, label):
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {label}: {csv_path}")


def run(args):
    fight_device = select_torch_device(args.fight_device)
    print(f"Fight model device: {fight_device}")
    print(f"YOLO device: {args.yolo_device}")
    print(f"Source: {args.source}")
    print(f"Queue size: {args.queue_size}")

    fight_detector = RollingFightDetector(args.fight_model, fight_device)
    yolo_model = YOLO(args.yolo_model)

    frame_queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()
    stats = StreamStats()
    capture = CaptureWorker(args, frame_queue, stop_event, stats)
    capture.start()

    writer = None
    prediction_rows = []
    health_rows = []
    frame_buffer = deque(maxlen=args.num_frames)
    fight_history = deque(maxlen=args.smooth_window)
    inference_times = deque(maxlen=60)

    start_time = time.time()
    deadline = start_time + args.duration if args.duration > 0 else None
    next_health_log = start_time
    processed_count = 0
    current_raw_prob = 0.0
    current_smooth_prob = 0.0
    current_fight_label = "Collecting"
    last_people = []

    try:
        while not stop_event.is_set() or not frame_queue.empty():
            now = time.time()
            if deadline and now >= deadline:
                stop_event.set()
                break

            try:
                packet = frame_queue.get(timeout=args.queue_timeout)
            except queue.Empty:
                snapshot = stats.snapshot()
                last_age = now - snapshot["last_frame_time"] if snapshot["last_frame_time"] else 0.0
                if snapshot["status"] == "LIVE" and last_age > args.stale_after:
                    stats.set_status("STALE")
                continue

            now = time.time()
            elapsed = now - start_time
            latency_ms = (now - packet.captured_at) * 1000.0
            inference_times.append(now)
            inference_fps = 0.0
            if len(inference_times) > 1 and inference_times[-1] > inference_times[0]:
                inference_fps = (len(inference_times) - 1) / (inference_times[-1] - inference_times[0])

            frame = packet.frame
            height, width = frame.shape[:2]

            if writer is None and args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args.output_fps, (width, height))

            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb.astype(np.float32) / 255.0)

            if len(frame_buffer) == args.num_frames and processed_count % args.fight_stride == 0:
                current_raw_prob = fight_detector.predict(list(frame_buffer))
                fight_history.append(current_raw_prob)
                current_smooth_prob = float(np.mean(fight_history))
                current_fight_label = "FIGHT" if current_smooth_prob >= args.threshold else "Normal"

            if processed_count % max(args.yolo_every, 1) == 0:
                predict_kwargs = {
                    "classes": [PERSON_CLASS_ID],
                    "conf": args.yolo_conf,
                    "imgsz": args.yolo_imgsz,
                    "device": yolo_device(args.yolo_device),
                    "verbose": False,
                }
                if args.track:
                    yolo_results = yolo_model.track(frame, persist=True, **predict_kwargs)
                else:
                    yolo_results = yolo_model.predict(frame, **predict_kwargs)
                last_people = extract_people(yolo_results[0])

            for person in last_people:
                person_id = person["person_id"]
                label = f"person {person_id}" if person_id != "" else "person"
                draw_label_box(frame, person["box"], label, person["confidence"])

            snapshot = stats.snapshot()
            last_frame_age_ms = 0.0
            if snapshot["last_frame_time"]:
                last_frame_age_ms = max(0.0, (now - snapshot["last_frame_time"]) * 1000.0)
            stream_status = snapshot["status"]
            if stream_status == "LIVE" and last_frame_age_ms > args.stale_after * 1000.0:
                stream_status = "STALE"

            metrics = {
                "time_sec": elapsed,
                "capture_frame": packet.frame_index,
                "processed_frame": processed_count,
                "stream_status": stream_status,
                "fight_label": current_fight_label,
                "p_fight_raw": current_raw_prob,
                "p_fight_smooth": current_smooth_prob,
                "latency_ms": latency_ms,
                "last_frame_age_ms": last_frame_age_ms,
                "capture_fps": snapshot["capture_fps"],
                "inference_fps": inference_fps,
                "queue_size": frame_queue.qsize(),
                "dropped_frames": snapshot["dropped_frames"],
                "reconnect_count": snapshot["reconnect_count"],
                "frames_read": snapshot["frames_read"],
            }

            draw_status(
                frame,
                current_fight_label,
                current_smooth_prob,
                current_raw_prob,
                len(last_people),
                inference_fps,
                args.threshold,
            )
            draw_health(frame, metrics)

            append_prediction_rows(prediction_rows, metrics, last_people, width, height)

            if now >= next_health_log:
                append_health_row(health_rows, metrics)
                next_health_log = now + args.health_interval

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Safe Camera RTSP Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            if processed_count % max(args.log_every, 1) == 0:
                print(
                    f"t={elapsed:5.1f}s frame={packet.frame_index:5d} "
                    f"status={stream_status} people={len(last_people)} "
                    f"p_fight={current_smooth_prob:.3f} "
                    f"lat={latency_ms:.0f}ms cap_fps={snapshot['capture_fps']:.1f} "
                    f"infer_fps={inference_fps:.1f} drops={snapshot['dropped_frames']}",
                    flush=True,
                )

            processed_count += 1

    except KeyboardInterrupt:
        print("Interrupted; stopping.")
        stop_event.set()
    finally:
        stop_event.set()
        capture.join(timeout=2.0)
        if writer:
            writer.release()
            print(f"Saved annotated video: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    prediction_fields = [
        "time_sec",
        "capture_frame",
        "processed_frame",
        "stream_status",
        "fight_label",
        "p_fight_raw",
        "p_fight_smooth",
        "people_count",
        "person_id",
        "person_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "latency_ms",
        "capture_fps",
        "inference_fps",
        "queue_size",
        "dropped_frames",
        "reconnect_count",
    ]
    health_fields = [
        "time_sec",
        "stream_status",
        "capture_fps",
        "inference_fps",
        "latency_ms",
        "last_frame_age_ms",
        "queue_size",
        "dropped_frames",
        "reconnect_count",
        "frames_read",
        "frames_processed",
    ]
    write_csv(args.csv, prediction_rows, prediction_fields, "predictions CSV")
    write_csv(args.health_csv, health_rows, health_fields, "health CSV")

    snapshot = stats.snapshot()
    print(f"Processed frames: {processed_count}")
    print(f"Captured frames: {snapshot['frames_read']}")
    print(f"Dropped frames: {snapshot['dropped_frames']}")
    print(f"Reconnects: {snapshot['reconnect_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming RTSP/webcam safety monitor")
    parser.add_argument("--source", default="0", help="Webcam index, RTSP/HTTP URL, or video path")
    parser.add_argument("--fight-model", default="models/fight_detector.pth", help="Fight model checkpoint")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--output", default=None, help="Optional annotated output video")
    parser.add_argument("--csv", default=None, help="Optional predictions CSV")
    parser.add_argument("--health-csv", default=None, help="Optional stream health CSV")
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds; 0 means forever/file EOF")
    parser.add_argument("--queue-size", type=int, default=2, help="Max queued frames; old frames are dropped")
    parser.add_argument("--queue-timeout", type=float, default=0.5, help="Seconds to wait for a frame")
    parser.add_argument("--stale-after", type=float, default=2.0, help="Seconds without frames before STALE")
    parser.add_argument("--reconnect", action=argparse.BooleanOptionalAction, default=True, help="Reconnect on stream read failures")
    parser.add_argument("--reconnect-initial-delay", type=float, default=1.0, help="Initial reconnect delay")
    parser.add_argument("--reconnect-max-delay", type=float, default=5.0, help="Max reconnect delay")
    parser.add_argument("--pace-file", action=argparse.BooleanOptionalAction, default=True, help="Read local files at source FPS")
    parser.add_argument("--file-fps", type=float, default=25.0, help="Fallback FPS for local files")
    parser.add_argument("--output-fps", type=float, default=25.0, help="Output video FPS")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per fight model window")
    parser.add_argument("--fight-stride", type=int, default=8, help="Fight prediction stride over processed frames")
    parser.add_argument("--smooth-window", type=int, default=3, help="Fight probability smoothing window")
    parser.add_argument("--threshold", type=float, default=0.55, help="Fight alert threshold")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO person confidence threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--yolo-every", type=int, default=1, help="Run YOLO every N processed frames")
    parser.add_argument("--track", action="store_true", help="Use YOLO tracking IDs")
    parser.add_argument(
        "--fight-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device for the fight model",
    )
    parser.add_argument(
        "--yolo-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for YOLO",
    )
    parser.add_argument("--health-interval", type=float, default=1.0, help="Seconds between health CSV rows")
    parser.add_argument("--log-every", type=int, default=30, help="Print every N processed frames")
    parser.add_argument("--no-display", action="store_true", help="Do not open preview window")

    run(parser.parse_args())
