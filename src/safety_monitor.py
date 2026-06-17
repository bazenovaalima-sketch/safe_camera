"""
Combined live safety monitor.

Runs two models on the same webcam, RTSP stream, or video file:
- YOLO detects/tracks people in each frame.
- R3D-18 estimates fight probability from a rolling 16-frame window.

The output video shows person boxes plus the current fight status. The CSV
combines frame-level fight scores with person-level detections.
"""

import argparse
import csv
import sys
import time
from collections import deque
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


PERSON_CLASS_ID = 0


def parse_source(source):
    return int(source) if source.isdigit() else source


def select_torch_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device(name):
    return None if name == "auto" else name


class RollingFightDetector:
    def __init__(self, model_path, device, num_frames):
        self.device = device
        self.num_frames = num_frames
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


def clamp_box(xyxy, width, height):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    return [
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    ]


def draw_label_box(frame, xyxy, label, conf):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(xyxy, w, h)
    color = (40, 220, 80)

    font_scale = max(0.45, min(0.75, w / 1100.0))
    thickness = max(1, int(round(font_scale * 3)))
    text = f"{label} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, text_y1), (min(x1 + tw + 8, w - 1), y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, max(th + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_status(frame, fight_label, fight_prob, raw_prob, people_count, fps, threshold):
    h, w = frame.shape[:2]
    alert = fight_label == "FIGHT"
    fight_color = (0, 0, 255) if alert else (40, 220, 80)
    people_color = (40, 220, 80)
    font_scale = max(0.48, min(0.9, w / 950.0))
    thickness = max(1, int(round(font_scale * 3)))

    lines = [
        (f"{fight_label} smooth={fight_prob:.2f} raw={raw_prob:.2f}", fight_color),
        (f"people={people_count} fps={fps:.1f} threshold={threshold:.2f}", people_color),
    ]

    y = 30
    for line, color in lines:
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
        y += int(30 * font_scale) + 10

    if alert:
        cv2.rectangle(frame, (8, 8), (w - 8, h - 8), fight_color, 3)


def extract_people(result):
    boxes = result.boxes
    detections = []
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    for det_idx, box in enumerate(xyxy):
        person_id = int(ids[det_idx]) if ids is not None else ""
        detections.append({
            "person_id": person_id,
            "confidence": float(confs[det_idx]),
            "box": box,
        })
    return detections


def append_csv_rows(rows, elapsed, frame_idx, fight_label, raw_prob, smooth_prob, people, width, height):
    base = {
        "time_sec": f"{elapsed:.2f}",
        "frame": frame_idx,
        "fight_label": fight_label,
        "p_fight_raw": f"{raw_prob:.4f}",
        "p_fight_smooth": f"{smooth_prob:.4f}",
        "people_count": len(people),
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


def run(args):
    fight_device = select_torch_device(args.fight_device)
    print(f"Fight model device: {fight_device}")
    print(f"YOLO device: {args.yolo_device}")
    print(f"Source: {args.source}")

    fight_detector = RollingFightDetector(args.fight_model, fight_device, args.num_frames)
    yolo_model = YOLO(args.yolo_model)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera/video. For macOS webcam access, allow Camera "
            "for Terminal/Python in System Settings -> Privacy & Security -> Camera."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps if input_fps and input_fps > 1 else 25.0

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    frame_buffer = deque(maxlen=args.num_frames)
    fight_history = deque(maxlen=args.smooth_window)
    rows = []

    start_time = time.time()
    last_tick = start_time
    frame_idx = 0
    current_raw_prob = 0.0
    current_smooth_prob = 0.0
    current_fight_label = "Collecting"
    effective_fps = 0.0
    last_people = []
    fight_prediction_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            elapsed = now - start_time
            if args.duration and elapsed >= args.duration:
                break

            if now > last_tick:
                effective_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now

            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frame_buffer.append(rgb.astype(np.float32) / 255.0)

            if len(frame_buffer) == args.num_frames and frame_idx % args.fight_stride == 0:
                current_raw_prob = fight_detector.predict(list(frame_buffer))
                fight_history.append(current_raw_prob)
                current_smooth_prob = float(np.mean(fight_history))
                current_fight_label = "FIGHT" if current_smooth_prob >= args.threshold else "Normal"
                fight_prediction_count += 1

            if frame_idx % max(args.yolo_every, 1) == 0:
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

            draw_status(
                frame,
                current_fight_label,
                current_smooth_prob,
                current_raw_prob,
                len(last_people),
                effective_fps,
                args.threshold,
            )

            append_csv_rows(
                rows,
                elapsed,
                frame_idx,
                current_fight_label,
                current_raw_prob,
                current_smooth_prob,
                last_people,
                width,
                height,
            )

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Safe Camera Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % max(args.log_every, 1) == 0:
                print(
                    f"t={elapsed:5.1f}s frame={frame_idx:5d} "
                    f"people={len(last_people)} p_fight={current_smooth_prob:.3f} "
                    f"label={current_fight_label}",
                    flush=True,
                )

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved annotated video: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "time_sec",
            "frame",
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
        ]
        with csv_path.open("w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        print(f"Saved combined CSV: {csv_path}")

    print(f"Processed frames: {frame_idx}")
    print(f"Fight predictions: {fight_prediction_count}")
    print(f"CSV rows: {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined YOLO + fight detection monitor")
    parser.add_argument("--source", default="0", help="Webcam index, RTSP/HTTP URL, or video path")
    parser.add_argument("--fight-model", default="models/fight_detector.pth", help="Fight model checkpoint")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--output", default=None, help="Optional annotated output video")
    parser.add_argument("--csv", default=None, help="Optional combined CSV")
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds; 0 means full video")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per fight model window")
    parser.add_argument("--fight-stride", type=int, default=8, help="Fight prediction stride in frames")
    parser.add_argument("--smooth-window", type=int, default=3, help="Fight probability smoothing window")
    parser.add_argument("--threshold", type=float, default=0.55, help="Fight alert threshold")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO person confidence threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--yolo-every", type=int, default=1, help="Run YOLO every N frames")
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
    parser.add_argument("--log-every", type=int, default=30, help="Print every N frames")
    parser.add_argument("--no-display", action="store_true", help="Do not open preview window")

    run(parser.parse_args())
