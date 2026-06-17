"""
Live webcam/stream fight detection.

Reads frames from a webcam index or stream URL, keeps a rolling 16-frame
buffer, runs the trained R3D-18 classifier, and writes an optional annotated
video demo.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FightDetector


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_source(source):
    return int(source) if source.isdigit() else source


class LiveFightDetector:
    def __init__(self, model_path, device, num_frames=16, threshold=0.55):
        self.device = device
        self.num_frames = num_frames
        self.threshold = threshold

        self.model = FightDetector(num_classes=2, pretrained=False).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict_buffer(self, frame_buffer):
        frames_np = np.stack(frame_buffer).astype(np.float32)
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(frames_tensor)
            probs = torch.softmax(logits, dim=1)
            fight_prob = probs[0, 1].item()

        return fight_prob


def draw_status(frame, label, fight_prob, fps, threshold):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if label == "FIGHT" else (0, 255, 0)
    font_scale = max(0.45, min(0.9, w / 900.0))
    thickness = max(1, int(round(font_scale * 3)))

    cv2.putText(
        frame,
        f"{label} p_fight={fight_prob:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(
        frame,
        f"fps={fps:.1f} threshold={threshold:.2f}",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )
    if label == "FIGHT":
        cv2.rectangle(frame, (8, 8), (w - 8, h - 8), color, 3)


def run_live(args):
    device = select_device(args.device)
    print(f"Device: {device}")
    print(f"Source: {args.source}")

    detector = LiveFightDetector(
        args.model,
        device=device,
        num_frames=args.num_frames,
        threshold=args.threshold,
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera/stream. On macOS, allow Camera access for "
            "Terminal/Python in System Settings -> Privacy & Security -> Camera."
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
    predictions = deque(maxlen=args.smooth_window)
    rows = []

    start_time = time.time()
    last_tick = start_time
    frame_count = 0
    prediction_count = 0
    current_label = "Collecting"
    current_prob = 0.0
    effective_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received; stopping.")
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
            frame_norm = rgb.astype(np.float32) / 255.0
            frame_buffer.append(frame_norm)

            if len(frame_buffer) == args.num_frames and frame_count % args.stride == 0:
                fight_prob = detector.predict_buffer(list(frame_buffer))
                predictions.append(fight_prob)
                smoothed = float(np.mean(predictions))
                current_prob = smoothed
                current_label = "FIGHT" if smoothed >= args.threshold else "Normal"
                prediction_count += 1

                rows.append({
                    "time_sec": f"{elapsed:.2f}",
                    "frame": frame_count,
                    "p_fight": f"{fight_prob:.4f}",
                    "smoothed_p_fight": f"{smoothed:.4f}",
                    "label": current_label,
                })

                print(
                    f"t={elapsed:5.1f}s frame={frame_count:5d} "
                    f"p_fight={fight_prob:.3f} smooth={smoothed:.3f} label={current_label}",
                    flush=True,
                )

            draw_status(frame, current_label, current_prob, effective_fps, args.threshold)

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Safe Camera Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved video: {args.output}")
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            fieldnames = ["time_sec", "frame", "p_fight", "smoothed_p_fight", "label"]
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        print(f"Saved predictions: {csv_path}")

    print(f"Processed frames: {frame_count}")
    print(f"Predictions: {prediction_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam/stream fight detection")
    parser.add_argument("--source", default="0", help="Webcam index or RTSP/HTTP URL")
    parser.add_argument("--model", default="models/fight_detector.pth", help="Model path")
    parser.add_argument("--output", default=None, help="Optional annotated output video")
    parser.add_argument("--csv", default=None, help="Optional prediction CSV")
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per model window")
    parser.add_argument("--stride", type=int, default=8, help="Prediction stride in frames")
    parser.add_argument("--smooth-window", type=int, default=3, help="Smoothing window")
    parser.add_argument("--threshold", type=float, default=0.55, help="Fight alert threshold")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device",
    )
    parser.add_argument("--no-display", action="store_true", help="Do not open preview window")

    run_live(parser.parse_args())
