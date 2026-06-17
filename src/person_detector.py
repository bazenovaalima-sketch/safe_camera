"""
YOLO person detection/tracking for webcam, RTSP, or video files.

The default Ultralytics YOLO model is pretrained on COCO, where class 0 is
`person`. This script only keeps that class and writes optional annotations
and a CSV with detected boxes.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with `pip install ultralytics` "
        "or run `pip install -r requirements.txt`."
    ) from exc


PERSON_CLASS_ID = 0


def parse_source(source):
    return int(source) if source.isdigit() else source


def yolo_device(device):
    return None if device == "auto" else device


def draw_box(frame, xyxy, label, conf):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

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


def draw_header(frame, person_count, fps, mode):
    h, w = frame.shape[:2]
    font_scale = max(0.5, min(0.85, w / 1000.0))
    thickness = max(1, int(round(font_scale * 3)))
    text = f"YOLO {mode} | people={person_count} | fps={fps:.1f}"
    cv2.putText(
        frame,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (30, 30, 30),
        thickness,
        cv2.LINE_AA,
    )


def run(args):
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
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

    rows = []
    start_time = time.time()
    last_tick = start_time
    frame_idx = 0
    effective_fps = 0.0
    mode = "track" if args.track else "detect"

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

            predict_kwargs = {
                "classes": [PERSON_CLASS_ID],
                "conf": args.conf,
                "imgsz": args.imgsz,
                "device": yolo_device(args.device),
                "verbose": False,
            }
            if args.track:
                results = model.track(frame, persist=True, **predict_kwargs)
            else:
                results = model.predict(frame, **predict_kwargs)

            result = results[0]
            boxes = result.boxes
            person_count = 0

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = None
                if getattr(boxes, "id", None) is not None:
                    ids = boxes.id.cpu().numpy().astype(int)

                for det_idx, box in enumerate(xyxy):
                    person_count += 1
                    person_id = int(ids[det_idx]) if ids is not None else ""
                    label = f"person {person_id}" if person_id != "" else "person"
                    conf = float(confs[det_idx])
                    draw_box(frame, box, label, conf)
                    rows.append({
                        "time_sec": f"{elapsed:.2f}",
                        "frame": frame_idx,
                        "person_id": person_id,
                        "confidence": f"{conf:.4f}",
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                    })

            draw_header(frame, person_count, effective_fps, mode)

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Safe Camera YOLO People", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % max(args.log_every, 1) == 0:
                print(
                    f"t={elapsed:5.1f}s frame={frame_idx:5d} "
                    f"people={person_count}",
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
        with csv_path.open("w", newline="") as f:
            fieldnames = ["time_sec", "frame", "person_id", "confidence", "x1", "y1", "x2", "y2"]
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(rows)
        print(f"Saved detections CSV: {csv_path}")

    print(f"Processed frames: {frame_idx}")
    print(f"Person detections written: {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO person detector/tracker")
    parser.add_argument("--source", default="0", help="Webcam index, RTSP/HTTP URL, or video path")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Ultralytics YOLO model")
    parser.add_argument("--output", default=None, help="Optional annotated output video")
    parser.add_argument("--csv", default=None, help="Optional detections CSV")
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="YOLO inference device",
    )
    parser.add_argument("--track", action="store_true", help="Use YOLO tracking IDs")
    parser.add_argument("--log-every", type=int, default=30, help="Print every N frames")
    parser.add_argument("--no-display", action="store_true", help="Do not open preview window")

    run(parser.parse_args())
