# Safe Camera Fight Detection

Safe Camera is a video-surveillance prototype for detecting possible fights in camera footage.

It combines:

- an R3D-18 action-recognition model for `Fight` / `NonFight` classification;
- YOLO person detection and temporary person tracking boxes;
- a streaming monitor with a bounded frame queue, reconnect logic, FPS, latency, dropped-frame, and health metrics.

This is a research/prototype system, not a production safety product.

## Demo

Combined person detection + fight detection snapshot:

This example shows the combined monitor: YOLO draws green temporary person tracking boxes while the fight model reports the smoothed fight probability. In this 150-frame demo, all 134 scored frames were labeled `FIGHT`, with average smoothed `p_fight=0.800` and max smoothed `p_fight=0.877`.

![Combined person detection and fight detection snapshot](docs/assets/true_positive_snapshot.png)

Combined annotated demo video:

[Download/watch the MP4 demo](docs/assets/true_positive_demo.mp4)

## Current Results

The first baseline was trained on the RWF-2000 `train/` split with a frozen R3D-18 backbone and a newly trained binary classification head. Final evaluation was done on the official untouched RWF-2000 `val/` split.

| Metric | Value |
|---|---:|
| Official val videos | 400 |
| Accuracy | 74.5% |
| Fight precision | 71.9% |
| Fight recall | 80.5% |
| Fight F1 | 75.9% |
| Fight class accuracy | 80.5% |
| NonFight class accuracy | 68.5% |

Confusion matrix on the official validation split:

```text
                Pred NonFight   Pred Fight
True NonFight       137            63
True Fight           39           161
```

Training curves:

![Training curves](docs/assets/training_curves.png)

Official validation confusion matrix:

![Official validation confusion matrix](docs/assets/official_val_confusion_matrix.png)

## Pipeline

```text
Camera / video / RTSP stream
        |
        v
Capture thread with reconnect + bounded frame queue
        |
        v
YOLO person detection/tracking      R3D-18 fight classifier
        |                            |
        +------------+---------------+
                     v
Annotated video + predictions CSV + health CSV
```

The person IDs are temporary tracking IDs such as `person 1` and `person 2`. They are not face recognition and do not identify real people.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Large files are intentionally not stored in the repository:

- RWF-2000 dataset files;
- trained `.pth` fight model weights;
- generated videos and CSV outputs;
- downloaded YOLO weights such as `yolo11n.pt`.

Dataset source:

- RWF-2000 on Kaggle: https://www.kaggle.com/datasets/vulamnguyen/rwf2000?select=RWF-2000

Expected local RWF-2000 layout:

```text
data/rwf2000/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

## Recommended Run: Streaming Monitor

`src/rtsp_monitor.py` is the most complete runtime path. It reads frames in a separate capture thread, keeps only a small queue of fresh frames, drops old frames when inference is slower than the camera, and records stream health metrics such as FPS, latency, dropped frames, and reconnect count.

You can test it without an IP camera by using a local video file or the laptop webcam. Later, the same command accepts an RTSP URL.

Local video file:

```bash
python src/rtsp_monitor.py \
  --source docs/assets/true_positive_demo.mp4 \
  --fight-model models/fight_detector.pth \
  --output results/rtsp_file_demo.mp4 \
  --csv results/rtsp_file_predictions.csv \
  --health-csv results/rtsp_file_health.csv \
  --duration 0 \
  --fight-device auto \
  --yolo-device auto \
  --track \
  --no-display
```

Laptop webcam:

```bash
python src/rtsp_monitor.py \
  --source 0 \
  --fight-model models/fight_detector.pth \
  --output results/rtsp_webcam_demo.mp4 \
  --csv results/rtsp_webcam_predictions.csv \
  --health-csv results/rtsp_webcam_health.csv \
  --duration 20 \
  --fight-device auto \
  --yolo-device auto \
  --track
```

Future IP camera / RTSP stream:

```bash
python src/rtsp_monitor.py \
  --source "rtsp://user:password@192.168.1.10:554/stream1" \
  --fight-model models/fight_detector.pth \
  --output results/rtsp_camera_demo.mp4 \
  --csv results/rtsp_camera_predictions.csv \
  --health-csv results/rtsp_camera_health.csv \
  --duration 0 \
  --track
```

The overlay shows stream health, for example:

```text
LIVE cap=25.0 infer=8.2 lat=120ms drop=14 rec=0
```

## Other Runtime Modes

Combined YOLO + fight monitor without the threaded streaming layer:

```bash
python src/safety_monitor.py \
  --source 0 \
  --fight-model models/fight_detector.pth \
  --output results/safety_monitor.mp4 \
  --csv results/safety_monitor.csv \
  --duration 10 \
  --fight-device auto \
  --yolo-device auto \
  --track
```

YOLO person detection only:

```bash
python src/person_detector.py \
  --source 0 \
  --output results/webcam_people.mp4 \
  --csv results/webcam_people.csv \
  --duration 10 \
  --device auto \
  --track
```

Fight detection only on webcam or stream:

```bash
python src/webcam_processor.py \
  --source 0 \
  --model models/fight_detector.pth \
  --output results/webcam_fight_demo.mp4 \
  --csv results/webcam_fight_demo.csv \
  --duration 10 \
  --num-frames 16 \
  --stride 8 \
  --smooth-window 3 \
  --threshold 0.55 \
  --device auto
```

Single-video annotated inference:

```bash
python src/inference.py \
  data/rwf2000/RWF-2000/val/Fight/example.avi \
  --model models/fight_detector.pth \
  --output results/example_annotated.mp4 \
  --num-frames 16 \
  --stride 8 \
  --threshold 0.5 \
  --no-display
```

Long-video sliding-window processing:

```bash
python src/long_video_processor.py \
  input_video.mp4 \
  --model models/fight_detector.pth \
  --output results/input_video_annotated.mp4 \
  --num-frames 16 \
  --stride 8 \
  --threshold 0.55 \
  --smooth-window 5
```

This writes an annotated video and an `_events.csv` file with detected fight intervals.

## Training

```bash
python src/train_rwf.py \
  --data-dir data/rwf2000/RWF-2000 \
  --model-dir models \
  --epochs 20 \
  --batch-size 8 \
  --num-frames 16
```

The default mode freezes the pretrained R3D-18 backbone and trains the final classification layer.

## Project Structure

```text
docs/assets/                # Curated public result images and demo video
src/
├── dataset_rwf.py           # RWF-2000 video dataset
├── model.py                 # R3D-18 binary classifier
├── train_rwf.py             # Training loop
├── inference.py             # Annotated single-video inference
├── long_video_processor.py  # Sliding-window smoothing and event export
├── webcam_processor.py      # Live webcam/stream fight detection
├── person_detector.py       # YOLO person boxes and anonymous tracking IDs
├── safety_monitor.py        # Combined YOLO people + fight probability monitor
└── rtsp_monitor.py          # Streaming monitor with reconnect, frame queue, health metrics
```

## Notes

- Thresholds should be calibrated for the target camera domain and alert tolerance.
- The current model is a baseline trained on RWF-2000; real camera deployment needs more validation on the target environment.
- Person tracking IDs are temporary object-track IDs, not identity recognition.
