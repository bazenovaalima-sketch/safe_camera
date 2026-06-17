# Safe Camera Fight Detection

Video fight detection prototype built with a 3D ResNet-18 action-recognition model.

The project classifies short surveillance-style video clips as `Fight` or `NonFight`, then applies sliding-window temporal smoothing for longer videos.

## Features

- RWF-2000 dataset loader for `train/` and `val/` splits
- R3D-18 backbone pretrained on Kinetics-400
- Transfer learning with an optional frozen backbone
- Single-video annotated inference
- Long-video sliding-window processing with event CSV output
- CPU, CUDA, and Apple MPS device selection

## Project Structure

```text
docs/assets/                # Curated public result images and demo video
src/
├── dataset_rwf.py           # RWF-2000 video dataset
├── model.py                 # R3D-18 binary classifier
├── train_rwf.py             # Training loop
├── inference.py             # Annotated single-video inference
└── long_video_processor.py  # Sliding-window smoothing and event export
```

Large local assets are intentionally ignored: datasets, trained weights, generated videos, and experiment outputs.

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

True-positive demo snapshot:

This example was classified as `Fight` in all 17 sliding windows, with average `p_fight=0.801` and max `p_fight=0.903`.

![True-positive fight detection snapshot](docs/assets/true_positive_snapshot.png)

True-positive annotated demo video:

[Download/watch the MP4 demo](docs/assets/true_positive_demo.mp4)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected RWF-2000 layout:

```text
data/rwf2000/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

## Train

```bash
python src/train_rwf.py \
  --data-dir data/rwf2000/RWF-2000 \
  --model-dir models \
  --epochs 20 \
  --batch-size 8 \
  --num-frames 16
```

The default mode freezes the pretrained backbone and trains the final classification layer.

## Single Video Inference

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

## Long Video Processing

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

## Notes

This is a research/prototype pipeline, not a production safety system. Thresholds should be calibrated for the target camera domain and alert tolerance.
