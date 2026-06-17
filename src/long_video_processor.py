"""
Stage 2: Long Video Processing with Temporal Smoothing

Takes long video → processes with sliding window → outputs:
- Annotated video with alerts
- CSV log of events
"""

import cv2
import torch
import numpy as np
import csv
import logging
import sys
from pathlib import Path
from collections import deque
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


class LongVideoProcessor:
    """Process long video with sliding window and temporal smoothing"""

    def __init__(self, model, device='cpu', num_frames=16, stride=8):
        self.model = model
        self.device = device
        self.num_frames = num_frames
        self.stride = stride  # Window overlap (smaller = more frequent predictions)

        self.logger = logging.getLogger('LongVideoProcessor')

    def process_video(
        self,
        video_path,
        output_path=None,
        smooth_window=5,
        threshold=0.55,
        min_consecutive=2,
    ):
        """
        Process video with temporal smoothing

        Args:
            video_path: Input video file
            output_path: Output video file (optional)
            smooth_window: Number of predictions to average for smoothing
            threshold: Alert if smoothed_prob > threshold
            min_consecutive: Minimum consecutive windows to trigger alert
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {Path(video_path).name}")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        duration = total_frames / fps if fps else 0
        print(f"  Duration: {duration:.1f}s")
        print(f"  Window: {self.num_frames} frames, stride: {self.stride}")
        print(f"  Smoothing: {smooth_window}, threshold: {threshold:.2f}, min consecutive: {min_consecutive}")

        # Video writer
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        else:
            out = None

        # Frame buffer and predictions
        frame_buffer = deque(maxlen=self.num_frames)
        predictions = deque(maxlen=smooth_window)

        frame_count = 0
        current_event = None
        consecutive_alerts = 0
        events = []

        pbar = tqdm(total=total_frames, desc='Processing')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add frame to buffer
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frame_buffer.append(frame_norm)

            # Make prediction every stride frames
            if len(frame_buffer) == self.num_frames and frame_count % self.stride == 0:
                frames_np = np.stack(list(frame_buffer)).astype(np.float32)
                frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(frames_tensor)
                    probs = torch.softmax(logits, dim=1)
                    fight_prob = probs[0, 1].item()  # Class 1 = fight

                predictions.append(fight_prob)

            # Temporal smoothing
            label = "Normal"
            color = (0, 255, 0)

            if len(predictions) == smooth_window:
                smoothed_prob = np.mean(predictions)

                if smoothed_prob > threshold:
                    consecutive_alerts += 1
                    label = f"FIGHT ({smoothed_prob:.2f})"
                    color = (0, 0, 255)

                    # Start new event or update existing
                    if current_event is None and consecutive_alerts >= min_consecutive:
                        current_event = {
                            'start_frame': frame_count,
                            'start_time': frame_count / fps if fps else 0,
                            'max_confidence': smoothed_prob
                        }
                    elif current_event is not None:
                        current_event['max_confidence'] = max(
                            current_event['max_confidence'], smoothed_prob
                        )
                else:
                    # End event if confidence drops
                    if current_event is not None and consecutive_alerts >= min_consecutive:
                        current_event['end_frame'] = frame_count
                        current_event['end_time'] = frame_count / fps if fps else 0
                        current_event['duration'] = current_event['end_time'] - current_event['start_time']
                        events.append(current_event)
                        current_event = None

                    consecutive_alerts = 0
                    label = f"Normal ({smoothed_prob:.2f})"
                    color = (0, 255, 0)

            # Draw on frame
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            if 'FIGHT' in label:
                cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)

            # Write to output
            if out:
                out.write(frame)

            frame_count += 1
            pbar.update(1)

        pbar.close()

        # Handle last event
        if current_event is not None:
            current_event['end_frame'] = frame_count
            current_event['end_time'] = total_frames / fps if fps else 0
            current_event['duration'] = current_event['end_time'] - current_event['start_time']
            events.append(current_event)

        cap.release()
        if out:
            out.release()

        # Save events CSV, including empty files so the run is auditable.
        if output_path:
            csv_path = str(output_path).replace('.mp4', '_events.csv')
            with open(csv_path, 'w') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['start_time', 'end_time', 'duration', 'max_confidence']
                )
                writer.writeheader()
                for event in events:
                    writer.writerow({
                        'start_time': f"{event['start_time']:.2f}",
                        'end_time': f"{event['end_time']:.2f}",
                        'duration': f"{event['duration']:.2f}",
                        'max_confidence': f"{event['max_confidence']:.2f}",
                    })
            print(f"  Events saved to: {csv_path}")

        # Summary
        print(f"\n✓ Processed {total_frames} frames")
        print(f"✓ Found {len(events)} fight events")
        if output_path:
            print(f"✓ Output saved to: {output_path}")

        return events


def select_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


if __name__ == '__main__':
    import argparse
    from src.model import FightDetector

    parser = argparse.ArgumentParser(description='Process long video with temporal smoothing')
    parser.add_argument('video', type=str, help='Input video path')
    parser.add_argument('--model', type=str, default='models/fight_detector.pth',
                        help='Model path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Smoothing window size')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Fight detection threshold')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Frames per window')
    parser.add_argument('--stride', type=int, default=8,
                        help='Sliding-window stride in frames')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Inference device')

    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}\n")

    # Load model
    model = FightDetector(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Process video
    processor = LongVideoProcessor(
        model,
        device=device,
        num_frames=args.num_frames,
        stride=args.stride,
    )
    processor.process_video(
        args.video,
        output_path=args.output,
        smooth_window=args.smooth_window,
        threshold=args.threshold,
    )
