import cv2
import torch
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FightDetector


class FightDetectionPipeline:
    """Detect fights in video in real-time"""

    def __init__(self, model_path, device='cpu', num_frames=16, threshold=0.5):
        self.device = device
        self.num_frames = num_frames
        self.threshold = threshold

        # Load model
        self.model = FightDetector(num_classes=2, pretrained=False)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()

        self.class_names = ['Normal', 'FIGHT!']
        self.colors = [(0, 255, 0), (0, 0, 255)]  # Green, Red

    def predict_video(self, video_path, output_path=None, stride=8, display=True):
        """
        Process video and detect fights.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            stride: Process every N frames (higher = faster but less frequent predictions)
            display: Show OpenCV preview window
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error: Could not open {video_path}')
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'Video: {Path(video_path).name}')
        print(f'  Resolution: {w}x{h}, FPS: {fps:.2f}, Total frames: {total_frames}')
        print(f'  Window: {self.num_frames} frames, stride: {stride}, threshold: {self.threshold:.2f}')

        # Output video writer
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        else:
            out = None

        # Frame buffer for fixed-size clip window
        frame_buffer = []
        frame_count = 0
        last_prediction = None
        last_pred_frame = -stride
        fight_probs = []
        fight_windows = 0
        total_windows = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prepare frame for buffer
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb / 255.0
            frame_buffer.append(frame_norm)

            # Make prediction every 'stride' frames
            if len(frame_buffer) >= self.num_frames and (frame_count - last_pred_frame) >= stride:
                frames_to_predict = frame_buffer[-self.num_frames:]

                # Convert to tensor
                frames_np = np.stack(frames_to_predict).astype(np.float32)
                frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
                frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

                # Predict
                with torch.no_grad():
                    logits = self.model(frames_tensor)
                    probs = torch.softmax(logits, dim=1)
                    fight_prob = probs[0, 1].item()
                    pred_class = 1 if fight_prob >= self.threshold else 0
                    confidence = fight_prob if pred_class == 1 else 1.0 - fight_prob

                last_prediction = (pred_class, confidence, fight_prob)
                last_pred_frame = frame_count
                fight_probs.append(fight_prob)
                total_windows += 1
                if pred_class == 1:
                    fight_windows += 1

            # Draw prediction on frame
            if last_prediction:
                pred_class, confidence, fight_prob = last_prediction
                label = self.class_names[pred_class]
                color = self.colors[pred_class]
                font_scale = max(0.45, min(1.0, w / 900.0))
                thickness = max(1, int(round(font_scale * 3)))

                # Draw label
                cv2.putText(
                    frame,
                    f'{label} conf={confidence:.2f}',
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness
                )
                cv2.putText(
                    frame,
                    f'p_fight={fight_prob:.2f}',
                    (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness
                )

                # Draw border if fight detected
                if pred_class == 1:
                    cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 4)
            else:
                cv2.putText(
                    frame,
                    f'Collecting {self.num_frames} frames...',
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, min(1.0, w / 900.0)),
                    (255, 255, 255),
                    2
                )

            # Write output video
            if out:
                out.write(frame)

            # Display
            if display:
                cv2.imshow('Fight Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

            # Limit buffer to current clip window
            if len(frame_buffer) > self.num_frames:
                frame_buffer.pop(0)

        cap.release()
        if out:
            out.release()
            print(f'Output saved to: {output_path}')

        if display:
            cv2.destroyAllWindows()

        avg_fight_prob = float(np.mean(fight_probs)) if fight_probs else 0.0
        max_fight_prob = float(np.max(fight_probs)) if fight_probs else 0.0
        fight_window_ratio = fight_windows / total_windows if total_windows else 0.0

        summary = {
            'video': str(video_path),
            'output': str(output_path) if output_path else None,
            'frames': frame_count,
            'windows': total_windows,
            'fight_windows': fight_windows,
            'fight_window_ratio': fight_window_ratio,
            'avg_fight_prob': avg_fight_prob,
            'max_fight_prob': max_fight_prob,
        }

        print('Done!')
        print(f'  Windows: {total_windows}')
        print(f'  Fight windows: {fight_windows}/{total_windows} ({fight_window_ratio:.2%})')
        print(f'  Avg p_fight: {avg_fight_prob:.3f}')
        print(f'  Max p_fight: {max_fight_prob:.3f}')

        return summary


def select_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main(args):
    device = select_device(args.device)
    print(f'Device: {device}\n')

    # Initialize pipeline
    detector = FightDetectionPipeline(
        args.model,
        device=device,
        num_frames=args.num_frames,
        threshold=args.threshold,
    )

    # Process video
    output_path = args.output if args.output else None
    detector.predict_video(
        args.video,
        output_path=output_path,
        stride=args.stride,
        display=not args.no_display,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect fights in video')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='models/fight_detector.pth',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Process every N frames')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Frames per model window')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fight probability threshold')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Inference device')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not open an OpenCV preview window')

    args = parser.parse_args()
    main(args)
