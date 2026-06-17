"""
Stage 1: Train Fight Detector on RWF-2000

Strategy: Transfer Learning
- Use pretrained R3D-18 (Kinetics-400) as backbone
- Freeze backbone (keep learned features)
- Train only new FC layer (2 classes: fight/normal)

Time: ~30 min on CPU, ~3 min on GPU
Expected accuracy: 85-90%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.dataset_rwf import RWF2000Dataset
from src.model import FightDetector


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def freeze_backbone(model):
    """Freeze all layers except FC"""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    # Count parameters
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nFreeze Strategy:")
    print(f"  Frozen parameters: {frozen:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Total: {frozen + trainable:,}")
    print(f"  Training only {trainable / (frozen + trainable) * 100:.2f}% of model")


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🚀 SafeCam Training')
    print(f'Device: {device}')

    # Create output directory
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Dataset
    print(f'\n📺 Loading RWF-2000 dataset...')
    train_source = RWF2000Dataset(args.data_dir, split='train', num_frames=args.num_frames, augment=True)
    val_source = RWF2000Dataset(args.data_dir, split='train', num_frames=args.num_frames, augment=False)

    # Split into train/val
    train_size = int(0.8 * len(train_source))
    val_size = len(train_source) - train_size
    train_subset, val_subset = random_split(
        range(len(train_source)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataset = Subset(train_source, train_subset.indices)
    val_dataset = Subset(val_source, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # macOS: use 0
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # Model
    print(f'\n🧠 Loading R3D-18 (pretrained on Kinetics-400)...')
    model = FightDetector(num_classes=2, pretrained=True).to(device)

    # Strategy: Freeze backbone
    if args.freeze:
        freeze_backbone(model)
        params_to_optimize = model.backbone.fc.parameters()
    else:
        # Fine-tune all
        print(f"\nFine-tune Strategy:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Training ALL parameters: {total_params:,}")
        params_to_optimize = model.parameters()

    # Optimizer & Loss
    optimizer = optim.Adam(params_to_optimize, lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    print(f'\n📊 Training Configuration:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Learning rate: {args.lr}')
    print(f'  Num frames: {args.num_frames}')

    best_val_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    print(f'\n🔥 Starting training...\n')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Schedule
        scheduler.step()

        # Print
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')

        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model_path = Path(args.model_dir) / 'fight_detector.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  ✅ Model saved (best val acc: {best_val_acc:.4f})\n')
        else:
            print()

    # Save training history
    history_path = Path(args.model_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Summary
    print('=' * 60)
    print(f'✅ Training Complete!')
    print(f'   Best model at epoch {best_epoch}')
    print(f'   Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
    print(f'   Model saved to: {Path(args.model_dir) / "fight_detector.pth"}')
    print(f'   History saved to: {history_path}')
    print('=' * 60)

    # Next steps
    print(f'\n📺 Next: Test on videos')
    print(f'   python src/inference.py <video_path>')
    print(f'   python src/long_video_processor.py <video_path> --output out.mp4\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fight detection model on RWF-2000')

    parser.add_argument('--data-dir', type=str, default='data/rwf2000/RWF-2000',
                        help='Path to RWF-2000 root directory')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('--freeze', action=argparse.BooleanOptionalAction, default=True,
                        help='Freeze backbone (only train FC layer)')

    args = parser.parse_args()
    main(args)
