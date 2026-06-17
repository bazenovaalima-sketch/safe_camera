import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class FightDetector(nn.Module):
    """
    3D CNN for fight detection.

    Architecture:
    - R3D-18: 3D ResNet-18 backbone (pretrained on Kinetics-400)
    - Input: (B, 3, 16, 224, 224) - batch of 16 RGB frames
    - Output: (B, num_classes) - class logits

    How it works:
    3D convolutions extract spatiotemporal features (motion + appearance)
    instead of just spatial features like 2D CNN.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load pretrained 3D ResNet (trained on large video dataset)
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r3d_18(weights=weights)

        # Replace last layer for binary classification (fight vs normal)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, 3, T, H, W) - batch of videos
               B = batch size
               3 = RGB channels
               T = 16 frames
               H, W = 224x224

        Returns:
            logits: (B, num_classes) - raw predictions
        """
        return self.backbone(x)


if __name__ == '__main__':
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = FightDetector(num_classes=2, pretrained=True).to(device)
    print(model)

    # Dummy input: batch of 2 videos
    dummy_input = torch.randn(2, 3, 16, 224, 224).to(device)
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (2, 2)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
