import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        super().__init__(*layers)


class FCN8s(nn.Module):
    """FCN-8s with a VGG-16-style encoder and learned skip fusion."""

    def __init__(self, num_classes=21, dropout=0.5):
        super().__init__()
        self.block1 = VGGBlock(3, 64, 2)
        self.block2 = VGGBlock(64, 128, 2)
        self.block3 = VGGBlock(128, 256, 3)
        self.block4 = VGGBlock(256, 512, 3)
        self.block5 = VGGBlock(512, 512, 3)

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.block1(x)
        x = self.block2(x)
        pool3 = self.block3(x)
        pool4 = self.block4(pool3)
        pool5 = self.block5(pool4)

        score = self.score_fr(self.classifier(pool5))
        score = F.interpolate(score, size=pool4.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_pool4(pool4)
        score = F.interpolate(score, size=pool3.shape[-2:], mode="bilinear", align_corners=False)
        score = score + self.score_pool3(pool3)
        return F.interpolate(score, size=input_size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = FCN8s(num_classes=21)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters: {params:,}")
