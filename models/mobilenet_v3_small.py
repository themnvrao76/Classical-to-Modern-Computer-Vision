import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, expand_ch, use_se, use_hs):
        super().__init__()
        act = HSwish if use_hs else nn.ReLU
        layers = []
        if expand_ch != in_ch:
            layers += [
                nn.Conv2d(in_ch, expand_ch, 1, bias=False),
                nn.BatchNorm2d(expand_ch),
                act(),
            ]
        layers += [
            nn.Conv2d(expand_ch, expand_ch, kernel, stride, kernel // 2, groups=expand_ch, bias=False),
            nn.BatchNorm2d(expand_ch),
            act(),
        ]
        if use_se:
            layers.append(SqueezeExcitation(expand_ch))
        layers += [
            nn.Conv2d(expand_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish(),
        )
        cfg = [
            (16, 16, 3, 2, 16, True, False),
            (16, 24, 3, 2, 72, False, False),
            (24, 24, 3, 1, 88, False, False),
            (24, 40, 5, 2, 96, True, True),
            (40, 40, 5, 1, 240, True, True),
            (40, 40, 5, 1, 240, True, True),
            (40, 48, 5, 1, 120, True, True),
            (48, 48, 5, 1, 144, True, True),
            (48, 96, 5, 2, 288, True, True),
            (96, 96, 5, 1, 576, True, True),
            (96, 96, 5, 1, 576, True, True),
        ]
        self.blocks = nn.Sequential(*[InvertedResidual(*c) for c in cfg])
        self.head = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = MobileNetV3Small()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("output:", tuple(y.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
