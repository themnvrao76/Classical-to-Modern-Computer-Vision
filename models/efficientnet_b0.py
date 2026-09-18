import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, channels, 1)
        self.act = Swish()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.act(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        expanded = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                Swish(),
            ])

        layers.extend([
            nn.Conv2d(expanded, expanded, kernel_size, stride, kernel_size // 2,
                      groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            Swish(),
            SqueezeExcitation(expanded, max(1, in_channels // 4)),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetB0(nn.Module):
    config = [
        (1, 3, 1, 32, 16, 1),
        (6, 3, 2, 16, 24, 2),
        (6, 5, 2, 24, 40, 2),
        (6, 3, 2, 40, 80, 3),
        (6, 5, 1, 80, 112, 3),
        (6, 5, 2, 112, 192, 4),
        (6, 3, 1, 192, 320, 1),
    ]

    def __init__(self, num_classes=1000, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
        )

        blocks = []
        for expand, kernel, stride, in_channels, out_channels, repeats in self.config:
            for i in range(repeats):
                blocks.append(MBConv(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel,
                    stride if i == 0 else 1,
                    expand,
                ))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def efficientnet_b0(num_classes=1000):
    return EfficientNetB0(num_classes=num_classes)


if __name__ == "__main__":
    model = efficientnet_b0()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(y.shape)
    print(f"Parameters: {params:,}")
