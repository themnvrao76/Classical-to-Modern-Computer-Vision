import torch
import torch.nn as nn


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_channels, kernel_size=1))
        layers.extend([
            ConvBNReLU6(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2):
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        features = [ConvBNReLU6(3, input_channel, stride=2)]
        for expand_ratio, channels, repeats, stride in settings:
            output_channel = int(channels * width_mult)
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, block_stride, expand_ratio))
                input_channel = output_channel
        features.append(ConvBNReLU6(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_channel, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def mobilenet_v2(num_classes=1000, **kwargs):
    return MobileNetV2(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = mobilenet_v2()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    parameters = sum(p.numel() for p in model.parameters())
    print(y.shape)
    print(f"Parameters: {parameters:,}")
