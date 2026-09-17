import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        features = self.dropout(self.layers(x))
        return torch.cat([x, features], dim=1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        super().__init__(*layers)


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )


class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, bn_size=4, drop_rate=0.0):
        super().__init__()
        channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks = []
        for i, num_layers in enumerate((6, 12, 24, 16)):
            blocks.append(DenseBlock(num_layers, channels, growth_rate, bn_size, drop_rate))
            channels += num_layers * growth_rate
            if i != 3:
                out_channels = channels // 2
                blocks.append(Transition(channels, out_channels))
                channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.relu(self.norm(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def densenet121(num_classes=1000, **kwargs):
    return DenseNet121(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = densenet121()
    model.eval()
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(sample)
    params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {params:,}")
